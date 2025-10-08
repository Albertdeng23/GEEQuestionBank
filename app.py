import os
import json
import base64
import logging
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --- 1. 初始化与配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
RESULT_JSON_FILE = os.path.join("result", "processed_questions.json")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 初始化OpenAI客户端
try:
    client = openai.OpenAI(api_key=API_KEY, base_url=API_URL)
except Exception as e:
    logging.error(f"初始化OpenAI客户端失败: {e}")
    exit()

# --- 2. 加载数据库和向量模型 (在服务启动时执行一次) ---
db_questions = []
db_embeddings = None
embedding_model = None
EMBEDDINGS_FILE = os.path.join("result", "db_embeddings.npy") # 确保路径一致

def load_database_and_create_embeddings():
    """
    修改后的版本：
    加载JSON题库和预计算好的向量文件。
    """
    global db_questions, db_embeddings, embedding_model
    
    logging.info("正在加载题库JSON文件...")
    if not os.path.exists(RESULT_JSON_FILE):
        logging.error(f"错误: 题库文件 {RESULT_JSON_FILE} 不存在。请先运行core.py。")
        return
    with open(RESULT_JSON_FILE, 'r', encoding='utf-8') as f:
        db_questions = json.load(f)
    logging.info(f"题库加载成功，共 {len(db_questions)} 道题目。")
    
    logging.info("正在加载预计算的向量文件...")
    if not os.path.exists(EMBEDDINGS_FILE):
        logging.error(f"错误: 向量文件 {EMBEDDINGS_FILE} 不存在。请先运行 prepare_embeddings.py。")
        return
    db_embeddings = np.load(EMBEDDINGS_FILE)
    logging.info(f"向量加载成功，形状为: {db_embeddings.shape}")

    # 检查题目数量和向量数量是否匹配
    if len(db_questions) != db_embeddings.shape[0]:
        logging.error("错误：题库中的题目数量与向量数量不匹配！请重新运行 prepare_embeddings.py。")
        # 在这种情况下，最好不要启动服务
        exit()

    logging.info("正在加载向量模型 (用于处理用户查询)...")
    # 模型仍然需要加载，但只用于对单次的用户查询进行编码，这非常快
    embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    logging.info("应用准备就绪，可以快速启动。")

# --- 3. 辅助函数 ---
def image_to_base64(image):
    """将Pillow Image对象转换为Base64字符串"""
    buffered = BytesIO()
    # 确保图片是RGB格式
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_question_from_image(base64_image: str) -> str:
    """调用VLM从图片中提取结构化题目文本"""
    prompt = """
    你是一个精准的题目文本提取器。请分析这张图片，只提取其中的核心题干和选项，并以纯文本形式返回，所有公式使用LaTeX格式。
    例如: "求函数 $f(x) = x^2$ 在 $x=1$ 处的导数。 A. 1 B. 2 C. 3 D. 4"
    不要返回JSON，不要任何解释。
    """
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20", # 根据你的模型服务商提供的模型名称修改
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"VLM提取题目失败: {e}")
        return None

# --- 4. Flask路由 ---
@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/find_similar', methods=['POST'])
def find_similar():
    """API端点：接收图片，返回相似题目"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400

    try:
        image = Image.open(file.stream)
        base64_image = image_to_base64(image)
        
        # 1. 从用户图片中提取问题文本
        query_text = extract_question_from_image(base64_image)
        if not query_text:
            return jsonify({"error": "无法从图片中识别题目"}), 500
            
        # 2. 将提取的文本转换为向量
        query_embedding = embedding_model.encode([query_text])
        
        # 3. 计算与题库向量的余弦相似度
        similarities = cosine_similarity(query_embedding, db_embeddings)[0]
        
        # 4. 找到最相似的N个题目 (这里N=3)
        top_n_indices = np.argsort(similarities)[-3:][::-1]
        
        results = []
        for idx in top_n_indices:
            result = db_questions[idx]
            result['similarity'] = round(float(similarities[idx]), 4)
            results.append(result)
            
        return jsonify(results)

    except Exception as e:
        logging.error(f"处理相似题目查找失败: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@app.route('/analyze_and_generate', methods=['POST'])
def analyze_and_generate():
    """API端点：分析知识点并生成类似题 (流式输出) - 已修复"""
    if 'file' not in request.files:
        return Response("Error: No file part", status=400)
    file = request.files['file']
    if file.filename == '':
        return Response("Error: No selected file", status=400)

    # --- 关键修复 ---
    # 在请求上下文中立即读取文件内容到内存
    try:
        image_bytes = file.read()
    except Exception as e:
        logging.error(f"读取上传文件失败: {e}")
        return Response(f"Error reading file: {e}", status=500)
    # --- 修复结束 ---

    def generate_stream(img_bytes): # <--- 接收字节数据作为参数
        try:
            # 从内存中的字节数据创建Image对象
            image = Image.open(BytesIO(img_bytes))
            base64_image = image_to_base64(image)
            
            # 步骤1: 提取题目文本
            yield "data: ### 正在识别题目...\n\n"
            question_text = extract_question_from_image(base64_image)
            if not question_text:
                yield "data: 错误：无法从图片中识别题目。\n\n"
                return
            
            yield f"data: **已识别题目:** {question_text}\n\n---\n\n"

            # 步骤2: 流式调用API分析知识点
            yield "data: ### 考点分析\n\n"
            knowledge_prompt = f"你是一个资深考研数学老师。请分析以下题目，总结其中涉及的核心考点，并对每个考点做简要说明。请直接输出分析内容。\n\n题目：{question_text}"
            stream1 = client.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=[{"role": "user", "content": knowledge_prompt}],
                stream=True
            )
            for chunk in stream1:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}"

            # 步骤3: 流式调用API生成类似题目
            yield "\n\ndata: ---\n\n### 类似题目生成 (3道)\n\n"
            similar_prompt = f"你是一个考研数学出题专家。请根据以下题目，原创3道考点相似但题目内容和数字不同的题目，并给出答案。请使用LaTeX表示所有公式。\n\n原题：{question_text}"
            stream2 = client.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=[{"role": "user", "content": similar_prompt}],
                stream=True
            )
            for chunk in stream2:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}"
            
            yield "\n\ndata: --- END OF STREAM ---"

        except Exception as e:
            logging.error(f"流式生成失败: {e}")
            # 在流中向前端报告错误
            yield f"data: \n\n<div class='alert alert-danger'>服务器内部错误: {e}</div>\n\n"
            yield "data: --- END OF STREAM ---"

    # 将读取到的字节数据传递给生成器
    return Response(generate_stream(image_bytes), mimetype='text/event-stream')

# --- 5. 启动应用 ---
if __name__ == '__main__':
    load_database_and_create_embeddings()
    if db_questions:
        app.run(debug=True, port=5000)

