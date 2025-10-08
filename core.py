import os
import json
import base64
import glob
import logging
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. 配置与初始化 ---

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# 从.env文件读取常量
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

# 定义目录常量
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(PROJECT_ROOT, "files")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result")
TEMP_PATH = os.path.join(PROJECT_ROOT, "temp")
RESULT_JSON_FILE = os.path.join(RESULT_PATH, "processed_questions.json")
PROCESSED_LOG_FILE = os.path.join(RESULT_PATH, "processed_files.log")

# 初始化OpenAI客户端
# 确保你的API服务商兼容OpenAI的SDK
try:
    client = openai.OpenAI(api_key=API_KEY, base_url=API_URL)
except Exception as e:
    logging.error(f"初始化OpenAI客户端失败，请检查API_URL和API_KEY: {e}")
    exit()

# --- 2. 核心提示词 (采用方案二：高保真版) ---

VLM_PROMPT = """
你是一位精通LaTeX和Markdown的考研试卷数字化专家。你的任务是高保真地分析我提供的试卷图片，并以结构化的JSON格式提取所有题目信息。

请严格遵循以下指令：
1.  **精准识别**：识别图片中的所有题目，包括大题标题（如“一、选择题”）。
2.  **高保真转录**：
    *   **数学公式**：所有题目和选项中的数学公式，必须使用LaTeX格式进行转录。
    *   **表格**：如果题干或选项中包含表格，请将其转换为Markdown格式的表格。
3.  **结构化提取**：对于每一道题目，提取以下字段：
    *   `section_title`: 字符串类型，题目所属的大题标题，例如 "一、选择题" 或 "三、解答题"。如果无法判断，则为 `null`。
    *   `question_number`: 字符串类型，题目的编号，例如 "1" 或 "17"。
    *   `stem_text`: 字符串类型，包含完整题干，其中所有公式都使用LaTeX格式。
    *   `options`: JSON对象，键为选项字母，值为包含LaTeX公式的选项文本。非选择题则为 `null`。
    *   `image_description`: 字符串类型，如果题目附有图表，请详细描述图表内容。若无图，则为 "无"。
4.  **跨页处理**：如果当前页面包含一个题目的结尾部分，并且我提供了上一页的未完成题目信息，请将它们合并成一个完整的题目。
5.  **输出格式**：将所有题目组织成一个JSON列表。不要输出任何解释性文字或代码块标记。
"""

CONTINUATION_PROMPT_TEMPLATE = """
接续任务：上一页的最后一个题目没有完整，这是已提取的部分信息：
{context}

请分析当前页面，首先完成这个题目，然后再继续提取本页的其他新题目。请将完成的题目和新题目一起，按照标准JSON格式返回。
"""

# --- 3. 辅助函数 ---

def setup_directories():
    """创建结果和临时文件夹（如果不存在）。"""
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)

def pdf_to_images(pdf_path: str, temp_folder: str) -> List[str]:
    """
    使用PyMuPDF将PDF页面转换为图片。
    返回生成的图片路径列表。
    """
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for page_num in tqdm(range(len(doc)), desc=f"  - 渲染页面到图片", leave=False):
            page = doc.load_page(page_num)
            # 设置更高分辨率以获得更好OCR效果
            pix = page.get_pixmap(dpi=300) 
            img_path = os.path.join(temp_folder, f"{pdf_name}_page_{page_num + 1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        doc.close()
    except Exception as e:
        logging.error(f"处理PDF '{pdf_path}' 失败: {e}")
    return image_paths

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为Base64字符串。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vlm_api(image_base64: str, prompt: str) -> Optional[List[Dict]]:
    """调用VLM API并返回解析后的JSON数据。"""
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20",  # 根据你的模型服务商提供的模型名称修改
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=8192, # 根据需要调整
            temperature=0.0 # 使用低温以获得更确定的结构化输出
        )
        content = response.choices[0].message.content
        # 清理可能的Markdown代码块标记
        if content.strip().startswith("```json"):
            content = content.strip()[7:-3]
        
        return json.loads(content)
    except json.JSONDecodeError:
        logging.error(f"API返回的不是有效的JSON格式: {content}")
    except Exception as e:
        logging.error(f"调用API时发生错误: {e}")
    return None

def is_question_incomplete(question: Dict) -> bool:
    """
    一个启发式函数，用于判断一个提取出的题目是否可能不完整。
    这是处理跨页问题的关键。
    """
    stem = question.get("stem_text", "")
    options = question.get("options")
    
    # 规则1: 如果是选择题（有选项字母在题干末尾），但options字段为空，则很可能跨页。
    if any(letter in stem[-10:] for letter in ["A.", "B.", "C.", "D."]) and options is None:
        return True
        
    # 规则2: 题干以连接词或逗号结尾，可能不完整。
    if stem.endswith(("，", ",", "：", ":", "如图所示")):
        return True
        
    return False

def load_processed_files() -> set:
    """加载已处理的文件列表，用于增量更新。"""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    with open(PROCESSED_LOG_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def log_processed_file(filename: str):
    """记录已处理的文件名。"""
    with open(PROCESSED_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(filename + '\n')

def append_to_json_file(data: List[Dict], filepath: str):
    """增量写入JSON文件。"""
    existing_data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"{filepath} 文件内容不是有效的JSON，将覆盖。")
    
    existing_data.extend(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# --- 4. 主处理流程 ---

def main():
    """主函数，执行所有处理步骤。"""
    logging.info("--- 开始处理考研真题PDF ---")
    setup_directories()
    
    processed_files = load_processed_files()
    pdf_files = glob.glob(os.path.join(FILE_PATH, "*.pdf"))
    
    if not pdf_files:
        logging.warning(f"在 '{FILE_PATH}' 目录下没有找到PDF文件。")
        return

    for pdf_path in tqdm(pdf_files, desc="处理PDF文件"):
        pdf_filename = os.path.basename(pdf_path)
        if pdf_filename in processed_files:
            logging.info(f"跳过已处理的文件: {pdf_filename}")
            continue

        logging.info(f"开始处理新文件: {pdf_filename}")
        
        image_paths = pdf_to_images(pdf_path, TEMP_PATH)
        if not image_paths:
            continue

        all_questions_for_this_pdf = []
        incomplete_question_buffer = None

        for i, img_path in enumerate(tqdm(image_paths, desc="  - 处理页面", leave=False)):
            page_num = i + 1
            logging.info(f"    - 正在处理 {pdf_filename} 的第 {page_num} 页...")
            
            base64_image = encode_image_to_base64(img_path)
            
            current_prompt = VLM_PROMPT
            if incomplete_question_buffer:
                context_str = json.dumps(incomplete_question_buffer, ensure_ascii=False)
                current_prompt = CONTINUATION_PROMPT_TEMPLATE.format(context=context_str) + "\n" + VLM_PROMPT
                logging.info("    - 检测到未完成题目，使用接续提示词。")

            page_results = call_vlm_api(base64_image, current_prompt)
            
            if page_results:
                # 如果有未完成题目，第一个返回的结果应该是合并后的完整题目
                if incomplete_question_buffer and page_results:
                    # 将合并后的题目信息更新到列表中
                    all_questions_for_this_pdf.append(page_results[0])
                    # 处理本页剩余的新题目
                    remaining_results = page_results[1:]
                    incomplete_question_buffer = None # 清空缓冲区
                else:
                    remaining_results = page_results

                # 检查本页最后一个题目是否完整
                if remaining_results and is_question_incomplete(remaining_results[-1]):
                    incomplete_question_buffer = remaining_results.pop()
                    logging.warning(f"    - 第 {page_num} 页最后一个题目可能不完整，已存入缓冲区。")
                
                # 为所有本页已完成的题目添加元数据
                for question in remaining_results:
                    question['source_pdf'] = pdf_filename
                    question['source_page'] = page_num
                    # 5. 为向量检索做准备
                    question['searchable_text'] = f"{question.get('section_title', '')} {question.get('question_number', '')}: {question.get('stem_text', '')}"
                    all_questions_for_this_pdf.append(question)

        # 清理当前PDF的临时图片
        for img_path in image_paths:
            os.remove(img_path)
        
        # 4. 增量更新JSON文件
        if all_questions_for_this_pdf:
            logging.info(f"文件 {pdf_filename} 处理完毕，共提取 {len(all_questions_for_this_pdf)} 道题目。正在写入JSON...")
            append_to_json_file(all_questions_for_this_pdf, RESULT_JSON_FILE)
            log_processed_file(pdf_filename)
        else:
            logging.warning(f"文件 {pdf_filename} 未能提取出任何题目。")

    logging.info("--- 所有文件处理完毕 ---")


if __name__ == "__main__":
    main()
