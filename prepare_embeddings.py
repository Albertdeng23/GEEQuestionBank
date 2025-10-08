import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RESULT_JSON_FILE = os.path.join("result", "processed_questions.json")
EMBEDDINGS_FILE = os.path.join("result", "db_embeddings.npy") # 定义向量文件的保存路径

def main():
    """
    这是一个一次性运行的脚本。
    它会加载题库JSON，计算所有题目的向量，并将结果保存为.npy文件。
    """
    logging.info("--- 开始预计算题库向量 ---")

    # 1. 检查题库文件是否存在
    if not os.path.exists(RESULT_JSON_FILE):
        logging.error(f"错误: 题库文件 {RESULT_JSON_FILE} 不存在。请先运行core.py生成题库。")
        return

    # 2. 加载题库
    logging.info("正在加载题库JSON文件...")
    with open(RESULT_JSON_FILE, 'r', encoding='utf-8') as f:
        db_questions = json.load(f)
    logging.info(f"题库加载成功，共 {len(db_questions)} 道题目。")

    # 3. 加载向量模型
    logging.info("正在加载向量模型 (第一次可能需要下载)...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    # 4. 提取需要编码的文本
    searchable_texts = [q.get('searchable_text', '') for q in db_questions]
    if not searchable_texts:
        logging.error("题库中没有可供搜索的文本 (searchable_text字段为空)。")
        return

    # 5. 计算向量 (这是最耗时的部分)
    logging.info("正在为题库生成向量，请耐心等待...")
    embeddings = model.encode(searchable_texts, show_progress_bar=True, normalize_embeddings=True)

    # 6. 保存向量到文件
    logging.info(f"向量生成完毕，形状为: {embeddings.shape}")
    np.save(EMBEDDINGS_FILE, embeddings)
    logging.info(f"向量已成功保存到: {EMBEDDINGS_FILE}")
    logging.info("--- 预计算完成 ---")

if __name__ == '__main__':
    main()
