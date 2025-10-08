# 考研智能题库 (AI-Powered Exam Question Bank)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于多模态大语言模型（VLM）的智能考研题库系统。用户可以上传题目图片，系统能够自动在题库中检索相似题目、深度分析题目考点，并生成全新的练习题。

---

## ✨ 功能演示 (Features Showcase)

*(建议此处替换为你自己录制的GIF或截图)*

**1. 查找相似题目**
用户上传一张题目图片，系统快速从预处理好的题库中检索出最相似的3道题目，并显示来源和相似度分数。

  <!-- 替换为你自己的截图链接 -->

**2. 分析考点 & 生成新题**
用户上传图片后，系统会流式输出对该题目的考点分析、知识点总结，并紧接着生成3道考点类似但内容全新的题目，所有数学公式均使用LaTeX完美渲染。

 <!-- 替换为你自己的截图链接 -->

---

## 🚀 核心功能

*   **📚 PDF 自动解析**: 自动处理 `files` 目录下的PDF真题卷，提取题目并结构化存储。
*   **🔍 AI 驱动的相似度搜索**: 利用 `Sentence-Transformers` 将题目文本向量化，实现高效、精准的语义相似度匹配。
*   **🧠 智能考点分析**: 调用VLM深度分析题目，精准总结核心考点和解题思路。
*   **✍️ 相似题目生成**: 根据原题的考点，动态生成内容和数据不同的新题目，帮助用户举一反三。
*   **🖥️ 友好 Web 界面**: 基于 `Flask` 和 `Bootstrap` 构建，支持图片上传、实时预览和流式响应。
*   **📐 LaTeX 公式渲染**: 前端集成 `MathJax`，完美展示所有复杂的数学公式。

---

## 🛠️ 技术栈 (Technology Stack)

| 分类      | 技术                                                                                             |
| :-------- | :----------------------------------------------------------------------------------------------- |
| **后端**  | `Flask`, `OpenAI API`, `Sentence-Transformers`, `Scikit-learn`, `PyMuPDF (fitz)`, `Numpy`          |
| **前端**  | `HTML5`, `Bootstrap 5`, `JavaScript`, `MathJax 3`, `Marked.js`                                   |
| **模型**  | 多模态大语言模型 (VLM, e.g., Gemini), 文本向量模型 (`BAAI/bge-large-zh-v1.5`)                      |
| **部署**  | `Gunicorn` (推荐), `Nginx` (推荐)                                                                |

---

## ⚙️ 安装与部署指南

### 1. 环境准备

*   确保你已安装 Python 3.8 或更高版本。
*   一个兼容OpenAI API的多模态模型服务商，并获取其 **API Key** 和 **API URL**。

### 2. 克隆与安装依赖

```bash
# 克隆本仓库
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 创建并激活虚拟环境 (推荐)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt
