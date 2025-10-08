document.addEventListener('DOMContentLoaded', function() {

    // --- START: 新增图片预览功能 ---
    const similarImageUpload = document.getElementById('similar-image-upload');
    const similarImagePreview = document.getElementById('similar-image-preview');
    const analyzeImageUpload = document.getElementById('analyze-image-upload');
    const analyzeImagePreview = document.getElementById('analyze-image-preview');

    /**
     * 为文件输入框设置实时预览功能
     * @param {HTMLInputElement} inputElement - 文件输入框元素
     * @param {HTMLImageElement} previewElement - 用于预览的img元素
     */
    function setupImagePreview(inputElement, previewElement) {
        inputElement.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    // 当文件读取完成后，将结果（Data URL）赋给img的src
                    previewElement.src = e.target.result;
                    // 移除d-none类，使图片可见
                    previewElement.classList.remove('d-none');
                }
                
                // 读取文件内容作为Data URL
                reader.readAsDataURL(file);
            } else {
                // 如果用户取消选择，则隐藏预览
                previewElement.classList.add('d-none');
                previewElement.src = "#";
            }
        });
    }

    // 为两个上传框分别设置预览
    setupImagePreview(similarImageUpload, similarImagePreview);
    setupImagePreview(analyzeImageUpload, analyzeImagePreview);
    // --- END: 新增图片预览功能 ---


    // --- 查找类似题目功能 (原有代码) ---
    const btnFindSimilar = document.getElementById('btn-find-similar');
    const similarResultsDiv = document.getElementById('similar-results');
    const similarSpinner = document.getElementById('similar-spinner');

    btnFindSimilar.addEventListener('click', async () => {
        const file = similarImageUpload.files[0];
        if (!file) {
            alert('请先选择一个图片文件！');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        similarSpinner.classList.remove('d-none');
        similarResultsDiv.innerHTML = '';

        try {
            const response = await fetch('/find_similar', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '服务器发生错误');
            }

            const results = await response.json();
            displaySimilarResults(results);

        } catch (error) {
            similarResultsDiv.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        } finally {
            similarSpinner.classList.add('d-none');
        }
    });

    function displaySimilarResults(results) {
        if (results.length === 0) {
            similarResultsDiv.innerHTML = '<p>未找到相似题目。</p>';
            return;
        }

        let html = '<h3>相似题目 Top 3</h3>';
        results.forEach(q => {
            html += `
                <div class="card mb-3">
                    <div class="card-header">
                        <strong>来源:</strong> ${q.source_pdf} (第 ${q.source_page} 页) | <strong>相似度:</strong> ${q.similarity}
                    </div>
                    <div class="card-body">
                        <p class="card-text">${q.stem_text}</p>
                        ${q.options ? Object.entries(q.options).map(([key, value]) => `<p>${key}. ${value}</p>`).join('') : ''}
                    </div>
                </div>
            `;
        });
        similarResultsDiv.innerHTML = html;
        // 渲染LaTeX公式
        MathJax.typesetPromise([similarResultsDiv]);
    }


    // --- 分析 & 生成题目功能 (原有代码) ---
    const btnAnalyze = document.getElementById('btn-analyze');
    const analysisResultsDiv = document.getElementById('analysis-results');

    btnAnalyze.addEventListener('click', async () => {
        const file = analyzeImageUpload.files[0];
        if (!file) {
            alert('请先选择一个图片文件！');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        
        analysisResultsDiv.innerHTML = '<div class="spinner-border text-success" role="status"><span class="visually-hidden">Loading...</span></div>';
        
        try {
            const response = await fetch('/analyze_and_generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            analysisResultsDiv.innerHTML = ''; // 清空等待提示

            let buffer = '';
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                const parts = buffer.split('\n\n');
                buffer = parts.pop(); 

                for (const part of parts) {
                    if (part.startsWith('data: ')) {
                        let content = part.substring(6);
                        if (content.includes('--- END OF STREAM ---')) {
                            return;
                        }
                        analysisResultsDiv.innerHTML += marked.parse(content);
                        MathJax.typesetPromise([analysisResultsDiv]);
                    }
                }
            }
        } catch (error) {
            analysisResultsDiv.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        }
    });
});
