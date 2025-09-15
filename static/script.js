document.addEventListener("DOMContentLoaded", function () {
    let eventSource = null;
    let firstquestionYN = true;
    let currentLoadingMessage = null;
    let uploadProgressOverlay = null;
    
    // 로딩 단계 정의
    const loadingSteps = [
        { id: 'analyze', text: '질문 분석 중...', icon: '🔍' },
        { id: 'search', text: '문서 검색 중...', icon: '📚' },
        { id: 'generate', text: '답변 생성 중...', icon: '✨' }
    ];
    
    removeMapButton();
    addBotMessage("안녕하세요! 무엇을 도와드릴까요?");
    setupFileUpload();
  
    async function startChatStream(question) {
        const chatBox = document.getElementById("chat-box");
        if (!chatBox) return;
    
        removeMapButton();
    
        // 고급 로딩 인디케이터 표시
        currentLoadingMessage = addAdvancedLoadingMessage();
        
        // 단계별 로딩 처리
        try {
            // 1. 질문 분석 단계
            updateLoadingStep('analyze', 'active');
            await sleep(800);
            updateLoadingStep('analyze', 'completed');
            
            // 2. 문서 검색 단계
            updateLoadingStep('search', 'active');
            await sleep(600);
            updateLoadingStep('search', 'completed');
            
            // 3. 답변 생성 단계
            updateLoadingStep('generate', 'active');
            showTypingAnimation();
            await sleep(400);
            
            // 실제 API 호출
            const res = await fetch("/generate-answer", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ question: question })
            });
        
            if (!res.ok) {
                throw new Error(`서버 오류: ${res.status} ${res.statusText}`);
            }
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
        
            let botMessageDiv = null;
            let messageContentEl = null;
            let fullAnswer = "";
        
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
        
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n").filter(line => line.startsWith("data: "));
        
                for (let line of lines) {
                    const jsonStr = line.replace(/^data:\s*/, "").trim();
                    if (!jsonStr) continue;
                    
                    // 특별한 제어 메시지 처리
                    if (jsonStr === "[DONE]") {
                        console.log("스트림 완료");
                        createMapButton(); // 스트림 완료 시 지도 버튼 생성
                        continue; // 다음 라인으로 넘어감
                    }
        
                    try {
                        const event = JSON.parse(jsonStr);
                        
                        // 상태 메시지 처리 (개선): 질문 유형, 검색 상태 등
                        if (event.status) {
                            if (loadingMessage) loadingMessage.remove();
                            
                            if (event.status === "intent_analysis") {
                                // 질문 유형 정보 표시
                                addBotMessage(`🔍 ${event.message}`);
                            } else if (event.status === "knowledge_search") {
                                // 검색 결과 정보 표시
                                addBotMessage(`📚 ${event.message}`);
                            } else if (event.status === "completed") {
                                createMapButton();
                            }
                        }
        
                        // 답변 내용 처리
                        if (event.token) {
                            if (!botMessageDiv) {
                                if (currentLoadingMessage) {
                                    currentLoadingMessage.remove();
                                    currentLoadingMessage = null;
                                }
                                
                                botMessageDiv = document.createElement("div");
                                botMessageDiv.classList.add("message", "bot-message");
                                botMessageDiv.innerHTML = `
                                    <div class="avatar bot-avatar"></div>
                                    <div class="message-content"><strong></strong> </div>
                                `;
                                chatBox.appendChild(botMessageDiv);
                                messageContentEl = botMessageDiv.querySelector(".message-content");
                            }
                            
                            // 새로운 응답 내용 포매팅 추가 (소스 표시 강조)
                            const formattedToken = event.token
                                .replace(/\[([^,\]]+)\.pdf(?:, 페이지: (\d+))?\]/g, '<span class="source-citation">[$1.pdf$2]</span>')
                                .replace(/\n/g, "<br>");
                                
                            fullAnswer = event.token; // 전체 응답으로 대체 (stream이 아니므로)
                            messageContentEl.innerHTML = `<strong></strong> ${formattedToken}`;
                            chatBox.scrollTop = chatBox.scrollHeight;
                        }
                    } catch (err) {
                        console.error("파싱 오류:", err, line);
                    }
                }
            }
        } catch (error) {
            console.error("채팅 스트림 오류:", error);
            if (currentLoadingMessage) {
                currentLoadingMessage.remove();
                currentLoadingMessage = null;
            }
            addBotMessage(`오류가 발생했습니다: ${error.message}`);
        }
    }

    function sendQuestion() {
        const questionInput = document.getElementById("user-input");
        const askButton = document.getElementById("ask-btn");
        const chatBox = document.getElementById("chat-box");
    
        const question = questionInput.value.trim();
        if (!question) {
            alert("질문을 입력하세요!");
            return;
        }
    
        const userMessage = document.createElement("div");
        userMessage.classList.add("message", "user-message");
        userMessage.innerHTML = `
            <div class="avatar user-avatar"></div>
            <div class="message-content">${question}</div>
        `;
        chatBox.appendChild(userMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    
        askButton.disabled = true;
        questionInput.disabled = true;
    
        startChatStream(question);
        questionInput.value = "";
    
        setTimeout(() => {
            askButton.disabled = false;
            questionInput.disabled = false;
        }, 3000);
    }
    
    function removeMapButton() {
        const existingButton = document.querySelector('.map-button');
        if (existingButton) {
            existingButton.remove();
        }
    }

    function createMapButton() {
        if (document.querySelector('.map-button')) {
            return;
        }

        console.log("지도 버튼 생성");
        const mapButton = document.createElement("button");
        mapButton.className = "map-button";
        mapButton.innerHTML = `
            <svg class="map-icon" viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
            </svg>
        `;
        mapButton.onclick = function() { openNaverMap(); };
        document.body.appendChild(mapButton);
    }

    function addBotMessage(message) {
        const chatBox = document.getElementById("chat-box");
        if (!chatBox) return;
    
        const botMessage = document.createElement("div");
        botMessage.classList.add("message", "bot-message");
        botMessage.innerHTML = `
            <div class="avatar bot-avatar"></div>
            <div class="message-content">${message}</div>
        `;
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    const askButton = document.getElementById("ask-btn");
    if (askButton) {
        askButton.addEventListener("click", sendQuestion);
    } else {
        console.error("ask-btn 요소를 찾을 수 없습니다.");
    }

    const userInput = document.getElementById("user-input");
    if (userInput) {
        userInput.addEventListener("keypress", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                sendQuestion();
            }
        });
    } else {
        console.error("user-input 요소를 찾을 수 없습니다.");
    }

    // 파일 업로드 기능 설정
    function setupFileUpload() {
        const uploadBtn = document.getElementById('upload-btn');
        if (!uploadBtn) return;

        uploadBtn.addEventListener('click', function() {
            // 파일 선택 다이얼로그 생성
            showFileUploadDialog();
        });
    }

    function showFileUploadDialog() {
        // 파일 선택 다이얼로그 HTML 생성
        const dialogHTML = `
            <div class="file-drop-area" id="file-drop-area">
                <div class="file-drop-container">
                    <h2>문서 업로드</h2>
                    <p>지식 데이터베이스에 추가할 문서를 선택하세요</p>
                    <div class="file-drop-zone" id="file-drop-zone">
                        <p>파일을 여기에 끌어다 놓거나 클릭하여 선택하세요</p>
                        <p>지원 형식: PDF, DOCX, XLSX, XLS, HWP</p>
                    </div>
                    <div class="file-list" id="file-list"></div>
                    <div class="upload-actions">
                        <button id="cancel-upload">취소</button>
                        <button id="confirm-upload">업로드</button>
                    </div>
                </div>
            </div>
        `;

        // DOM에 추가
        document.body.insertAdjacentHTML('beforeend', dialogHTML);

        // 파일 선택 이벤트
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.multiple = true;
        fileInput.accept = '.pdf,.docx,.xlsx,.xls,.hwp';
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);

        const dropArea = document.getElementById('file-drop-area');
        const dropZone = document.getElementById('file-drop-zone');
        const fileList = document.getElementById('file-list');
        const cancelBtn = document.getElementById('cancel-upload');
        const confirmBtn = document.getElementById('confirm-upload');

        // 선택된 파일 목록
        let selectedFiles = [];

        // 이벤트 핸들러
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFiles = Array.from(e.target.files);
                updateFileList();
            }
        });

        // 드래그 앤 드롭 처리
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.background = '#e6f3ff';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.background = '';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.background = '';
            
            if (e.dataTransfer.files.length > 0) {
                selectedFiles = Array.from(e.dataTransfer.files);
                updateFileList();
            }
        });

        // 파일 목록 업데이트
        function updateFileList() {
            fileList.innerHTML = '';
            selectedFiles.forEach(file => {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.innerHTML = `
                    <span>${file.name} (${formatFileSize(file.size)})</span>
                    <button class="remove-file" data-name="${file.name}">삭제</button>
                `;
                fileList.appendChild(item);
            });

            // 삭제 버튼 이벤트
            document.querySelectorAll('.remove-file').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const fileName = e.target.getAttribute('data-name');
                    selectedFiles = selectedFiles.filter(file => file.name !== fileName);
                    updateFileList();
                });
            });
        }

        // 파일 크기 포맷
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            else return (bytes / 1048576).toFixed(1) + ' MB';
        }

        // 취소 버튼
        cancelBtn.addEventListener('click', () => {
            dropArea.remove();
            fileInput.remove();
        });

        // 업로드 버튼
        confirmBtn.addEventListener('click', () => {
            if (selectedFiles.length === 0) {
                alert('파일을 선택해주세요.');
                return;
            }

            uploadFiles(selectedFiles);
            dropArea.remove();
            fileInput.remove();
        });

        // 다이얼로그 표시
        dropArea.style.display = 'flex';
    }

    // 파일 업로드 실행 (고급 진행률 표시 추가)
    function uploadFiles(files) {
        if (files.length === 0) return;
        
        // 첫 번째 파일만 업로드
        const file = files[0];
        
        // 파일 유효성 검사
        if (!file || file.size === 0) {
            addBotMessage(`오류: 선택한 파일이 비어있거나 유효하지 않습니다.`);
            return;
        }
        
        // 업로드 진행률 표시
        showUploadProgress(file.name);
        
        const formData = new FormData();
        formData.append('file', file);
        
        // 선택적 파라미터 추가 (문자열로 전송)
        formData.append('description', '');
        formData.append('auto_detect_synonyms', 'false');

        // 디버깅: FormData 내용 확인
        console.log("업로드 파일 정보:", {
            "파일명": file.name,
            "파일 크기": file.size,
            "파일 타입": file.type
        });
        
        console.log("FormData 내용:");
        for (let pair of formData.entries()) {
            console.log(pair[0] + ': ' + (pair[0] === 'file' ? pair[1].name : pair[1]));
        }

        // XMLHttpRequest를 사용해서 업로드 진행률 추적
        const xhr = new XMLHttpRequest();
        
        // 진행률 업데이트
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                updateUploadProgress(percentComplete, '업로드 중...');
            }
        });

        // 업로드 완료 처리
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                updateUploadProgress(100, '업로드 완료!');
                
                setTimeout(() => {
                    hideUploadProgress();
                    addBotMessage(`📄 "${file.name}" 파일이 성공적으로 업로드되었습니다. 이제 이 문서에 대해 질문하실 수 있습니다.`);
                }, 1000);
            } else {
                throw new Error('업로드 실패');
            }
        });

        // 에러 처리
        xhr.addEventListener('error', () => {
            throw new Error('업로드 중 네트워크 오류가 발생했습니다.');
        });

        xhr.open('POST', '/api/documents/upload');
        xhr.send(formData);
        
        /*
        fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // 응답 상태와 텍스트를 함께 처리
            return response.text().then(text => {
                console.log("서버 응답 전체:", text);
                try {
                    const json = JSON.parse(text);
                    console.log("파싱된 응답:", json);
                    if (!response.ok) {
                        throw new Error(json.detail || '파일 업로드 실패');
                    }
                    return json;  // 이 반환값이 다음 then으로 전달됨
                } catch (e) {
                    console.error("응답 파싱 오류:", e);
                    throw new Error(text || response.statusText);
                }
            });
        })
        .then(data => {
            console.log("업로드 성공 응답:", data);
            addBotMessage(`문서가 성공적으로 업로드되었습니다! 이제 문서 내용에 대해 질문할 수 있습니다.`);
            
            // 나머지 파일이 있으면 재귀적으로 업로드
            if (files.length > 1) {
                const remainingFiles = Array.from(files).slice(1);
                setTimeout(() => uploadFiles(remainingFiles), 1000);
            }
        })
        .catch(error => {
            console.error("업로드 오류 세부정보:", error);
            addBotMessage(`업로드 중 오류가 발생했습니다: ${error.message}`);
        });
        */
        
        // XHR 오류 처리를 위한 try-catch
        try {
            // 위의 XHR 코드가 실행됨
        } catch (error) {
            console.error('업로드 오류:', error);
            hideUploadProgress();
            addBotMessage(`❌ 파일 업로드 중 오류가 발생했습니다: ${error.message}`);
        }
    }
    // 네이버 지도 열기 함수 (구현 필요)
    function openNaverMap() {
        // 네이버 지도 관련 기능 구현
        console.log("네이버 지도 열기");
    }
    
    // 고급 로딩 메시지 추가 함수
    function addAdvancedLoadingMessage() {
        const chatBox = document.getElementById("chat-box");
        if (!chatBox) return null;
        
        const loadingDiv = document.createElement("div");
        loadingDiv.classList.add("advanced-loading");
        
        const stepsHTML = loadingSteps.map(step => 
            `<li class="loading-step" id="step-${step.id}">
                <div class="step-indicator">
                    <div class="step-spinner" style="display: none;"></div>
                    <span class="step-icon" style="display: none;">${step.icon}</span>
                    <span class="step-number">•</span>
                </div>
                <span class="step-text">${step.text}</span>
            </li>`
        ).join('');
        
        loadingDiv.innerHTML = `
            <div class="loading-header">
                <div class="loading-avatar">🤖</div>
                <div class="loading-title">AI가 생각하고 있습니다</div>
            </div>
            <ul class="loading-steps">
                ${stepsHTML}
            </ul>
        `;
        
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        return loadingDiv;
    }
    
    // 로딩 단계 업데이트 함수
    function updateLoadingStep(stepId, status) {
        const stepElement = document.getElementById(`step-${stepId}`);
        if (!stepElement) return;
        
        const spinner = stepElement.querySelector('.step-spinner');
        const icon = stepElement.querySelector('.step-icon');
        const number = stepElement.querySelector('.step-number');
        
        // 모든 상태 클래스 제거
        stepElement.classList.remove('active', 'completed');
        
        if (status === 'active') {
            stepElement.classList.add('active');
            spinner.style.display = 'block';
            icon.style.display = 'none';
            number.style.display = 'none';
        } else if (status === 'completed') {
            stepElement.classList.add('completed');
            spinner.style.display = 'none';
            icon.style.display = 'block';
            number.style.display = 'none';
        }
    }
    
    // 타이핑 애니메이션 표시 함수
    function showTypingAnimation() {
        if (!currentLoadingMessage) return;
        
        const stepsContainer = currentLoadingMessage.querySelector('.loading-steps');
        stepsContainer.innerHTML = `
            <li class="loading-step active">
                <div class="step-indicator">
                    <div class="step-spinner"></div>
                </div>
                <span class="step-text">답변을 입력하고 있습니다</span>
            </li>
            <div class="enhanced-typing-animation">
                <div class="enhanced-typing-dot"></div>
                <div class="enhanced-typing-dot"></div>
                <div class="enhanced-typing-dot"></div>
            </div>
        `;
    }
    
    // 업로드 진행률 표시 함수
    function showUploadProgress(filename) {
        // 기존 오버레이 제거
        hideUploadProgress();
        
        // 오버레이 생성
        uploadProgressOverlay = document.createElement('div');
        uploadProgressOverlay.className = 'upload-overlay';
        
        uploadProgressOverlay.innerHTML = `
            <div class="upload-progress">
                <div class="upload-status" id="upload-status">파일 업로드 중...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="upload-filename" id="upload-filename">${filename}</div>
            </div>
        `;
        
        document.body.appendChild(uploadProgressOverlay);
    }
    
    // 업로드 진행률 업데이트 함수
    function updateUploadProgress(percentage, status) {
        const progressFill = document.getElementById('progress-fill');
        const uploadStatus = document.getElementById('upload-status');
        
        if (progressFill) {
            progressFill.style.width = percentage + '%';
        }
        if (uploadStatus) {
            uploadStatus.textContent = status;
        }
    }
    
    // 업로드 진행률 숨기기 함수
    function hideUploadProgress() {
        if (uploadProgressOverlay) {
            uploadProgressOverlay.remove();
            uploadProgressOverlay = null;
        }
    }
    
    // 유틸리티 함수
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});