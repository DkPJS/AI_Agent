# 🤖 Intelligent Multi-Agent RAG Document Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-red.svg)](https://neo4j.com)
[![Weaviate](https://img.shields.io/badge/Weaviate-1.20+-orange.svg)](https://weaviate.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

지능형 멀티 에이전트 기반 RAG(Retrieval-Augmented Generation) 문서 챗봇으로, 다양한 문서 형식을 처리하고 그래프 기반 지식베이스를 활용하여 정확한 질의응답을 제공합니다.

##  주요 기능

###  다양한 문서 형식 지원
- **지원 형식**: PDF, DOCX, XLSX, XLS, HWP, TXT
- **구조 분석**: 헤더, 섹션, 메타데이터 자동 추출
- **스마트 청킹**: 적응형 의미 기반 문서 분할
- **문서 분류**: 자동 문서 유형 판별 (일반, 보고서, 매뉴얼, 계약서)

###  멀티 에이전트 시스템
- **Agent Orchestrator**: 중앙 조정자 - 작업 분산 및 로드 밸런싱
- **Search Agent**: 검색 전문가 - 다중 검색 전략 동시 실행
- **QA Agent**: 질의응답 전문가 - 컨텍스트 기반 추론
- **RAG Coordinator**: 통합 조정자 - 검색-생성 파이프라인 최적화

###  하이브리드 검색 엔진
1. **Vector Search** (Weaviate) - 의미적 유사도 검색
2. **Keyword Search** (TF-IDF) - 정확한 키워드 매칭
3. **Graph Search** (Neo4j) - 엔티티 관계 기반 검색
4. **Entity Search** - 명명된 개체 기반 동의어 확장 검색

###  지식 그래프 시스템
- **Neo4j 그래프 DB**: Document, Chunk, Entity, Synonym 노드
- **자동 관계 추출**: 엔티티 간 관계 자동 발견
- **동의어 시스템**: 문서별 자동 동의어 탐지 및 도메인 특화 용어 매핑

##  시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                    │
│                    (FastAPI REST API)                   │
├─────────────────────────────────────────────────────────┤
│                 Business Logic Layer                    │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│ │Multi-Agent  │ │   Document  │ │    RAG System       │ │
│ │   System    │ │  Processing │ │                     │ │
│ └─────────────┘ └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Infrastructure Layer                     │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│ │PostgreSQL   │ │   Neo4j     │ │     Weaviate        │ │
│ │(Metadata)   │ │(Knowledge)  │ │    (Vectors)        │ │
│ └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

##  처리 프로세스

### Phase 1: 문서 수집 및 전처리
```
문서 업로드 → 형식 검증 → 텍스트 추출 → 구조 분석 → 스마트 청킹
```

### Phase 2: 지식 그래프 구축
```
엔티티 추출 → 관계 분석 → Neo4j 저장 → 동의어 매핑
```

### Phase 3: 벡터화 및 인덱싱
```
임베딩 생성 → Weaviate 저장 → TF-IDF 인덱싱 → 하이브리드 검색 준비
```

### Phase 4: 질의응답
```
질문 분석 → 멀티 에이전트 협업 → 동적 검색 전략 → 답변 생성
```

##  설치 및 실행

### 사전 요구사항
- Python 3.10+
- Docker & Docker Compose
- Git

### 1. 저장소 클론
```bash
git clone https://github.com/DkPJS/AI_Agent.git
cd AI_Agent
```

### 2. 환경 설정
```bash
# .env 파일 생성 및 설정
cp .env.example .env
# 필요한 환경 변수 설정 (데이터베이스 URL, API 키 등)
```

### 3. Docker로 실행 (권장)
```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 4. 수동 설치
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 초기화
python -m alembic upgrade head

# 애플리케이션 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

##  API 엔드포인트

###  Documents API
```http
POST /api/v1/documents/upload     # 문서 업로드 및 처리
GET  /api/v1/documents/list       # 문서 목록 조회
DELETE /api/v1/documents/{id}     # 문서 삭제
```

###  Chat API
```http
POST /api/v1/chat/                # 질의응답
GET  /api/v1/chat/sessions        # 세션 목록
GET  /api/v1/chat/sessions/{id}   # 대화 히스토리
```

###  Agent RAG API
```http
POST /api/v1/agent-rag/multi-agent-query  # 멀티 에이전트 질의
GET  /api/v1/agent-rag/agent-status       # 에이전트 상태 조회
```

##  사용 예시

### 1. 문서 업로드
```python
import requests

files = {'file': open('document.pdf', 'rb')}
data = {
    'description': '정책 문서',
    'auto_detect_synonyms': 'true',
    'analyze_structure': 'true'
}

response = requests.post(
    'http://localhost:8000/api/v1/documents/upload',
    files=files,
    data=data
)
```

### 2. 질의응답
```python
response = requests.post(
    'http://localhost:8000/api/v1/chat/',
    json={
        'message': '지원 사업의 선정 기준은 무엇인가요?',
        'session_id': 'your-session-id'
    }
)

answer = response.json()['answer']
sources = response.json()['sources']
```

##  프로젝트 구조

```
rag-document-chatbot/
├── 📄 app.py                               # FastAPI 메인 애플리케이션
├── 📄 requirements.txt                     # Python 패키지 의존성
├── 📄 docker-compose.yml                   # Docker 컨테이너 설정
├── 📁 app/                                 # 메인 애플리케이션 코드
│   ├── 📁 api/v1/                          # API 엔드포인트
│   ├── 📁 agents/                          # 멀티 에이전트 시스템
│   │   ├── 📁 coordinators/                # 조정자 에이전트
│   │   └── 📁 specialized/                 # 전문 에이전트
│   ├── 📁 core/                            # 핵심 설정
│   ├── 📁 infrastructure/                  # 인프라스트럭처 계층
│   │   ├── 📁 database/                    # 데이터베이스 클라이언트
│   │   └── 📁 external/                    # 외부 서비스 연동
│   ├── 📁 models/                          # 데이터 모델
│   ├── 📁 rag/                             # RAG 시스템 핵심
│   │   ├── 📁 generation/                  # 답변 생성
│   │   ├── 📁 retrieval/                   # 검색 및 검색
│   │   └── 📁 utils/                       # RAG 유틸리티
│   ├── 📁 services/                        # 비즈니스 로직 서비스
│   │   ├── 📁 document/                    # 문서 처리 서비스
│   │   └── 📁 nlp/                         # 자연어 처리 서비스
│   └── 📁 utils/                           # 공통 유틸리티
```

## 주요 기술 스택

### Backend Framework
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **SQLAlchemy**: ORM 및 데이터베이스 관리
- **Pydantic**: 데이터 검증 및 설정 관리

### 데이터 저장소
- **PostgreSQL**: 구조적 데이터 (문서 메타데이터, 세션)
- **Neo4j**: 그래프 데이터베이스 (지식 그래프, 엔티티 관계)
- **Weaviate**: 벡터 데이터베이스 (문서 임베딩, 의미적 검색)

### AI/ML
- **Hugging Face Transformers**: 임베딩 모델
- **spaCy**: 자연어 처리 및 엔티티 추출
- **scikit-learn**: TF-IDF 및 텍스트 분석

### Document Processing
- **PyPDF2/pdfplumber**: PDF 처리
- **python-docx**: DOCX 처리
- **openpyxl**: Excel 처리
- **olefile**: HWP 처리

##  최근 개선사항

###  답변 정확도 향상 (v2.0)
-  **프롬프트 최적화**: 간결하고 효과적인 LLM 프롬프트
-  **스마트 검색 결합**: 품질 점수 기반 지능형 결과 융합
-  **동적 가중치 시스템**: 질문 유형별 적응형 검색 조정
-  **적응형 청킹**: 문서 특성 기반 최적 분할
-  **다층적 품질 평가**: 6차원 답변 품질 평가 시스템

###  성능 최적화
-  **배치 임베딩**: 멀티스레드 임베딩 생성
-  **캐싱 전략**: 검색 결과 및 임베딩 캐시
-  **로드 밸런싱**: 에이전트 간 작업 분산

## 🔧 설정 옵션

### 환경 변수
```bash
# 데이터베이스 설정
DATABASE_URL=postgresql://user:pass@localhost/dbname
NEO4J_URI=bolt://localhost:7687
WEAVIATE_URL=http://localhost:8080

# LLM 설정
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=llama2

# 임베딩 모델
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 업로드 설정
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=100MB
```

##  모니터링 및 로깅

시스템은 상세한 로깅과 성능 메트릭을 제공합니다:

- **처리 시간 추적**: 각 단계별 성능 측정
- **상세 로그**: 문서 처리, 검색, 생성 과정 추적
- **에러 추적**: 예외 상황 모니터링
- **품질 메트릭**: 답변 품질 및 사용자 만족도 측정


---
