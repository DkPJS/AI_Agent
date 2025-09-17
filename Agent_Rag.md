  RAG Document Chatbot 전체 시스템 아키텍처

  1. 시스템 개요
  지능형 멀티 에이전트 기반 RAG(Retrieval-Augmented Generation) 문서 챗봇으로, 
  다양한 문서 형식을 처리하고 그래프 기반 지식베이스를 활용하여 정확한 질의응답을 제공합니다.

  ---
  
  2. 전체 프로세스 플로우
  Phase 1: 문서 수집 및 전처리
  문서 업로드 → 형식 검증 → 텍스트 추출 → 구조 분석 → 청킹

  Phase 2: 지식 그래프 구축
  엔티티 추출 → 관계 분석 → Neo4j 저장 → 동의어 매핑

  Phase 3: 벡터화 및 인덱싱
  임베딩 생성 → Weaviate 저장 → TF-IDF 인덱싱 → 하이브리드 검색 준비

  Phase 4: 질의응답
  질문 분석 → 멀티 에이전트 협업 → 검색 전략 선택 → 답변 생성

  ---
  
  3. 핵심 기능 모듈

  Document Processing Layer
  - 지원 형식: PDF, DOCX, XLSX, XLS, HWP, TXT
  - 구조 분석: 헤더, 섹션, 메타데이터 자동 추출
  - 청킹 전략: 의미 단위 기반 분할
  - 문서 분류: 자동 문서 유형 판별

  # 문서 처리 예시
   processor = BaseDocumentProcessor.get_processor(file_path)
   chunks = processor.process(file_path)
   document_type = processor.classify_document_type(text, structure)

  Multi-Agent System
  Agent Orchestrator - 중앙 조정자
  - 작업 분산 및 로드 밸런싱
  - Agent 간 협업 조정
  - 성능 기반 Agent 선택

  Search Agent - 검색 전문가
  - 기능: semantic_search, hybrid_search, entity_search, graph_search
  - 다중 검색 전략 동시 실행
  - 결과 융합 및 랭킹

  QA Agent - 질의응답 전문가
  - 기능: question_answering, context_analysis, reasoning
  - 컨텍스트 기반 추론
  - 답변 품질 자기 평가

  RAG Coordinator - 통합 조정자
  - 기능: rag_orchestration, retrieval_optimization
  - 검색-생성 파이프라인 최적화
  - 성능 모니터링

  Advanced Retrieval System
  하이브리드 검색 엔진
  1. Vector Search (Weaviate)
    - 의미적 유사도 검색
    - 임베딩 기반 검색
  2. Keyword Search (TF-IDF)
    - 정확한 키워드 매칭
    - 통계적 중요도 계산
  3. Graph Search (Neo4j)
    - 엔티티 관계 기반 검색
    - 지식 그래프 탐색
  4. Entity Search
    - 명명된 개체 기반 검색
    - 동의어 확장 검색

  지능형 검색 전략
  class SearchStrategy:
      - AUTO: 질문 유형 자동 판별
      - SEMANTIC: 의미적 검색 우선
      - KEYWORD: 키워드 검색 우선
      - HYBRID: 복합 검색 전략
      - GRAPH: 그래프 검색 중심

  Knowledge Graph System
  Neo4j 그래프 데이터베이스
  - 노드 타입: Document, Chunk, Entity, Synonym
  - 관계 타입: CONTAINS, MENTIONS, RELATES_TO, SYNONYM_OF
  - 자동 관계 추출: 엔티티 간 관계 자동 발견

  엔티티 관리
  # 지원 엔티티 타입
  PERSON, ORGANIZATION, LOCATION, DATE,
  MONEY, PERCENT, TIME, PRODUCT, EVENT

  Generation & QA System
  프롬프트 템플릿 엔진
  - 질문 유형별 최적화된 프롬프트
  - 컨텍스트 주입 전략
  - 다국어 지원

  답변 생성 파이프라인
  1. 질문 분석 및 분류
  2. 최적 검색 전략 선택
  3. 컨텍스트 수집 및 랭킹
  4. 답변 생성 및 검증
  5. 후처리 및 포맷팅

  ---
  
  4. API 엔드포인트
  Documents API (/api/documents)
  - POST /upload - 문서 업로드 및 처리
  - GET /list - 문서 목록 조회
  - DELETE /{id} - 문서 삭제

  Chat API (/api/chat)
  - POST /query - 질의응답
  - POST /sessions - 세션 생성
  - GET /sessions/{id}/history - 대화 히스토리

  QA API (/api/qa)
  - POST /ask - 단일 질문 처리
  - POST /batch - 배치 질문 처리

  Agent RAG API (/api/agent-rag)
  - POST /multi-agent-query - 멀티 에이전트 질의
  - GET /agent-status - 에이전트 상태 조회

  ---
  
  5. 데이터 저장소
  PostgreSQL - 구조적 데이터
  - 문서 메타데이터
  - 사용자 세션
  - 시스템 설정

  Weaviate - 벡터 데이터베이스
  - 문서 청크 임베딩
  - 의미적 검색 인덱스

  Neo4j - 그래프 데이터베이스
  - 지식 그래프
  - 엔티티 관계
  - 문서 구조

  ---
  
  6. 기능
  지능형 동의어 시스템
  - 문서별 자동 동의어 탐지
  - 도메인 특화 용어 매핑
  - 검색 확장 기능

  성능 최적화
  - 멀티 스레드 임베딩 생성
  - 캐싱 전략
  - 배치 처리 최적화

  보안 기능
  - 파일 타입 검증
  - 크기 제한
  - 입력 검증

  모니터링 & 로깅
  - 상세한 처리 로그
  - 성능 메트릭 수집
  - 에러 추적

📁 RAG Document Chatbot 디렉토리 구조
  rag-document-chatbot/
  ├── 📄 app.py                               # FastAPI 메인 애플리케이션
  ├── 📄 requirements.txt                     # Python 패키지 의존성
  ├── 📄 docker-compose.yml                   # Docker 컨테이너 설정
  ├── 📄 .env                                 # 환경 변수 설정
  ├── 📄 rag_chatbot.db                       # SQLite 데이터베이스
  ├── 📄 init-ollama.sh                       # Ollama 초기화 스크립트
  ├── 📁 static/                              # 정적 파일
  ├── 📁 templates/                           # HTML 템플릿
  ├── 📁 venv/                                # Python 가상환경
  └── 📁 app/                                 # 메인 애플리케이션 코드
      ├── 📄 __init__.py                      
      ├── 📄 main.py                          # FastAPI 앱 설정 및 라우터 등록
      │                                       
      ├── 📁 api/                             # API 엔드포인트
      │   └── 📁 v1/                          
      │       ├── 📄 __init__.py              
      │       ├── 📄 agent_rag.py             # 멀티 에이전트 RAG API
      │       ├── 📄 chat.py                  # 채팅 세션 API
      │       ├── 📄 documents.py             # 문서 업로드/관리 API
      │       ├── 📄 qa.py                    # 질의응답 API
      │       └── 📄 synonyms.py              # 동의어 관리 API
      │                                       
      ├── 📁 agents/                          # 멀티 에이전트 시스템
      │   ├── 📄 __init__.py                  
      │   ├── 📄 agent_system.py              # 통합 에이전트 시스템
      │   ├── 📄 base_agent.py                # 에이전트 기본 클래스
      │   ├── 📁 coordinators/                # 조정자 에이전트
      │   │   ├── 📄 __init__.py              
      │   │   ├── 📄 orchestrator.py          # 메인 오케스트레이터
      │   │   └── 📄 rag_coordinator.py       # RAG 전용 조정자
      │   └── 📁 specialized/                 # 전문 에이전트
      │       ├── 📄 __init__.py
      │       ├── 📄 search_agent.py          # 검색 전문 에이전트
      │       ├── 📄 enhanced_search_agent.py # 고급 검색 에이전트
      │       └── 📄 qa_agent.py              # QA 전문 에이전트
      │                                       
      ├── 📁 core/                            # 핵심 설정
      │   ├── 📄 __init__.py                  
      │   └── 📄 config.py                    # 시스템 설정 및 환경변수
      │                                       
      ├── 📁 infrastructure/                  # 인프라스트럭처 계층
      │   ├── 📄 __init__.py                  
      │   ├── 📁 database/                    # 데이터베이스 클라이언트
      │   │   ├── 📄 __init__.py              
      │   │   ├── 📄 neo4j_client.py          # Neo4j 그래프 DB 클라이언트
      │   │   └── 📄 sql_client.py            # PostgreSQL/SQLite 클라이언트
      │   └── 📁 external/                    # 외부 서비스 연동
      │       ├── 📄 __init__.py              
      │       ├── 📄 embedding_config.py      # 임베딩 설정
      │       ├── 📄 embedding_model.py       # 임베딩 모델 관리
      │       ├── 📄 text_processor.py        # 텍스트 전처리
      │       └── 📄 weaviate_client.py       # Weaviate 벡터 DB 클라이언트
      │                                       
      ├── 📁 models/                          # 데이터 모델
      │   ├── 📄 chat.py                      # 채팅 데이터 모델
      │   └── 📄 document.py                  # 문서 데이터 모델
      │                                       
      ├── 📁 rag/                             # RAG 시스템 핵심
      │   ├── 📁 generation/                  # 답변 생성
      │   │   ├── 📄 prompt_templates.py      # 프롬프트 템플릿
      │   │   └── 📄 qa_system.py             # 고급 QA 시스템
      │   ├── 📁 retrieval/                   # 검색 및 검색
      │   │   ├── 📄 embedder.py              # 문서 임베딩 처리
      │   │   ├── 📄 graph_retriever.py       # 그래프 기반 검색
      │   │   ├── 📄 question_analyzer.py     # 질문 분석기
      │   │   └── 📄 search_strategy.py       # 검색 전략 관리
      │   └── 📁 utils/                       # RAG 유틸리티
      │       ├── 📄 cypher_queries.py        # Cypher 쿼리 템플릿
      │       └── 📄 domain_config.py         # 도메인별 설정
      │
      ├── 📁 services/                        # 비즈니스 로직 서비스
      │   ├── 📁 document/                    # 문서 처리 서비스
      │   │   ├── 📄 __init__.py              
      │   │   ├── 📄 base_processor.py        # 기본 문서 처리기
      │   │   ├── 📄 pdf_processor.py         # PDF 처리기
      │   │   ├── 📄 docx_processor.py        # DOCX 처리기
      │   │   ├── 📄 excel_processor.py       # Excel 처리기
      │   │   └── 📄 hwp_processor.py         # HWP 처리기
      │   └── 📁 nlp/                         # 자연어 처리 서비스
      │       ├── 📄 __init__.py
      │       ├── 📄 entity_extractor.py      # 엔티티 추출기
      │       └── 📄 synonym_mapper.py        # 동의어 매핑
      │                                       
      └── 📁 utils/                           # 공통 유틸리티
          ├── 📄 logging_config.py            # 로깅 설정
          └── 📄 security.py                  # 보안 유틸리티

  
  
  아키텍처 계층 구조
  Presentation Layer (API)
  - REST API 엔드포인트 (/api/v1/)
  - FastAPI 라우터 및 HTTP 요청 처리

  Business Logic Layer (Services & Agents)
  - 멀티 에이전트 시스템 (/agents/)
  - 문서 처리 서비스 (/services/)
  - RAG 핵심 로직 (/rag/)

  Infrastructure Layer
  - 데이터베이스 클라이언트 (/infrastructure/database/)
  - 외부 서비스 연동 (/infrastructure/external/)
  - 설정 관리 (/core/)

  Data Layer
  - 데이터 모델 (/models/)
  - 유틸리티 (/utils/)

  주요 의존성 관계
  API Layer → Services Layer → Infrastructure Layer
      ↓           ↓                    ↓
    agents/    services/         infrastructure/
      ↓           ↓                    ↓
     rag/      models/              core/

