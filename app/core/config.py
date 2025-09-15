import os
from typing import Optional
from dotenv import load_dotenv
import logging

load_dotenv()

# 통합 로깅 설정 사용
from app.utils.logging_config import setup_application_logging, get_logger

# 애플리케이션 로깅 초기화
logger = setup_application_logging()

class Settings:
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    
    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot.db")
    
    # Docker 환경 감지
    IN_DOCKER: bool = os.path.exists('/.dockerenv') or bool(os.environ.get('DOCKER_CONTAINER'))
    
    # Neo4j 설정
    NEO4J_URI: str = os.getenv("NEO4J_URI", 
                               "bolt://neo4j:7687" if IN_DOCKER else "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Weaviate 설정
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", 
                                  "http://weaviate:8080" if IN_DOCKER else "http://localhost:8080")
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", 
                                     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    KOREAN_MODEL: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    
    # LLM 설정
    LLM_API_URL: str = os.getenv("LLM_API_URL", 
                                 "http://llm:11434/api/generate" if IN_DOCKER else "http://localhost:11434/api/generate")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
    
    # 업로드 설정
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./static/uploads")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "104857600"))
    
    # 동의어 사전 설정
    SYNONYM_DICT_PATH: str = os.getenv("SYNONYM_DICT_PATH", "./static/synonyms.json")
    ENABLE_WORD_VECTORS: bool = os.getenv("ENABLE_WORD_VECTORS", "False").lower() in ("true", "1", "t")
    WORD_VECTORS_PATH: str = os.getenv("WORD_VECTORS_PATH", "./static/word_vectors.bin")
    
    # 임베딩 설정
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))
    WEAVIATE_STARTUP_PERIOD: int = int(os.getenv("WEAVIATE_STARTUP_PERIOD", "30"))
    
    # 검색 설정 (성능 최적화)
    DEFAULT_SEARCH_LIMIT: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "3"))  # 5->3로 축소
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.8"))  # 시맨틱 검색 비중 증가
    KEYWORD_WEIGHT: float = float(os.getenv("KEYWORD_WEIGHT", "0.2"))  # 키워드 검색 비중 축소
    
    # 성능 최적화 설정
    ENABLE_SEARCH_CACHING: bool = os.getenv("ENABLE_SEARCH_CACHING", "True").lower() in ("true", "1", "t")
    CACHE_TTL_MINUTES: int = int(os.getenv("CACHE_TTL_MINUTES", "10"))
    MAX_ENTITY_EXTRACTION_TIME: float = float(os.getenv("MAX_ENTITY_EXTRACTION_TIME", "2.0"))
    SKIP_DUPLICATE_SEARCHES: bool = os.getenv("SKIP_DUPLICATE_SEARCHES", "True").lower() in ("true", "1", "t")

    # GPU 설정
    USE_GPU: bool = os.getenv("USE_GPU", "True").lower() in ("true", "1", "t")
    GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "False").lower() in ("true", "1", "t")
    
    @classmethod
    def validate(cls) -> None:
        """설정값 검증"""
        if cls.PORT < 1 or cls.PORT > 65535:
            raise ValueError(f"Invalid PORT: {cls.PORT}")
        if cls.MAX_UPLOAD_SIZE <= 0:
            raise ValueError(f"Invalid MAX_UPLOAD_SIZE: {cls.MAX_UPLOAD_SIZE}")
        if not (0 <= cls.SEMANTIC_WEIGHT <= 1):
            raise ValueError(f"SEMANTIC_WEIGHT must be between 0 and 1: {cls.SEMANTIC_WEIGHT}")
        if not (0 <= cls.KEYWORD_WEIGHT <= 1):
            raise ValueError(f"KEYWORD_WEIGHT must be between 0 and 1: {cls.KEYWORD_WEIGHT}")
        if abs(cls.SEMANTIC_WEIGHT + cls.KEYWORD_WEIGHT - 1.0) > 0.01:
            raise ValueError("SEMANTIC_WEIGHT + KEYWORD_WEIGHT must equal 1.0")

# 설정 인스턴스 생성 및 검증
settings = Settings()
settings.validate()

if settings.DEBUG:
    logger.info(f"환경: {'Docker' if settings.IN_DOCKER else '로컬'}")
    logger.info(f"Weaviate URL: {settings.WEAVIATE_URL}")
    logger.info(f"Neo4j URI: {settings.NEO4J_URI}")
    logger.info(f"LLM API URL: {settings.LLM_API_URL}")