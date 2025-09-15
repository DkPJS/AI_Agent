"""
통합 로깅 설정 모듈
중앙화된 로그 설정으로 중복 방지
"""

import logging
import sys
from typing import Optional
from pathlib import Path

class SingletonLogger:
    """싱글톤 패턴으로 로거 중복 설정 방지"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
        return cls._instance
    
    def setup_logging(
        self, 
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
    ):
        """통합 로깅 설정"""
        if self._initialized:
            return
        
        # 기본 포맷
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 기존 핸들러 제거 (중복 방지)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 새 핸들러 설정
        handlers = []
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # 파일 핸들러 (선택적)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                format_string,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # 로거 설정
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True  # 기존 설정 강제 덮어쓰기
        )
        
        # 외부 라이브러리 로그 레벨 조정
        self._configure_external_loggers()
        
        self._initialized = True
        logging.info("통합 로깅 시스템 초기화 완료")
    
    def _configure_external_loggers(self):
        """외부 라이브러리 로거 설정"""
        # SQLAlchemy 로그 억제
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
        
        # HTTP 클라이언트 로그 억제
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        # Weaviate 클라이언트 로그 억제
        logging.getLogger('weaviate').setLevel(logging.ERROR)
        
        # Neo4j 로그 억제
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        
        # FastAPI/Uvicorn 로그 조정
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        logging.getLogger('fastapi').setLevel(logging.INFO)
        
        # Transformers/Torch 로그 억제
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        
        # 기타 노이즈 로그 억제
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

def setup_application_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_debug: bool = False
):
    """애플리케이션 로깅 설정 함수"""
    if enable_debug:
        level = logging.DEBUG
    
    logger_instance = SingletonLogger()
    logger_instance.setup_logging(level=level, log_file=log_file)
    
    return logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)

# 애플리케이션 시작 시 자동 초기화
def auto_setup():
    """자동 로깅 설정 (import 시 실행)"""
    import os
    
    # 환경변수에서 로그 레벨 확인
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level, logging.INFO)
    
    # 로그 파일 경로 (선택적)
    log_file = os.getenv('LOG_FILE', None)
    
    # 디버그 모드
    debug_mode = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    
    setup_application_logging(level=level, log_file=log_file, enable_debug=debug_mode)

# 모듈 import 시 자동 설정 실행
if not SingletonLogger()._initialized:
    auto_setup()