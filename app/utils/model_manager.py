"""모델 관리자 - 싱글턴 패턴으로 모델 인스턴스 공유"""
import logging
from typing import Optional
from threading import Lock
from app.infrastructure.external.embedding_model import EmbeddingModel
from app.utils.gpu_utils import clear_gpu_cache

logger = logging.getLogger(__name__)

class ModelManager:
    """모델 인스턴스를 관리하는 싱글턴 클래스"""

    _instance: Optional['ModelManager'] = None
    _lock: Lock = Lock()

    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._embedding_model: Optional[EmbeddingModel] = None
        self._model_loading_lock = Lock()
        self._initialized = True
        logger.info("ModelManager 초기화 완료")

    def get_embedding_model(self) -> EmbeddingModel:
        """임베딩 모델 인스턴스 반환 (싱글턴)"""
        if self._embedding_model is None:
            with self._model_loading_lock:
                if self._embedding_model is None:
                    logger.info("공유 임베딩 모델 로딩 중...")
                    try:
                        # GPU 메모리 정리
                        clear_gpu_cache()

                        # 임베딩 모델 로드
                        self._embedding_model = EmbeddingModel()
                        logger.info("공유 임베딩 모델 로딩 완료")

                    except Exception as e:
                        logger.error(f"임베딩 모델 로딩 실패: {e}")
                        # GPU 메모리 정리 후 재시도
                        clear_gpu_cache()
                        import gc
                        gc.collect()

                        try:
                            logger.info("GPU 메모리 정리 후 재시도...")
                            self._embedding_model = EmbeddingModel()
                            logger.info("재시도로 임베딩 모델 로딩 성공")
                        except Exception as retry_error:
                            logger.error(f"재시도 실패: {retry_error}")
                            raise retry_error

        return self._embedding_model

    def clear_models(self) -> None:
        """모든 모델 인스턴스 정리"""
        with self._model_loading_lock:
            if self._embedding_model is not None:
                del self._embedding_model
                self._embedding_model = None
                clear_gpu_cache()
                logger.info("모델 인스턴스 정리 완료")

    def get_memory_info(self) -> dict:
        """모델 메모리 사용 정보"""
        from app.utils.gpu_utils import monitor_gpu_memory
        allocated, total = monitor_gpu_memory()

        return {
            "embedding_model_loaded": self._embedding_model is not None,
            "gpu_memory_allocated": f"{allocated:.3f}GB",
            "gpu_memory_total": f"{total:.1f}GB",
            "gpu_memory_usage": f"{(allocated/total*100) if total > 0 else 0:.1f}%"
        }

# 전역 모델 매니저 인스턴스
model_manager = ModelManager()

def get_shared_embedding_model() -> EmbeddingModel:
    """공유 임베딩 모델 반환"""
    return model_manager.get_embedding_model()

def clear_shared_models() -> None:
    """공유 모델 정리"""
    model_manager.clear_models()

def get_model_memory_info() -> dict:
    """모델 메모리 정보 반환"""
    return model_manager.get_memory_info()