"""임베딩 모델 관리"""
import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.utils.gpu_utils import setup_gpu_device, optimize_model_for_inference, clear_gpu_cache

class EmbeddingModel:
    """임베딩 모델 래퍼"""
    
    def __init__(self):
        self.device = setup_gpu_device()
        self.model = self._load_model()

        # 모델 최적화 (sentence-transformers는 내부적으로 모델을 관리하므로 주의 필요)
        if hasattr(self.model, '_modules'):
            try:
                clear_gpu_cache()
                logging.info("임베딩 모델 GPU 설정 완료")
            except Exception as e:
                logging.warning(f"임베딩 모델 최적화 중 오류: {e}")
    
    def _load_model(self) -> SentenceTransformer:
        """임베딩 모델 로드 (GPU 메모리 부족시 CPU fallback)"""
        models_to_try = [
            (settings.KOREAN_MODEL, "한국어 최적화"),
            (settings.EMBEDDING_MODEL, "기본"),
            ("sentence-transformers/all-MiniLM-L6-v2", "경량화")  # 더 가벼운 모델
        ]

        for model_name, model_type in models_to_try:
            try:
                # GPU에서 먼저 시도
                if self.device.type == 'cuda':
                    try:
                        clear_gpu_cache()  # 메모리 정리
                        model = SentenceTransformer(model_name, device=self.device)
                        logging.info(f"{model_type} 임베딩 모델 로드 성공: {model_name} (GPU)")
                        return model
                    except RuntimeError as gpu_error:
                        if "CUDA out of memory" in str(gpu_error):
                            logging.warning(f"GPU 메모리 부족으로 {model_type} 모델 로드 실패, CPU로 fallback")
                            # CPU로 fallback
                            model = SentenceTransformer(model_name, device='cpu')
                            logging.info(f"{model_type} 임베딩 모델 로드 성공: {model_name} (CPU)")
                            return model
                        else:
                            raise gpu_error
                else:
                    # CPU 모드
                    model = SentenceTransformer(model_name, device=self.device)
                    logging.info(f"{model_type} 임베딩 모델 로드 성공: {model_name} (CPU)")
                    return model

            except Exception as e:
                logging.warning(f"{model_type} 모델({model_name}) 로드 실패: {str(e)}")
                continue

        # 모든 모델 로드 실패
        raise Exception("모든 임베딩 모델 로드 실패")
    
    def create_embedding(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        if not text:
            return []
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logging.error(f"임베딩 생성 중 오류: {str(e)}")
            return []
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """배치 텍스트 임베딩 생성"""
        if not texts:
            return []
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logging.error(f"배치 임베딩 생성 중 오류: {str(e)}")
            return [self.create_embedding(text) for text in texts]