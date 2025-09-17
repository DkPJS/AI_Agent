"""문장 간 유사도 계산 모듈"""
import logging
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.model_manager import get_shared_embedding_model


class SimilarityCalculator:
    """문장 간 유사도 계산 클래스"""

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: 유사도 임계값
        """
        self.similarity_threshold = similarity_threshold
        self._embedding_cache = {}

    def calculate_sentence_similarities(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        문장들 간의 유사도 행렬 계산

        Args:
            sentences: 문장 목록

        Returns:
            Optional[np.ndarray]: 유사도 행렬 (실패시 None)
        """
        if len(sentences) < 2:
            return None

        try:
            embeddings = self._get_embeddings_for_sentences(sentences)
            if not embeddings:
                return None

            # 유사도 행렬 계산
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix

        except Exception as e:
            logging.error(f"유사도 계산 실패: {e}")
            return None

    def find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """
        의미적 경계점 찾기

        Args:
            sentences: 문장 목록

        Returns:
            List[int]: 경계점이 되는 문장 인덱스 목록
        """
        if len(sentences) < 3:
            return []

        try:
            embeddings = self._get_embeddings_for_sentences(sentences)
            if not embeddings:
                return []

            boundaries = []
            embeddings_array = np.array(embeddings)

            # 인접한 문장들 간의 유사도 계산
            for i in range(1, len(sentences)):
                similarity = cosine_similarity(
                    [embeddings_array[i]],
                    [embeddings_array[i-1]]
                )[0][0]

                # 유사도가 임계값보다 낮으면 경계점으로 판단
                if similarity < self.similarity_threshold:
                    boundaries.append(i)

            return boundaries

        except Exception as e:
            logging.error(f"의미적 경계점 탐지 실패: {e}")
            return []

    def calculate_pairwise_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간 유사도 계산

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            float: 유사도 점수 (0-1)
        """
        try:
            embeddings = self._get_embeddings_for_sentences([text1, text2])
            if len(embeddings) != 2:
                return 0.0

            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception as e:
            logging.error(f"쌍별 유사도 계산 실패: {e}")
            return 0.0

    def _get_embeddings_for_sentences(self, sentences: List[str]) -> List[List[float]]:
        """
        문장들에 대한 임베딩 생성 (캐시 활용)

        Args:
            sentences: 문장 목록

        Returns:
            List[List[float]]: 임베딩 벡터 목록
        """
        try:
            # 캐시되지 않은 문장들 찾기
            uncached_sentences = []
            uncached_indices = []

            for i, sentence in enumerate(sentences):
                if sentence not in self._embedding_cache:
                    uncached_sentences.append(sentence)
                    uncached_indices.append(i)

            # 새로운 임베딩 생성
            if uncached_sentences:
                embedding_model = get_shared_embedding_model()
                new_embeddings = embedding_model.create_embeddings_batch(
                    uncached_sentences,
                    batch_size=32
                )

                # 캐시에 저장
                for sentence, embedding in zip(uncached_sentences, new_embeddings):
                    if embedding:  # 빈 임베딩이 아닌 경우만
                        self._embedding_cache[sentence] = embedding

            # 전체 임베딩 목록 구성
            embeddings = []
            for sentence in sentences:
                if sentence in self._embedding_cache:
                    embeddings.append(self._embedding_cache[sentence])
                else:
                    # 기본 임베딩 (모든 0)
                    embeddings.append([0.0] * 768)

            return embeddings

        except Exception as e:
            logging.error(f"임베딩 생성 실패: {e}")
            return []

    def clear_cache(self) -> None:
        """임베딩 캐시 초기화"""
        self._embedding_cache.clear()
        logging.debug("유사도 계산기 캐시 초기화 완료")

    def get_cache_stats(self) -> dict:
        """캐시 통계 정보 반환"""
        return {
            "cached_sentences": len(self._embedding_cache),
            "average_embedding_length": np.mean([
                len(emb) for emb in self._embedding_cache.values()
            ]) if self._embedding_cache else 0
        }