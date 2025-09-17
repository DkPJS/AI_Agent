"""의미 기반 문서 청크 분할기 (리팩토링 버전)"""
import re
import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from .chunking_config import (
    DocumentType, ChunkingMethod, ChunkingConfig, ChunkingStrategies
)
from .chunking_strategies import ChunkingStrategyFactory
from .text_splitter import KoreanTextSplitter, PatternMatcher
from .similarity_calculator import SimilarityCalculator
from app.utils.model_manager import get_shared_embedding_model


class SemanticChunker:
    """의미적 유사도 기반 지능형 청크 분할기"""

    def __init__(self, config: ChunkingConfig = None):
        """
        Args:
            config: 청킹 설정 객체
        """
        self.config = config or ChunkingConfig()
        self.text_splitter = KoreanTextSplitter()
        self.pattern_matcher = PatternMatcher()
        self.similarity_calculator = SimilarityCalculator(
            similarity_threshold=self.config.similarity_threshold
        )

    def chunk_document(self, text: str, document_type: str = "general") -> List[Dict[str, Any]]:
        """
        문서를 의미 기반으로 청킹

        Args:
            text: 입력 텍스트
            document_type: 문서 유형 (general, report, manual, contract)

        Returns:
            List[Dict]: 청크 정보 리스트 (content, metadata 포함)
        """
        if not self._is_valid_text(text):
            return self._create_single_chunk(text, ChunkingMethod.SINGLE)

        # 1. 문장 분할
        sentences = self.text_splitter.split_into_sentences(text)
        if len(sentences) <= 3:
            return self._create_single_chunk(text, ChunkingMethod.SHORT)

        # 2. 문서 유형별 청킹 전략 적용
        doc_type_enum = self._get_document_type_enum(document_type)
        chunks = self._apply_chunking_strategy(sentences, doc_type_enum)

        # 3. 메타데이터 추가 및 반환
        return self._enrich_chunks_with_metadata(chunks, doc_type_enum)

    def _is_valid_text(self, text: str) -> bool:
        """텍스트 유효성 검사"""
        return text and len(text.strip()) >= self.config.min_chunk_size

    def _get_document_type_enum(self, document_type: str) -> DocumentType:
        """문자열을 DocumentType enum으로 변환"""
        try:
            return DocumentType(document_type.lower())
        except ValueError:
            logging.warning(f"알 수 없는 문서 유형: {document_type}, 기본값 사용")
            return DocumentType.GENERAL

    def _create_single_chunk(self, text: str, chunk_type: ChunkingMethod) -> List[Dict[str, Any]]:
        """단일 청크 생성"""
        return [{
            "content": text,
            "metadata": {
                "chunk_type": chunk_type.value,
                "size": len(text),
                "sentences": len(text.split('.')) if text else 0
            }
        }]

    def _apply_chunking_strategy(self, sentences: List[str], doc_type: DocumentType) -> List[str]:
        """문서 유형에 맞는 청킹 전략 적용"""
        try:
            strategy = ChunkingStrategyFactory.create_strategy(doc_type, self.config)
            return strategy.chunk_sentences(sentences)
        except Exception as e:
            logging.error(f"청킹 전략 적용 실패: {e}")
            return self._fallback_chunking(sentences)

    def _enrich_chunks_with_metadata(self, chunks: List[str], doc_type: DocumentType) -> List[Dict[str, Any]]:
        """청크에 메타데이터 추가"""
        result = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
                "document_type": doc_type.value,
                "chunking_method": "semantic",
                "has_numbers": bool(re.search(r'\d+', chunk)),
                "has_dates": bool(re.search(r'\d{4}년|\d{4}-\d{2}-\d{2}', chunk)),
            }
            result.append({
                "content": chunk,
                "metadata": metadata
            })
        return result











    def _fallback_chunking(self, sentences: List[str]) -> List[str]:
        """임베딩 실패 시 대체 청킹 (기본 크기 기반)"""
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_size + sentence_length > self.config.max_chunk_size and current_size > self.config.min_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

