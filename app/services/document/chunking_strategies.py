"""문서 유형별 청킹 전략 구현"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .chunking_config import DocumentType, ChunkingConfig
from .text_splitter import PatternMatcher
from .similarity_calculator import SimilarityCalculator


class ChunkingStrategy(ABC):
    """청킹 전략 추상 기본 클래스"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.pattern_matcher = PatternMatcher()

    @abstractmethod
    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """문장들을 청크로 분할"""
        pass

    def _should_split_chunk(self, current_chunk: List[str], new_sentence: str) -> bool:
        """청크 분할 여부 결정"""
        current_size = sum(len(s) for s in current_chunk)
        new_size = current_size + len(new_sentence)

        # 크기 기준 분할
        if new_size > self.config.max_chunk_size and current_size > self.config.min_chunk_size:
            return True

        return False

    def _finalize_chunks(self, chunk_groups: List[List[str]]) -> List[str]:
        """청크 그룹을 최종 청크 문자열로 변환"""
        chunks = []
        for chunk_group in chunk_groups:
            if chunk_group:
                chunk_content = ' '.join(chunk_group).strip()
                if chunk_content:
                    chunks.append(chunk_content)
        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """의미 기반 청킹 전략"""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.similarity_calculator = SimilarityCalculator(config.similarity_threshold)

    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """의미적 유사도를 기반으로 청킹"""
        if len(sentences) <= 5:
            return [' '.join(sentences)]

        try:
            # 의미적 경계점 찾기
            boundaries = self.similarity_calculator.find_semantic_boundaries(sentences)

            # 경계점을 기반으로 청킹
            chunks = []
            current_chunk = [sentences[0]]
            current_size = len(sentences[0])

            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_length = len(sentence)

                # 의미적 경계이거나 크기 초과시 분할
                should_split = (
                    i in boundaries or
                    self._should_split_chunk(current_chunk, sentence) or
                    self.pattern_matcher.is_topic_boundary(sentence)
                )

                if should_split and current_size > self.config.min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_length

            # 마지막 청크 추가
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        except Exception as e:
            logging.error(f"의미 기반 청킹 실패: {e}")
            return self._fallback_chunking(sentences)

    def _fallback_chunking(self, sentences: List[str]) -> List[str]:
        """의미 기반 청킹 실패 시 기본 청킹"""
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if self._should_split_chunk(current_chunk, sentence):
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class StructuredChunkingStrategy(ChunkingStrategy):
    """구조 기반 청킹 전략 (보고서, 매뉴얼, 계약서용)"""

    def __init__(self, config: ChunkingConfig, pattern_detector_method: str):
        super().__init__(config)
        self.pattern_detector_method = pattern_detector_method

    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """구조적 패턴을 기반으로 청킹"""
        detector_method = getattr(self.pattern_matcher, self.pattern_detector_method)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 구조적 패턴 감지
            is_boundary = detector_method(sentence)

            if is_boundary and current_chunk and current_size > self.config.min_chunk_size:
                # 기존 청크 완료 후 새 청크 시작
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_length
            elif self._should_split_chunk(current_chunk, sentence):
                # 크기 기준 분할
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class ChunkingStrategyFactory:
    """청킹 전략 팩토리"""

    @staticmethod
    def create_strategy(doc_type: DocumentType, config: ChunkingConfig) -> ChunkingStrategy:
        """문서 유형에 따른 청킹 전략 생성"""

        strategy_map = {
            DocumentType.GENERAL: lambda: SemanticChunkingStrategy(config),
            DocumentType.REPORT: lambda: StructuredChunkingStrategy(config, "is_header_pattern"),
            DocumentType.MANUAL: lambda: StructuredChunkingStrategy(config, "is_step_pattern"),
            DocumentType.CONTRACT: lambda: StructuredChunkingStrategy(config, "is_clause_pattern")
        }

        strategy_factory = strategy_map.get(doc_type)
        if strategy_factory:
            return strategy_factory()
        else:
            logging.warning(f"알 수 없는 문서 유형: {doc_type}, 기본 전략 사용")
            return SemanticChunkingStrategy(config)