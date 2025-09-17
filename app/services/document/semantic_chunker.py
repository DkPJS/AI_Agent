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
        # 동적 임계값으로 시작
        self.adaptive_threshold = self.config.similarity_threshold
        self.similarity_calculator = SimilarityCalculator(
            similarity_threshold=self.adaptive_threshold
        )

    def chunk_document(self, text: str, document_type: str = "general") -> List[Dict[str, Any]]:
        """
        적응형 의미 기반 문서 청킹

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

        # 2. 문서 특성 분석 및 임계값 조정
        doc_characteristics = self._analyze_document_characteristics(text, sentences)
        optimized_threshold = self._calculate_optimal_threshold(doc_characteristics, document_type)

        # 3. 임계값 업데이트
        self._update_similarity_threshold(optimized_threshold)

        # 4. 문서 유형별 청킹 전략 적용
        doc_type_enum = self._get_document_type_enum(document_type)
        chunks = self._apply_chunking_strategy(sentences, doc_type_enum)

        # 5. 청킹 품질 평가 및 필요시 재조정
        chunks = self._evaluate_and_improve_chunks(chunks, doc_characteristics)

        # 6. 메타데이터 추가 및 반환
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

    def _analyze_document_characteristics(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """문서 특성 분석"""
        characteristics = {}

        # 텍스트 길이 및 복잡도
        characteristics["total_length"] = len(text)
        characteristics["sentence_count"] = len(sentences)
        characteristics["avg_sentence_length"] = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

        # 구조화 정도
        import re
        structured_patterns = len(re.findall(r'^\d+\.|\n\d+\.|\n-|제\d+조|제\d+항', text))
        characteristics["structure_density"] = structured_patterns / len(sentences) if sentences else 0

        # 정보 밀도
        korean_chars = len(re.findall(r'[가-힣]', text))
        numbers = len(re.findall(r'\d+', text))
        characteristics["info_density"] = (korean_chars + numbers * 2) / len(text) if text else 0

        # 문장 유사성 분산도 (문서의 일관성 측정)
        if len(sentences) >= 5:
            try:
                # 샘플 문장들의 유사도 계산
                sample_sentences = sentences[::max(1, len(sentences)//10)][:10]
                similarities = []
                for i in range(len(sample_sentences)-1):
                    for j in range(i+1, len(sample_sentences)):
                        # 간단한 어휘 유사도 계산
                        words1 = set(sample_sentences[i].split())
                        words2 = set(sample_sentences[j].split())
                        if words1 and words2:
                            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                            similarities.append(similarity)

                characteristics["coherence_variance"] = np.var(similarities) if similarities else 0.5
            except:
                characteristics["coherence_variance"] = 0.5
        else:
            characteristics["coherence_variance"] = 0.3

        return characteristics

    def _calculate_optimal_threshold(self, characteristics: Dict[str, Any], document_type: str) -> float:
        """문서 특성 기반 최적 임계값 계산"""
        base_threshold = self.config.similarity_threshold

        # 문서 유형별 기본 조정
        type_adjustments = {
            "general": 0.0,
            "report": -0.05,    # 보고서는 구조가 있어 임계값 낮춤
            "manual": -0.1,     # 매뉴얼은 단계별로 나누어야 함
            "contract": -0.03   # 계약서는 조항별 분리
        }

        adjusted_threshold = base_threshold + type_adjustments.get(document_type, 0.0)

        # 구조화 정도에 따른 조정
        structure_density = characteristics.get("structure_density", 0)
        if structure_density > 0.3:  # 구조화된 문서
            adjusted_threshold -= 0.08
        elif structure_density > 0.1:
            adjusted_threshold -= 0.04

        # 정보 밀도에 따른 조정
        info_density = characteristics.get("info_density", 0.5)
        if info_density > 0.8:  # 정보 밀도가 높으면 더 세밀하게 분할
            adjusted_threshold -= 0.05
        elif info_density < 0.3:  # 정보 밀도가 낮으면 더 크게 분할
            adjusted_threshold += 0.05

        # 일관성에 따른 조정
        coherence_variance = characteristics.get("coherence_variance", 0.5)
        if coherence_variance > 0.6:  # 일관성이 낮으면 더 세밀하게
            adjusted_threshold -= 0.03
        elif coherence_variance < 0.2:  # 일관성이 높으면 더 크게
            adjusted_threshold += 0.03

        # 임계값 범위 제한
        final_threshold = max(0.3, min(0.9, adjusted_threshold))

        logging.info(f"임계값 최적화: {base_threshold:.3f} → {final_threshold:.3f} "
                    f"(구조: {structure_density:.3f}, 밀도: {info_density:.3f}, 일관성분산: {coherence_variance:.3f})")

        return final_threshold

    def _update_similarity_threshold(self, new_threshold: float):
        """유사도 임계값 업데이트"""
        self.adaptive_threshold = new_threshold
        self.similarity_calculator.similarity_threshold = new_threshold

    def _evaluate_and_improve_chunks(self, chunks: List[str], characteristics: Dict[str, Any]) -> List[str]:
        """청킹 품질 평가 및 개선"""
        if not chunks:
            return chunks

        # 청크 품질 지표 계산
        quality_metrics = self._calculate_chunk_quality_metrics(chunks)

        # 품질이 낮으면 재조정
        if quality_metrics["overall_quality"] < 0.6:
            logging.info(f"청킹 품질 낮음 ({quality_metrics['overall_quality']:.3f}), 재조정 수행")
            improved_chunks = self._improve_chunk_quality(chunks, quality_metrics, characteristics)
            return improved_chunks

        return chunks

    def _calculate_chunk_quality_metrics(self, chunks: List[str]) -> Dict[str, float]:
        """청크 품질 지표 계산"""
        if not chunks:
            return {"overall_quality": 0.0}

        metrics = {}

        # 크기 일관성
        chunk_lengths = [len(chunk) for chunk in chunks]
        length_variance = np.var(chunk_lengths) / (np.mean(chunk_lengths) ** 2) if chunk_lengths else 1.0
        metrics["size_consistency"] = max(0.0, 1.0 - length_variance)

        # 적정 크기 비율
        optimal_size_count = sum(1 for length in chunk_lengths
                               if self.config.min_chunk_size <= length <= self.config.max_chunk_size)
        metrics["size_appropriateness"] = optimal_size_count / len(chunks) if chunks else 0.0

        # 내용 완성도 (문장 끝으로 끝나는 비율)
        complete_chunks = sum(1 for chunk in chunks if chunk.strip().endswith(('.', '다', '음', '됨')))
        metrics["completeness"] = complete_chunks / len(chunks) if chunks else 0.0

        # 전체 품질 점수
        metrics["overall_quality"] = (
            metrics["size_consistency"] * 0.3 +
            metrics["size_appropriateness"] * 0.4 +
            metrics["completeness"] * 0.3
        )

        return metrics

    def _improve_chunk_quality(self, chunks: List[str], quality_metrics: Dict[str, float],
                              characteristics: Dict[str, Any]) -> List[str]:
        """청킹 품질 개선"""
        # 크기가 너무 작은 청크들 병합
        improved_chunks = []
        current_chunk = ""

        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= self.config.max_chunk_size:
                current_chunk = current_chunk + " " + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    improved_chunks.append(current_chunk.strip())
                current_chunk = chunk

        if current_chunk:
            improved_chunks.append(current_chunk.strip())

        # 너무 큰 청크 분할
        final_chunks = []
        for chunk in improved_chunks:
            if len(chunk) > self.config.max_chunk_size * 1.5:
                # 문장 단위로 재분할
                sentences = self.text_splitter.split_into_sentences(chunk)
                sub_chunks = self._fallback_chunking(sentences)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

