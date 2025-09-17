"""청킹 관련 설정 및 상수"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List


class DocumentType(Enum):
    """문서 유형 열거형"""
    GENERAL = "general"
    REPORT = "report"
    MANUAL = "manual"
    CONTRACT = "contract"


class ChunkingMethod(Enum):
    """청킹 방법 열거형"""
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    SINGLE = "single"
    SHORT = "short"


@dataclass
class ChunkingConfig:
    """청킹 설정 클래스"""
    min_chunk_size: int = 200
    max_chunk_size: int = 2000  # 청크 크기 증가로 청크 수 감소
    similarity_threshold: float = 0.6  # 임계값 낮춰서 빠른 처리
    window_size: int = 2  # 윈도우 크기 감소
    merge_short_sentences: bool = True
    min_sentence_length: int = 30  # 최소 길이 감소


class TextPatterns:
    """텍스트 패턴 상수"""

    # 문장 구분 패턴
    SENTENCE_PATTERNS = [
        r'[.!?]+\s+',      # 문장 부호 + 공백
        r'\n\n+',          # 단락 구분
        r'(?<=[다음음임함])\.\s',  # 한국어 종결어미 + 마침표
        r'(?<=[다음음임함])\s*\n'  # 한국어 종결어미 + 줄바꿈
    ]

    # 주제 전환 표현
    TOPIC_INDICATORS = [
        r'^(그런데|하지만|그러나|반면|한편)',  # 접속사
        r'^(다음으로|또한|아울러|더불어)',      # 전환 표현
        r'^(결론적으로|요약하면|마지막으로)',    # 결론 표현
    ]

    # 헤더 패턴
    HEADER_PATTERNS = [
        r'^\d+\.\s+',           # 1. 서론
        r'^제\d+장',            # 제1장
        r'^\d+\.\d+\s+',        # 1.1 배경
        r'^[가나다라마바사]\.\s+', # 가. 개요
    ]

    # 단계/절차 패턴
    STEP_PATTERNS = [
        r'^단계\s*\d+',
        r'^\d+단계',
        r'^step\s*\d+',
        r'^절차\s*\d+',
        r'^방법\s*\d+',
    ]

    # 조항 패턴
    CLAUSE_PATTERNS = [
        r'^제\d+조',            # 제1조
        r'^조\s*\d+',           # 조 1
        r'^항\s*\d+',           # 항 1
        r'^\(\d+\)',            # (1)
    ]


class ChunkingStrategies:
    """문서 유형별 청킹 전략"""

    @staticmethod
    def get_strategy_for_document_type(doc_type: DocumentType) -> Dict[str, any]:
        """문서 유형별 청킹 전략 반환"""
        strategies = {
            DocumentType.GENERAL: {
                "use_semantic_similarity": True,
                "respect_paragraph_breaks": True,
                "merge_short_chunks": True,
                "primary_patterns": TextPatterns.TOPIC_INDICATORS
            },
            DocumentType.REPORT: {
                "use_semantic_similarity": False,
                "respect_headers": True,
                "section_aware": True,
                "primary_patterns": TextPatterns.HEADER_PATTERNS
            },
            DocumentType.MANUAL: {
                "use_semantic_similarity": False,
                "step_aware": True,
                "procedure_focused": True,
                "primary_patterns": TextPatterns.STEP_PATTERNS
            },
            DocumentType.CONTRACT: {
                "use_semantic_similarity": False,
                "clause_aware": True,
                "legal_structure": True,
                "primary_patterns": TextPatterns.CLAUSE_PATTERNS
            }
        }

        return strategies.get(doc_type, strategies[DocumentType.GENERAL])