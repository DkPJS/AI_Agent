"""텍스트 분할 유틸리티"""
import re
import logging
from typing import List, Tuple
from .chunking_config import TextPatterns


class KoreanTextSplitter:
    """한국어 텍스트 분할 전용 클래스"""

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        한국어 텍스트를 문장으로 분할

        Args:
            text: 입력 텍스트

        Returns:
            List[str]: 분할된 문장 목록
        """
        if not text or not text.strip():
            return []

        sentences = []
        current_text = text.strip()

        # 패턴별로 시도
        for pattern in TextPatterns.SENTENCE_PATTERNS:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                sentences = [part.strip() for part in parts if part.strip()]
                break
        else:
            # 기본 분할
            sentences = [s.strip() for s in re.split(r'[.!?]+', current_text) if s.strip()]

        return KoreanTextSplitter._merge_short_sentences(sentences)

    @staticmethod
    def _merge_short_sentences(sentences: List[str], min_length: int = 50) -> List[str]:
        """
        짧은 문장들을 병합

        Args:
            sentences: 원본 문장 목록
            min_length: 최소 문장 길이

        Returns:
            List[str]: 병합된 문장 목록
        """
        if not sentences:
            return []

        merged_sentences = []
        current_sentence = ""

        for sentence in sentences:
            combined_length = len(current_sentence + sentence)

            if combined_length < min_length and current_sentence:
                current_sentence += " " + sentence
            else:
                if current_sentence:
                    merged_sentences.append(current_sentence.strip())
                current_sentence = sentence

        # 마지막 문장 추가
        if current_sentence:
            merged_sentences.append(current_sentence.strip())

        return merged_sentences

    @staticmethod
    def detect_pattern_boundaries(text: str, patterns: List[str]) -> List[Tuple[int, str]]:
        """
        텍스트에서 특정 패턴의 경계점 탐지

        Args:
            text: 대상 텍스트
            patterns: 탐지할 패턴 목록

        Returns:
            List[Tuple[int, str]]: (위치, 매칭된_패턴) 튜플 목록
        """
        boundaries = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in patterns:
                if re.match(pattern, line_stripped):
                    boundaries.append((i, pattern))
                    break

        return boundaries


class PatternMatcher:
    """패턴 매칭 유틸리티"""

    @staticmethod
    def is_topic_boundary(sentence: str) -> bool:
        """주제 경계 여부 판단"""
        return PatternMatcher._matches_any_pattern(
            sentence, TextPatterns.TOPIC_INDICATORS
        )

    @staticmethod
    def is_header_pattern(sentence: str) -> bool:
        """헤더 패턴 여부 판단"""
        return PatternMatcher._matches_any_pattern(
            sentence, TextPatterns.HEADER_PATTERNS
        )

    @staticmethod
    def is_step_pattern(sentence: str) -> bool:
        """단계 패턴 여부 판단"""
        return PatternMatcher._matches_any_pattern(
            sentence, TextPatterns.STEP_PATTERNS
        )

    @staticmethod
    def is_clause_pattern(sentence: str) -> bool:
        """조항 패턴 여부 판단"""
        return PatternMatcher._matches_any_pattern(
            sentence, TextPatterns.CLAUSE_PATTERNS
        )

    @staticmethod
    def _matches_any_pattern(text: str, patterns: List[str]) -> bool:
        """
        텍스트가 패턴 중 하나와 일치하는지 확인

        Args:
            text: 확인할 텍스트
            patterns: 패턴 목록

        Returns:
            bool: 매칭 여부
        """
        text_stripped = text.strip()
        return any(
            re.search(pattern, text_stripped, re.IGNORECASE)
            for pattern in patterns
        )