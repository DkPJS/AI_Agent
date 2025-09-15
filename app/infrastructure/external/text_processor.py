"""텍스트 전처리 유틸리티"""
import re
from typing import Set
from app.infrastructure.external.embedding_config import EmbeddingConstants

class TextProcessor:
    """한국어 텍스트 전처리"""
    
    @staticmethod
    def preprocess_korean_text(text: str) -> str:
        """한국어 텍스트 전처리"""
        if not text:
            return ""
        
        # 기본 정제
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 불용어 제거
        words = text.split()
        filtered_words = [
            word for word in words 
            if word not in EmbeddingConstants.KOREAN_STOPWORDS and len(word) > 1
        ]
        
        return ' '.join(filtered_words)
    
    @staticmethod
    def has_direct_keyword_match(query: str, content: str) -> bool:
        """쿼리의 핵심 키워드가 내용에 직접 포함되는지 확인"""
        query_words = set(TextProcessor.preprocess_korean_text(query).split())
        content_words = set(TextProcessor.preprocess_korean_text(content).split())
        
        # 2글자 이상 단어만 고려
        query_words = {word for word in query_words if len(word) >= 2}
        
        # 교집합이 있으면 직접 매칭
        return len(query_words & content_words) > 0