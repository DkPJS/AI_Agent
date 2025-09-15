"""질문 분석 모듈"""
from typing import List, Tuple
from app.services.nlp.entity_extractor import EntityExtractor

class QuestionAnalyzer:
    """질문 유형 및 엔티티 분석기"""
    
    QUESTION_TYPE_KEYWORDS = {
        "factual": ["뭐", "무엇", "어떤", "어떻게"],
        "summary": ["요약", "간략", "정리"],
        "comparison": ["비교", "차이", "다른점"],
        "causal": ["왜", "이유", "원인"],
        "procedural": ["방법", "단계", "어떻게 하", "어떻게 만"]
    }
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
    
    def analyze_question(self, question: str) -> Tuple[str, List[str]]:
        """
        질문 유형 및 핵심 엔티티 분석
        
        Args:
            question: 분석할 질문
            
        Returns:
            (질문_유형, 핵심_엔티티_목록)
        """
        question_type = self._classify_question_type(question)
        focus_entities = self._extract_focus_entities(question)
        
        return question_type, focus_entities
    
    def _classify_question_type(self, question: str) -> str:
        """질문 유형 분류"""
        question_lower = question.lower()
        
        for qtype, keywords in self.QUESTION_TYPE_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                return qtype
        
        return "general"
    
    def _extract_focus_entities(self, question: str) -> List[str]:
        """핵심 엔티티 추출"""
        entities = self.entity_extractor.extract_entities(question)
        return [e["name"] for e in entities]