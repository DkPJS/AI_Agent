"""도메인별 검색 최적화 설정"""
from typing import Dict, List
from app.core.config import settings

class DomainConfig:
    """도메인별 검색 설정"""
    
    # 특정 도메인 키워드별 검색 최적화 설정
    DOMAIN_OPTIMIZATIONS = {
        "과제선정_평가": {
            "keywords": ["과제선정", "평가기준", "평가 기준", "선정기준", "선정 기준"],
            "vector_limit": 8,
            "entity_limit": 3,
            "final_limit": 7,
            "description": "과제선정 평가 기준 관련 쿼리"
        },
        "연구개발": {
            "keywords": ["연구개발", "R&D", "기술개발", "혁신"],
            "vector_limit": 6,
            "entity_limit": 2,
            "final_limit": 6,
            "description": "연구개발 관련 쿼리"
        },
        "지원사업": {
            "keywords": ["지원사업", "보조금", "지원금", "사업공고"],
            "vector_limit": 7,
            "entity_limit": 3,
            "final_limit": 6,
            "description": "지원사업 관련 쿼리"
        }
    }
    
    # 기본 검색 설정
    DEFAULT_SETTINGS = {
        "vector_limit": settings.DEFAULT_SEARCH_LIMIT,
        "entity_limit": 2,
        "final_limit": settings.DEFAULT_SEARCH_LIMIT,
        "description": "일반 쿼리"
    }
    
    @classmethod
    def get_optimization_for_query(cls, query: str) -> Dict:
        """쿼리에 맞는 최적화 설정 반환"""
        query_lower = query.lower()
        
        for domain, config in cls.DOMAIN_OPTIMIZATIONS.items():
            if any(keyword in query_lower for keyword in config["keywords"]):
                return config
        
        return cls.DEFAULT_SETTINGS
    
    @classmethod
    def is_optimized_query(cls, query: str) -> bool:
        """최적화 대상 쿼리인지 확인"""
        return cls.get_optimization_for_query(query) != cls.DEFAULT_SETTINGS