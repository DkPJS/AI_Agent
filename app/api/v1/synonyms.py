from fastapi import APIRouter, Depends, HTTPException, Body, status
from typing import List, Dict, Optional
from pydantic import BaseModel
from app.services.nlp.synonym_mapper import SynonymMapper
from app.infrastructure.database.neo4j_client import Neo4jClient

router = APIRouter(prefix="/api/synonyms", tags=["synonyms"])

class SynonymPair(BaseModel):
    """동의어 쌍 모델"""
    term: str
    synonym: str

class SynonymGroup(BaseModel):
    """동의어 그룹 모델"""
    term: str
    synonyms: List[str]

# 싱글톤 인스턴스
_synonym_mapper = None

def get_synonym_mapper():
    """동의어 매퍼 싱글톤 인스턴스 가져오기"""
    global _synonym_mapper
    if _synonym_mapper is None:
        _synonym_mapper = SynonymMapper()
    return _synonym_mapper

@router.post("/", status_code=status.HTTP_201_CREATED)
async def add_synonym(pair: SynonymPair):
    """새 동의어 쌍 추가"""
    mapper = get_synonym_mapper()
    
    # 동의어 매퍼에 저장
    mapper.add_synonym_pair(pair.term, pair.synonym)
    
    # Neo4j 그래프에도 저장
    with Neo4jClient() as neo4j_client:
        neo4j_client.add_synonym_relation(pair.term, pair.synonym)
    
    return {"status": "success", "message": "동의어가 추가되었습니다."}

@router.get("/{term}")
async def get_synonyms(term: str):
    """특정 용어의 동의어 목록 조회"""
    mapper = get_synonym_mapper()
    synonyms = mapper.get_synonyms(term)
    return {"term": term, "synonyms": list(synonyms)}

@router.post("/batch", status_code=status.HTTP_201_CREATED)
async def add_synonym_group(group: SynonymGroup):
    """동의어 그룹 한번에 추가"""
    mapper = get_synonym_mapper()
    
    # Neo4j 클라이언트 초기화
    with Neo4jClient() as neo4j_client:
        # 주요 용어와 각 동의어 간의 관계 추가
        for synonym in group.synonyms:
            # 동의어 매퍼에 저장
            mapper.add_synonym_pair(group.term, synonym)
            
            # Neo4j 그래프에 저장
            neo4j_client.add_synonym_relation(group.term, synonym)
    
    return {"status": "success", "message": "동의어 그룹이 추가되었습니다."}

@router.post("/detect")
async def detect_synonyms(text: str = Body(..., embed=True)):
    """텍스트에서 동의어 자동 탐지"""
    mapper = get_synonym_mapper()
    synonyms = mapper.detect_synonyms_from_document(text)
    
    # Dict[str, Set[str]]를 Dict[str, List[str]]로 변환
    result = {k: list(v) for k, v in synonyms.items()}
    return {"detected_synonyms": result}

@router.get("/")
async def list_all_synonyms():
    """모든 동의어 관계 조회"""
    mapper = get_synonym_mapper()
    
    # Dict[str, Set[str]]를 Dict[str, List[str]]로 변환
    result = {k: list(v) for k, v in mapper.synonym_dict.items()}
    return {"synonyms": result}