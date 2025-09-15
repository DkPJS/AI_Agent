"""검색 전략 모듈"""
from typing import List, Dict, Any, Tuple
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.rag.retrieval.embedder import DocumentEmbedder
from app.services.nlp.synonym_mapper import SynonymMapper
from app.core.config import settings
from app.utils.model_manager import get_shared_embedding_model

class SearchStrategy:
    """질문 유형별 검색 전략"""
    
    def __init__(self):
        self.embedder = DocumentEmbedder()
        self.synonym_mapper = SynonymMapper()
        # 임베딩 모델은 공유 모델 매니저에서 가져옴
        self._shared_embedding_model = None
    
    async def execute_search(
        self, 
        question: str, 
        question_type: str, 
        focus_entities: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """질문 유형별 최적 검색 전략 실행"""
        sources = []
        
        if question_type == "factual":
            sources = await self._factual_search(question, focus_entities)
        elif question_type == "comparison":
            sources = await self._comparison_search(question, focus_entities)
        elif question_type == "summary":
            sources = await self._summary_search(question)
        elif question_type == "procedural":
            sources = await self._procedural_search(question)
        else:
            sources = await self._general_search(question)
        
        unique_sources = self._deduplicate_sources(sources)
        context = self._build_context(unique_sources)
        
        return context, unique_sources
    
    async def _factual_search(self, question: str, focus_entities: List[str]) -> List[Dict[str, Any]]:
        """사실 기반 질문 검색"""
        sources = []
        
        # 하이브리드 검색
        hybrid_results = self.embedder.hybrid_search(
            question, 
            limit=settings.DEFAULT_SEARCH_LIMIT,
            semantic_weight=0.6, 
            keyword_weight=0.4
        )
        sources.extend(hybrid_results)
        
        # 엔티티 기반 검색
        for entity in focus_entities:
            synonyms = self.synonym_mapper.get_synonyms(entity)
            all_terms = [entity] + list(synonyms)
            
            with Neo4jClient() as neo4j_client:
                for term in all_terms:
                    entity_results = neo4j_client.search_by_entity(term, limit=2)
                    sources.extend(entity_results)
        
        return sources
    
    async def _comparison_search(self, question: str, focus_entities: List[str]) -> List[Dict[str, Any]]:
        """비교 질문 검색"""
        sources = []
        
        # 하이브리드 검색
        hybrid_results = self.embedder.hybrid_search(
            question, 
            limit=4,
            semantic_weight=0.5, 
            keyword_weight=0.5
        )
        sources.extend(hybrid_results)
        
        # 여러 엔티티를 모두 포함하는 청크 검색
        if len(focus_entities) >= 2:
            comparison_results = self._find_chunks_with_multiple_entities(focus_entities)
            sources.extend(comparison_results)
        
        return sources
    
    async def _summary_search(self, question: str) -> List[Dict[str, Any]]:
        """요약 질문 검색"""
        return self.embedder.hybrid_search(
            question, 
            limit=6,
            semantic_weight=0.8, 
            keyword_weight=0.2
        )
    
    async def _procedural_search(self, question: str) -> List[Dict[str, Any]]:
        """절차적 질문 검색"""
        return self.embedder.hybrid_search(
            question, 
            limit=settings.DEFAULT_SEARCH_LIMIT,
            semantic_weight=0.4, 
            keyword_weight=0.6
        )
    
    async def _general_search(self, question: str) -> List[Dict[str, Any]]:
        """일반 질문 검색"""
        sources = []
        
        # 하이브리드 검색
        hybrid_results = self.embedder.hybrid_search(
            question, 
            limit=settings.DEFAULT_SEARCH_LIMIT,
            semantic_weight=settings.SEMANTIC_WEIGHT, 
            keyword_weight=settings.KEYWORD_WEIGHT
        )
        sources.extend(hybrid_results)
        
        # 키워드 검색
        with Neo4jClient() as neo4j_client:
            keyword_results = neo4j_client.search_related_chunks(question, limit=2)
            sources.extend(keyword_results)
        
        return sources
    
    def _find_chunks_with_multiple_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """여러 엔티티가 모두 등장하는 청크 검색"""
        if not entities or len(entities) < 2:
            return []
        
        query = "MATCH (c:Chunk) WHERE "
        conditions = [f"EXISTS ((:Entity {{name: $entity{i}}})<-[:MENTIONS]-(c))" 
                     for i, entity in enumerate(entities)]
        
        query += " AND ".join(conditions)
        query += """
        MATCH (doc:Document)-[:CONTAINS]->(c)
        RETURN 
            c.id AS chunk_id,
            c.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            1.0 AS relevance
        LIMIT 3
        """
        
        params = {f"entity{i}": entity for i, entity in enumerate(entities)}
        
        with Neo4jClient() as neo4j_client:
            with neo4j_client.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """소스 중복 제거 및 정렬"""
        seen_chunks = set()
        unique_sources = []
        
        for source in sources:
            chunk_id = source.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_sources.append(source)
        
        # 관련성 점수 기준 정렬
        return sorted(unique_sources, key=lambda x: x.get("relevance", 0.0), reverse=True)[:5]
    
    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """컨텍스트 구성"""
        context_chunks = []
        
        for source in sources:
            if "content" in source:
                filename = source.get("filename", "문서")
                chunk_content = source["content"]
                
                if "[파일:" not in chunk_content:
                    chunk_content = f"[파일: {filename}]\n{chunk_content}"
                
                context_chunks.append(chunk_content)
        
        return "\n\n".join(context_chunks)