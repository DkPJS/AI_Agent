from typing import List, Dict, Any, Tuple
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.rag.retrieval.embedder import DocumentEmbedder
from app.services.nlp.synonym_mapper import SynonymMapper
from app.rag.utils.domain_config import DomainConfig
from app.rag.utils.cypher_queries import CypherQueries
import logging
import re

class GraphRetriever:
    """그래프 기반 문서 검색 클래스"""
    
    def __init__(self):
        """초기화"""
        self.embedder = DocumentEmbedder()
        self.synonym_mapper = SynonymMapper()
    
    async def retrieve(self, query: str, limit: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        그래프 기반 검색 및 컨텍스트 구성
        
        Args:
            query: 검색 쿼리
            limit: 결과 제한 수
            
        Returns:
            컨텍스트 문자열, 소스 정보 목록
        """
        # 확장 쿼리 생성 (동의어 포함)
        expanded_query = self._expand_query_with_synonyms(query)
        logging.info(f"Original query: '{query}', Expanded query: '{expanded_query}'")
        
        # 도메인별 최적화 설정 가져오기
        optimization = DomainConfig.get_optimization_for_query(query)
        vector_limit = optimization["vector_limit"]
        entity_limit = optimization["entity_limit"]
        final_limit = optimization["final_limit"]
        
        if DomainConfig.is_optimized_query(query):
            logging.info(f"{optimization['description']} 감지: 검색 파라미터 최적화")
        
        # 1. 벡터 검색으로 관련 청크 찾기
        logging.info("Starting vector similarity search")
        vector_results = await self.embedder.search_similar_chunks(expanded_query, limit=vector_limit)
        logging.info(f"Vector search found {len(vector_results)} results")
        
        # 결과 로깅 (디버깅 용도)
        for i, result in enumerate(vector_results):
            doc_id = result.get("document_id", "")
            filename = result.get("filename", "unknown")
            content_preview = result.get("content", "")[:50] + "..." if result.get("content") else ""
            logging.info(f"Vector result {i+1}: {filename} (ID: {doc_id}) - {content_preview}")
        
        # 2. 엔티티 기반 검색으로 추가 청크 찾기
        logging.info("Starting entity-based search")
        entity_results = self._search_by_entities(query, limit=entity_limit)
        logging.info(f"Entity search found {len(entity_results)} results")
        
        # 3. 그래프 관계를 활용한 검색으로 추가 청크 찾기
        logging.info("Starting graph relationship search")
        graph_results = self._search_related_in_graph(query, limit=2)
        logging.info(f"Graph relationship search found {len(graph_results)} results")
        
        # 4. 결과 통합 (중복 제거)
        combined_results = self._combine_results(
            vector_results, 
            entity_results, 
            graph_results, 
            limit=final_limit
        )
        logging.info(f"Combined search results: {len(combined_results)} chunks after deduplication")
        
        # 5. 컨텍스트 구성
        context_str = ""
        sources = []
        
        for result in combined_results:
            # 컨텍스트 문자열에 청크 추가
            content = result.get('content', '')
            if content:
                context_str += f"{content}\n\n"
            
            # 소스 정보 추가 - 페이지 정보 추출 및 메타데이터 추가
            filename = result.get("filename", "")
            metadata = result.get("metadata", {})
            
            # 페이지 정보 추출 시도
            page_info = None
            page_match = re.search(r'\[페이지\s+(\d+)\]', content)
            if page_match:
                page_info = page_match.group(1)
            
            source_info = {
                "document_id": result.get("document_id", ""),
                "filename": filename,
                "chunk_id": result.get("chunk_id", ""),
                "relevance": result.get("relevance", 0),
                "content": content,
                "page": page_info,
                "metadata": metadata
            }
            
            sources.append(source_info)
        
        if not context_str.strip():
            logging.warning(f"No results found for query: '{query}'")
            
        return context_str, sources
    
    def _combine_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        entity_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        검색 결과 통합 및 중복 제거
        
        Args:
            vector_results: 벡터 검색 결과
            entity_results: 엔티티 검색 결과
            graph_results: 그래프 검색 결과
            limit: 결과 제한 수
            
        Returns:
            중복 제거된 통합 결과
        """
        # 결과 통합
        all_results = []
        seen_chunks = set()
        
        # 벡터 검색 결과 추가 (우선순위 높음)
        for result in vector_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                all_results.append(result)
        
        # 엔티티 검색 결과 추가
        for result in entity_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                all_results.append(result)
        
        # 그래프 검색 결과 추가
        for result in graph_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                all_results.append(result)
        
        # 상위 N개 결과만 반환
        return all_results[:limit]
    
    def _expand_query_with_synonyms(self, query: str) -> str:
        """
        동의어를 포함한 확장 쿼리 생성
        
        Args:
            query: 원본 쿼리
            
        Returns:
            확장된 쿼리
        """
        return self.synonym_mapper.expand_query(query)
    
    def _search_by_entities(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        쿼리에서 엔티티를 추출하여 관련 청크 검색
        
        Args:
            query: 검색 쿼리
            limit: 각 엔티티별 결과 제한
            
        Returns:
            엔티티 관련 청크 목록
        """
        from app.services.nlp.entity_extractor import EntityExtractor
        
        # 엔티티 추출기 초기화
        entity_extractor = EntityExtractor()
        
        # 쿼리에서 엔티티 추출
        entities = entity_extractor.extract_entities(query)
        
        # 결과 목록
        results = []
        
        # Neo4j 클라이언트 초기화 및 사용
        with Neo4jClient() as neo4j_client:
            # 각 엔티티에 대해 검색
            for entity in entities[:3]:  # 상위 3개 엔티티만 사용
                entity_name = entity["name"]
                
                # 엔티티로 청크 검색
                entity_chunks = self._search_by_entity_name(neo4j_client, entity_name, limit=1)
                results.extend(entity_chunks)
                
                # 동의어 검색
                synonyms = self.synonym_mapper.get_synonyms(entity_name)
                for synonym in list(synonyms)[:2]:  # 각 엔티티당 최대 2개 동의어만 사용
                    synonym_chunks = self._search_by_entity_name(neo4j_client, synonym, limit=1)
                    results.extend(synonym_chunks)
        
        return results
    
    def _search_by_entity_name(self, neo4j_client, entity_name: str, limit: int = 1) -> List[Dict[str, Any]]:
        """엔티티명으로 청크 검색"""
        try:
            with neo4j_client.driver.session() as session:
                result = session.run(
                    CypherQueries.get_entity_chunks_query(),
                    entity_name=entity_name,
                    limit=limit
                )
                return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"엔티티 '{entity_name}' 검색 중 오류: {e}")
            return []
    
    def _search_related_in_graph(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        그래프 관계를 활용한 검색 (Neo4j에서 복잡한 패턴 매칭)
        
        Args:
            query: 검색 쿼리
            limit: 결과 제한
            
        Returns:
            그래프 관계 기반 청크 목록
        """
        try:
            with Neo4jClient() as neo4j_client:
                with neo4j_client.driver.session() as session:
                    result = session.run(
                        CypherQueries.get_related_chunks_query(),
                        query_text=query,
                        limit=limit
                    )
                    return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"그래프 검색 중 오류 발생: {e}")
            return []