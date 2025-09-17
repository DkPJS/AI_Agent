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
        스마트 검색 결과 통합 및 중복 제거

        Args:
            vector_results: 벡터 검색 결과
            entity_results: 엔티티 검색 결과
            graph_results: 그래프 검색 결과
            limit: 결과 제한 수

        Returns:
            품질 점수 기반 정렬된 통합 결과
        """
        # 검색 방법별 가중치 설정
        method_weights = {
            "vector": 0.4,    # 벡터 검색 신뢰도
            "entity": 0.3,    # 엔티티 검색 신뢰도
            "graph": 0.3      # 그래프 검색 신뢰도
        }

        # 결과 통합 및 점수 계산
        combined_chunks = {}

        # 벡터 검색 결과 처리
        for result in vector_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                base_score = result.get("relevance", 0.5)
                quality_score = self._calculate_content_quality(result.get("content", ""))
                combined_score = (base_score * method_weights["vector"]) + (quality_score * 0.2)

                combined_chunks[chunk_id] = {
                    **result,
                    "combined_score": combined_score,
                    "search_methods": ["vector"],
                    "method_scores": {"vector": base_score}
                }

        # 엔티티 검색 결과 처리
        for result in entity_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                base_score = result.get("relevance", 0.5)
                quality_score = self._calculate_content_quality(result.get("content", ""))
                entity_score = (base_score * method_weights["entity"]) + (quality_score * 0.2)

                if chunk_id in combined_chunks:
                    # 기존 청크에 점수 추가
                    combined_chunks[chunk_id]["combined_score"] += entity_score * 0.5  # 부스트
                    combined_chunks[chunk_id]["search_methods"].append("entity")
                    combined_chunks[chunk_id]["method_scores"]["entity"] = base_score
                else:
                    combined_chunks[chunk_id] = {
                        **result,
                        "combined_score": entity_score,
                        "search_methods": ["entity"],
                        "method_scores": {"entity": base_score}
                    }

        # 그래프 검색 결과 처리
        for result in graph_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                base_score = result.get("relevance", 0.5)
                quality_score = self._calculate_content_quality(result.get("content", ""))
                graph_score = (base_score * method_weights["graph"]) + (quality_score * 0.2)

                if chunk_id in combined_chunks:
                    # 기존 청크에 점수 추가
                    combined_chunks[chunk_id]["combined_score"] += graph_score * 0.4  # 부스트
                    combined_chunks[chunk_id]["search_methods"].append("graph")
                    combined_chunks[chunk_id]["method_scores"]["graph"] = base_score
                else:
                    combined_chunks[chunk_id] = {
                        **result,
                        "combined_score": graph_score,
                        "search_methods": ["graph"],
                        "method_scores": {"graph": base_score}
                    }

        # 다중 검색 방법에서 발견된 청크에 추가 부스트
        for chunk_id, chunk_data in combined_chunks.items():
            if len(chunk_data["search_methods"]) > 1:
                boost_factor = 1 + (len(chunk_data["search_methods"]) - 1) * 0.2
                chunk_data["combined_score"] *= boost_factor

        # 점수 기반 정렬 및 다양성 확보
        sorted_results = sorted(
            combined_chunks.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # 상위 결과 선택 (다양성 고려)
        final_results = self._ensure_diversity(sorted_results, limit)

        logging.info(f"검색 결과 통합 완료: {len(combined_chunks)}개 고유 청크 → {len(final_results)}개 최종 선택")
        return final_results

    def _calculate_content_quality(self, content: str) -> float:
        """콘텐츠 품질 점수 계산"""
        if not content:
            return 0.0

        quality_factors = []

        # 길이 적절성 (너무 짧거나 길지 않은 것이 좋음)
        length = len(content)
        if 100 <= length <= 1000:
            quality_factors.append(0.3)
        elif 50 <= length < 100 or 1000 < length <= 2000:
            quality_factors.append(0.2)
        else:
            quality_factors.append(0.1)

        # 문장 완성도 (마침표로 끝나는지)
        if content.strip().endswith(('.', '다', '음', '됨')):
            quality_factors.append(0.2)
        else:
            quality_factors.append(0.1)

        # 정보 밀도 (숫자, 한글 비율)
        import re
        korean_chars = len(re.findall(r'[가-힣]', content))
        numbers = len(re.findall(r'\d+', content))
        info_density = (korean_chars + numbers * 2) / len(content) if content else 0

        if info_density > 0.7:
            quality_factors.append(0.3)
        elif info_density > 0.5:
            quality_factors.append(0.2)
        else:
            quality_factors.append(0.1)

        # 구조화 정도 (제목, 번호 등)
        if re.search(r'^\d+\.|\n\d+\.|\n-|제\d+조|제\d+항', content):
            quality_factors.append(0.2)
        else:
            quality_factors.append(0.1)

        return sum(quality_factors)

    def _ensure_diversity(self, sorted_results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """결과 다양성 확보"""
        if len(sorted_results) <= limit:
            return sorted_results

        final_results = []
        used_documents = set()

        # 첫 번째 패스: 서로 다른 문서에서 선택
        for result in sorted_results:
            if len(final_results) >= limit:
                break

            doc_id = result.get("document_id", "")
            if doc_id not in used_documents:
                final_results.append(result)
                used_documents.add(doc_id)

        # 두 번째 패스: 남은 슬롯을 점수 순으로 채움
        for result in sorted_results:
            if len(final_results) >= limit:
                break

            if result not in final_results:
                final_results.append(result)

        return final_results[:limit]
    
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