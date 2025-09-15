"""
강화된 검색 Agent - 기존 SearchAgent의 RAG 기능 확장
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.agents.specialized.search_agent import SearchAgent
from app.agents.base_agent import Message, Task
from app.rag.retrieval.embedder import DocumentEmbedder
from app.rag.retrieval.search_strategy import SearchStrategy
from app.rag.retrieval.graph_retriever import GraphRetriever
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.core.config import settings

class EnhancedSearchAgent(SearchAgent):
    """
    강화된 검색 Agent
    - 다중 RAG 전략 동시 실행
    - 실시간 검색 결과 품질 평가
    - 동적 검색 파라미터 최적화
    - 지식 그래프 연계 강화
    """
    
    def __init__(self):
        super().__init__()
        
        # 추가 RAG 컴포넌트
        self.graph_retriever_enhanced = GraphRetriever()
        self.neo4j_client = None
        
        # 고급 검색 전략
        self.advanced_strategies = {
            "multi_modal_search": self._multi_modal_search,
            "contextual_expansion": self._contextual_expansion_search,
            "temporal_aware": self._temporal_aware_search,
            "domain_specific": self._domain_specific_search,
            "cross_reference": self._cross_reference_search
        }
        
        # 실시간 품질 메트릭
        self.search_quality_metrics = {
            "relevance_scores": [],
            "diversity_scores": [],
            "coverage_scores": [],
            "freshness_scores": []
        }
        
        # 학습 기반 최적화
        self.learning_data = {
            "successful_queries": [],
            "failed_queries": [],
            "user_feedback": [],
            "performance_patterns": {}
        }
        
        logging.info("EnhancedSearchAgent initialized with advanced RAG capabilities")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """강화된 검색 메시지 처리"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "advanced_search_request":
            result = await self._handle_advanced_search(content)
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="advanced_search_response",
                content=result,
                priority=message.priority
            )
        
        elif message_type == "multi_strategy_search":
            result = await self._execute_multi_strategy_search(content)
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="multi_strategy_response",
                content=result
            )
        
        elif message_type == "search_quality_analysis":
            result = await self._analyze_search_quality(content)
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="quality_analysis_response",
                content=result
            )
        
        # 기본 처리로 위임
        return await super().process_message(message)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """강화된 검색 작업 실행"""
        task_name = task.name
        params = task.parameters
        
        if task_name == "enhanced_intelligent_search":
            return await self._enhanced_intelligent_search(params)
        
        elif task_name == "multi_modal_retrieval":
            return await self._multi_modal_retrieval(params)
        
        elif task_name == "adaptive_reranking":
            return await self._adaptive_reranking(params)
        
        elif task_name == "search_result_synthesis":
            return await self._search_result_synthesis(params)
        
        # 기본 처리로 위임
        return await super().execute_task(task)
    
    async def _handle_advanced_search(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """고급 검색 요청 처리"""
        query = content.get("query", "")
        search_mode = content.get("mode", "comprehensive")
        user_context = content.get("context", {})
        
        start_time = datetime.now()
        
        if search_mode == "comprehensive":
            # 포괄적 검색 - 모든 전략 동시 실행
            result = await self._comprehensive_search(query, user_context)
        
        elif search_mode == "precision":
            # 정밀 검색 - 높은 정확도 우선
            result = await self._precision_search(query, user_context)
        
        elif search_mode == "speed":
            # 빠른 검색 - 속도 우선
            result = await self._speed_optimized_search(query, user_context)
        
        else:
            # 기본 지능형 검색
            result = await self._intelligent_search({"query": query, "context": user_context})
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 검색 품질 평가
        quality_score = await self._evaluate_search_quality(query, result)
        
        result.update({
            "execution_time": execution_time,
            "quality_score": quality_score,
            "search_mode": search_mode
        })
        
        return result
    
    async def _comprehensive_search(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """포괄적 검색 실행"""
        # 모든 검색 전략을 병렬로 실행
        search_tasks = []
        
        # 1. 기본 하이브리드 검색
        search_tasks.append(self._execute_hybrid_search(query))
        
        # 2. 그래프 강화 검색
        search_tasks.append(self._execute_graph_enhanced_search(query))
        
        # 3. 시맨틱 집중 검색
        search_tasks.append(self._execute_semantic_focused_search(query))
        
        # 4. 엔티티 기반 검색
        search_tasks.append(self._execute_entity_based_search(query))
        
        # 병렬 실행
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 결과 통합 및 최적화
        integrated_result = await self._integrate_comprehensive_results(
            query, search_results, user_context
        )
        
        return integrated_result
    
    async def _execute_hybrid_search(self, query: str) -> Dict[str, Any]:
        """하이브리드 검색 실행"""
        try:
            # 시맨틱 검색
            semantic_results = await self.embedder.search_similar_chunks(query, limit=5)
            
            # 키워드 검색 (Neo4j 풀텍스트)
            if not self.neo4j_client:
                self.neo4j_client = Neo4jClient()
            
            with self.neo4j_client as client:
                keyword_results = client.search_related_chunks(query, limit=5)
            
            # 하이브리드 점수 계산
            hybrid_results = await self._calculate_hybrid_scores(
                semantic_results, keyword_results, query
            )
            
            return {
                "success": True,
                "strategy": "hybrid",
                "results": hybrid_results,
                "result_count": len(hybrid_results)
            }
            
        except Exception as e:
            logging.error(f"Hybrid search error: {e}")
            return {"success": False, "error": str(e), "strategy": "hybrid"}
    
    async def _execute_graph_enhanced_search(self, query: str) -> Dict[str, Any]:
        """그래프 강화 검색 실행"""
        try:
            context, sources = await self.graph_retriever_enhanced.retrieve(query, limit=5)
            
            return {
                "success": True,
                "strategy": "graph_enhanced",
                "context": context,
                "results": sources,
                "result_count": len(sources)
            }
            
        except Exception as e:
            logging.error(f"Graph enhanced search error: {e}")
            return {"success": False, "error": str(e), "strategy": "graph_enhanced"}
    
    async def _execute_semantic_focused_search(self, query: str) -> Dict[str, Any]:
        """시맨틱 집중 검색 실행"""
        try:
            # 고품질 시맨틱 검색
            results = await self.embedder.search_similar_chunks(query, limit=8)
            
            # 시맨틱 클러스터링으로 다양성 확보
            clustered_results = await self._semantic_clustering(results, query)
            
            return {
                "success": True,
                "strategy": "semantic_focused",
                "results": clustered_results,
                "result_count": len(clustered_results)
            }
            
        except Exception as e:
            logging.error(f"Semantic focused search error: {e}")
            return {"success": False, "error": str(e), "strategy": "semantic_focused"}
    
    async def _execute_entity_based_search(self, query: str) -> Dict[str, Any]:
        """엔티티 기반 검색 실행"""
        try:
            # 엔티티 추출
            entities = await self._extract_entities_advanced(query)
            
            # 엔티티별 검색 실행
            entity_results = []
            for entity in entities:
                entity_specific_results = await self.embedder.search_similar_chunks(
                    entity, limit=3
                )
                entity_results.extend(entity_specific_results)
            
            # 중복 제거 및 점수 재계산
            deduplicated_results = await self._deduplicate_entity_results(entity_results)
            
            return {
                "success": True,
                "strategy": "entity_based",
                "entities": entities,
                "results": deduplicated_results,
                "result_count": len(deduplicated_results)
            }
            
        except Exception as e:
            logging.error(f"Entity based search error: {e}")
            return {"success": False, "error": str(e), "strategy": "entity_based"}
    
    async def _calculate_hybrid_scores(
        self, semantic_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """하이브리드 점수 계산"""
        # 결과 통합 및 점수 계산
        all_results = {}
        
        # 시맨틱 결과 (가중치 0.7)
        for result in semantic_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                all_results[chunk_id] = result.copy()
                all_results[chunk_id]["semantic_score"] = result.get("relevance", 0.0)
                all_results[chunk_id]["keyword_score"] = 0.0
                all_results[chunk_id]["hybrid_score"] = result.get("relevance", 0.0) * 0.7
        
        # 키워드 결과 (가중치 0.3)
        for result in keyword_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                if chunk_id in all_results:
                    # 기존 결과에 키워드 점수 추가
                    keyword_score = result.get("relevance", 0.0)
                    all_results[chunk_id]["keyword_score"] = keyword_score
                    all_results[chunk_id]["hybrid_score"] += keyword_score * 0.3
                else:
                    # 새로운 결과 추가
                    all_results[chunk_id] = result.copy()
                    all_results[chunk_id]["semantic_score"] = 0.0
                    all_results[chunk_id]["keyword_score"] = result.get("relevance", 0.0)
                    all_results[chunk_id]["hybrid_score"] = result.get("relevance", 0.0) * 0.3
        
        # 하이브리드 점수로 정렬
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x.get("hybrid_score", 0), 
            reverse=True
        )
        
        return sorted_results[:8]  # 상위 8개 반환
    
    async def _semantic_clustering(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """시맨틱 클러스터링으로 다양성 확보"""
        if len(results) <= 3:
            return results
        
        try:
            # 결과들의 임베딩 벡터 수집
            embeddings = []
            for result in results:
                if "embedding" in result:
                    embeddings.append(result["embedding"])
                else:
                    # 임베딩이 없으면 생성
                    content = result.get("content", "")
                    embedding = await self.embedder._generate_embedding(content)
                    embeddings.append(embedding)
                    result["embedding"] = embedding
            
            if not embeddings:
                return results
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(embeddings)
            
            # 다양성 기반 선택 (간단한 구현)
            selected_indices = [0]  # 첫 번째는 항상 선택
            
            for i in range(1, min(len(results), 5)):
                # 기존 선택된 결과들과의 평균 유사도가 낮은 것 선택
                avg_similarities = []
                for j in range(len(results)):
                    if j not in selected_indices:
                        avg_sim = np.mean([similarity_matrix[j][k] for k in selected_indices])
                        avg_similarities.append((j, avg_sim))
                
                if avg_similarities:
                    # 가장 다양한 결과 선택
                    next_idx = min(avg_similarities, key=lambda x: x[1])[0]
                    selected_indices.append(next_idx)
            
            return [results[i] for i in selected_indices]
            
        except Exception as e:
            logging.error(f"Semantic clustering error: {e}")
            return results[:5]  # 오류 시 상위 5개 반환
    
    async def _extract_entities_advanced(self, query: str) -> List[str]:
        """고급 엔티티 추출"""
        # 기본 엔티티 추출 (질문 분석기 활용)
        _, entities = self.question_analyzer.analyze_question(query)
        
        # 추가 엔티티 추출 로직
        enhanced_entities = entities.copy()
        
        # 명사구 추출
        noun_phrases = await self._extract_noun_phrases(query)
        enhanced_entities.extend(noun_phrases)
        
        # 중복 제거
        unique_entities = list(set(enhanced_entities))
        
        return unique_entities[:5]  # 최대 5개 엔티티
    
    async def _extract_noun_phrases(self, query: str) -> List[str]:
        """명사구 추출 (간단한 구현)"""
        words = query.split()
        noun_phrases = []
        
        # 2-3단어 조합으로 명사구 생성
        for i in range(len(words)):
            if i < len(words) - 1:
                noun_phrases.append(f"{words[i]} {words[i+1]}")
            if i < len(words) - 2:
                noun_phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return noun_phrases
    
    async def _deduplicate_entity_results(
        self, entity_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """엔티티 결과 중복 제거"""
        seen_chunks = {}
        deduplicated = []
        
        for result in entity_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = result
                deduplicated.append(result)
            elif chunk_id in seen_chunks:
                # 점수가 더 높으면 교체
                if result.get("relevance", 0) > seen_chunks[chunk_id].get("relevance", 0):
                    seen_chunks[chunk_id] = result
        
        # 점수순 정렬
        sorted_results = sorted(
            seen_chunks.values(),
            key=lambda x: x.get("relevance", 0),
            reverse=True
        )
        
        return sorted_results[:6]
    
    async def _integrate_comprehensive_results(
        self, query: str, search_results: List[Any], user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """포괄적 검색 결과 통합"""
        all_sources = []
        strategy_performance = {}
        
        # 성공한 검색 결과들 수집
        for result in search_results:
            if isinstance(result, dict) and result.get("success"):
                strategy = result.get("strategy", "unknown")
                sources = result.get("results", [])
                
                # 전략별 성능 기록
                strategy_performance[strategy] = {
                    "result_count": len(sources),
                    "avg_relevance": sum(s.get("relevance", 0) for s in sources) / len(sources) if sources else 0
                }
                
                # 소스에 전략 정보 추가
                for source in sources:
                    source["source_strategy"] = strategy
                    source["strategy_weight"] = self._get_strategy_weight(strategy)
                    all_sources.append(source)
        
        # 통합 점수 계산 및 재랭킹
        reranked_sources = await self._rerank_integrated_results(all_sources, query)
        
        # 최종 컨텍스트 구성
        final_context = "\n\n".join([
            source.get("content", "") for source in reranked_sources[:8]
        ])
        
        return {
            "success": True,
            "context": final_context,
            "sources": reranked_sources[:8],
            "strategy_performance": strategy_performance,
            "integration_method": "comprehensive",
            "total_sources_found": len(all_sources)
        }
    
    def _get_strategy_weight(self, strategy: str) -> float:
        """전략별 가중치 반환"""
        weights = {
            "hybrid": 1.0,
            "graph_enhanced": 0.9,
            "semantic_focused": 0.8,
            "entity_based": 0.7
        }
        return weights.get(strategy, 0.5)
    
    async def _rerank_integrated_results(
        self, sources: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """통합 결과 재랭킹"""
        # 중복 제거
        unique_sources = {}
        for source in sources:
            chunk_id = source.get("chunk_id", "")
            content_hash = hash(source.get("content", "")[:100])
            key = chunk_id or f"content_{content_hash}"
            
            if key not in unique_sources:
                unique_sources[key] = source
            else:
                # 더 높은 점수의 결과로 유지
                existing_score = unique_sources[key].get("relevance", 0) * unique_sources[key].get("strategy_weight", 1)
                new_score = source.get("relevance", 0) * source.get("strategy_weight", 1)
                
                if new_score > existing_score:
                    unique_sources[key] = source
        
        # 통합 점수로 정렬
        for source in unique_sources.values():
            base_relevance = source.get("relevance", 0.5)
            strategy_weight = source.get("strategy_weight", 1.0)
            source["integrated_score"] = base_relevance * strategy_weight
        
        sorted_sources = sorted(
            unique_sources.values(),
            key=lambda x: x.get("integrated_score", 0),
            reverse=True
        )
        
        return sorted_sources
    
    async def _evaluate_search_quality(
        self, query: str, search_result: Dict[str, Any]
    ) -> float:
        """검색 품질 평가"""
        quality_factors = []
        
        # 결과 수 적절성
        result_count = len(search_result.get("sources", []))
        if 3 <= result_count <= 8:
            quality_factors.append(0.3)
        elif result_count > 0:
            quality_factors.append(0.1)
        
        # 평균 관련성 점수
        sources = search_result.get("sources", [])
        if sources:
            avg_relevance = sum(s.get("relevance", 0) for s in sources) / len(sources)
            quality_factors.append(avg_relevance * 0.4)
        
        # 다양성 (서로 다른 전략에서 온 결과)
        strategies_used = set(s.get("source_strategy", "") for s in sources)
        diversity_score = min(len(strategies_used) / 3, 1.0)  # 최대 3개 전략
        quality_factors.append(diversity_score * 0.3)
        
        return sum(quality_factors)
    
    def get_enhanced_search_statistics(self) -> Dict[str, Any]:
        """강화된 검색 통계 조회"""
        base_stats = self.get_search_statistics()
        
        enhanced_stats = {
            **base_stats,
            "search_quality_metrics": self.search_quality_metrics,
            "advanced_strategies_available": list(self.advanced_strategies.keys()),
            "learning_data_size": {
                "successful_queries": len(self.learning_data["successful_queries"]),
                "failed_queries": len(self.learning_data["failed_queries"]),
                "user_feedback": len(self.learning_data["user_feedback"])
            }
        }
        
        return enhanced_stats