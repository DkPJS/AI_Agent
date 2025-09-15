"""Advanced Search Agent Module.

This module provides an intelligent search agent with advanced search strategies,
learning capabilities, and performance optimization. The SearchAgent can:

- Execute intelligent search strategies based on question types
- Learn from search history to optimize performance
- Cache results for improved response times
- Integrate multiple search approaches (semantic, hybrid, entity-based)
- Provide detailed search analytics and performance metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

from app.agents.base_agent import BaseAgent, Message, Task, AgentState
from app.rag.retrieval.embedder import DocumentEmbedder
from app.rag.retrieval.search_strategy import SearchStrategy
from app.rag.retrieval.graph_retriever import GraphRetriever
from app.rag.retrieval.question_analyzer import QuestionAnalyzer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Type aliases for better readability
SearchRecord = Dict[str, Any]
StrategyMetrics = Dict[str, Dict[str, Union[float, int]]]
OptimizationParams = Dict[str, Union[float, int, bool]]
CacheData = Dict[str, Dict[str, Any]]


class SearchAgent(BaseAgent):
    """Advanced AI Search Agent.

    An intelligent search agent that provides:
    - Dynamic search strategy selection based on query analysis
    - Performance learning and optimization
    - Multi-strategy search fusion
    - Comprehensive caching system
    - Detailed analytics and metrics

    Attributes:
        embedder: Document embedding and vector search component
        search_strategy: Traditional search strategy handler
        graph_retriever: Advanced graph-based search component
        question_analyzer: Query analysis and classification
        search_history: Historical search performance data
        optimization_params: Dynamic search parameter configuration
        strategy_performance: Performance metrics per search strategy
        entity_cache: Cache for entity extraction results
        search_cache: Cache for search results
        cache_ttl: Time-to-live for cache entries
    """

    def __init__(self) -> None:
        """Initialize the SearchAgent with all necessary components."""
        super().__init__(
            name="SearchAgent",
            capabilities=[
                "semantic_search",
                "hybrid_search",
                "entity_search",
                "graph_search",
                "search_optimization",
                "query_understanding",
                "result_ranking"
            ]
        )

        # Initialize search components
        self._initialize_search_components()

        # Initialize performance tracking
        self._initialize_performance_tracking()

        # Initialize caching system
        self._initialize_cache_system()

        logger.info("SearchAgent initialized with advanced search capabilities")

    def _initialize_search_components(self) -> None:
        """Initialize all search-related components."""
        self.embedder = DocumentEmbedder()
        self.search_strategy = SearchStrategy()
        self.graph_retriever = GraphRetriever()
        self.question_analyzer = QuestionAnalyzer()

    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking and optimization parameters."""
        self.search_history: List[SearchRecord] = []
        self.optimization_params: OptimizationParams = {
            "semantic_weight": getattr(settings, 'SEMANTIC_WEIGHT', 0.7),
            "keyword_weight": getattr(settings, 'KEYWORD_WEIGHT', 0.3),
            "min_similarity_threshold": 0.1,
            "max_results_per_strategy": 10,
            "adaptive_limit_enabled": True
        }

        self.strategy_performance: StrategyMetrics = {
            "semantic": {"avg_relevance": 0.7, "usage_count": 0},
            "hybrid": {"avg_relevance": 0.8, "usage_count": 0},
            "entity": {"avg_relevance": 0.6, "usage_count": 0},
            "graph": {"avg_relevance": 0.75, "usage_count": 0}
        }

    def _initialize_cache_system(self) -> None:
        """Initialize caching system for performance optimization."""
        self.entity_cache: CacheData = {}
        self.search_cache: CacheData = {}
        self.cache_ttl = timedelta(minutes=10)
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process search-related messages with task-based handling.

        Args:
            message: The incoming message to process

        Returns:
            Optional response message based on the request type

        Raises:
            Exception: If message processing fails
        """
        try:
            message_type = message.message_type
            content = message.content

            if message_type == "search_request":
                return await self._handle_search_message(message, content)
            elif message_type == "optimize_search":
                return await self._handle_optimization_message(message)
            elif message_type == "search_feedback":
                await self._process_search_feedback(content)
                return None
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return None

        except Exception as e:
            logger.error(f"SearchAgent message processing error: {e}")
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )

    async def _handle_search_message(self, message: Message, content: Dict[str, Any]) -> Message:
        """Handle search request messages.

        Args:
            message: The original search message
            content: The message content with search parameters

        Returns:
            Response message with search results
        """
        # Create and queue search task
        task = Task(
            name=f"search_query_{message.id[:8]}",
            description="Execute search query",
            parameters={
                "message": message,
                "query": content.get("query", ""),
                "limit": content.get("limit", 5),
                "type": content.get("type", "auto")
            },
            priority=message.priority
        )

        await self.add_task(task)

        # Execute search and get results
        result = await self._handle_search_request(content)

        # Update performance metrics
        self.performance_metrics["tasks_completed"] += 1
        self._update_performance_metrics()

        return Message(
            sender=self.name,
            receiver=message.sender,
            message_type="search_response",
            content=result,
            priority=message.priority
        )

    async def _handle_optimization_message(self, message: Message) -> Message:
        """Handle search optimization requests.

        Args:
            message: The optimization request message

        Returns:
            Response message with optimization results
        """
        await self._optimize_search_parameters()

        return Message(
            sender=self.name,
            receiver=message.sender,
            message_type="optimization_complete",
            content={"status": "optimized", "params": self.optimization_params}
        )
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """검색 작업 실행"""
        task_name = task.name
        params = task.parameters
        
        try:
            if task_name == "intelligent_search":
                return await self._intelligent_search(params)
            
            elif task_name == "batch_search":
                return await self._batch_search(params)
            
            elif task_name == "search_analysis":
                return await self._analyze_search_patterns()
            
            elif task_name == "index_optimization":
                return await self._optimize_search_index()
            
            else:
                raise ValueError(f"Unknown task: {task_name}")
                
        except Exception as e:
            logging.error(f"SearchAgent task execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """검색 관련 의사결정"""
        # 현재 상황 분석
        pending_searches = context.get("pending_tasks", 0)
        recent_performance = self.performance_metrics.get("success_rate", 0.8)
        
        # 의사결정 로직
        if pending_searches > 0:
            return {
                "action": "process_next_task",
                "reason": f"Processing {pending_searches} pending search tasks"
            }
        
        # 성능이 저하되면 최적화 수행
        if recent_performance < 0.7 and len(self.search_history) > 20:
            return {
                "action": "optimize_search",
                "reason": "Performance degradation detected, initiating optimization"
            }
        
        # 주기적으로 검색 패턴 분석
        if len(self.search_history) % 50 == 0 and len(self.search_history) > 0:
            return {
                "action": "analyze_patterns",
                "reason": "Regular pattern analysis"
            }
        
        return {"action": "monitor", "reason": "Monitoring search performance"}
    
    async def _handle_search_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """지능형 검색 요청 처리 (GraphRetriever 통합)"""
        query = content.get("query", "")
        limit = content.get("limit", settings.DEFAULT_SEARCH_LIMIT)
        search_type = content.get("type", "auto")
        
        # 쿼리 분석
        question_type, focus_entities = self.question_analyzer.analyze_question(query)
        
        # 검색 전략 선택 (학습된 패턴 기반)
        optimal_strategy = await self._select_optimal_strategy(query, question_type)
        
        # 검색 실행
        start_time = datetime.now()
        
        # GraphRetriever의 고급 기능을 우선 사용
        if search_type == "advanced" or optimal_strategy == "advanced":
            # 고급 그래프 검색 (도메인 최적화, 동의어 확장 포함)
            context, sources = await self.graph_retriever.retrieve(query, limit=limit)
            search_method = "GraphRetriever (Advanced)"
        elif search_type == "auto" or search_type == optimal_strategy:
            # 기존 검색 전략과 그래프 검색 결과 융합
            context, sources = await self._hybrid_advanced_search(
                query, question_type, focus_entities, limit
            )
            search_method = "Hybrid (SearchStrategy + GraphRetriever)"
        else:
            # 특정 전략 강제 사용
            context, sources = await self._execute_specific_strategy(
                search_type, query, question_type, focus_entities, limit
            )
            search_method = f"Specific Strategy ({search_type})"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 검색 기록 저장
        search_record = {
            "query": query,
            "question_type": question_type,
            "strategy": optimal_strategy,
            "search_method": search_method,
            "execution_time": execution_time,
            "results_count": len(sources),
            "timestamp": datetime.now()
        }
        self.search_history.append(search_record)
        
        # 전략 성능 업데이트
        self.strategy_performance[optimal_strategy]["usage_count"] += 1
        
        return {
            "success": True,
            "context": context,
            "sources": sources,
            "metadata": {
                "strategy_used": optimal_strategy,
                "execution_time": execution_time,
                "question_type": question_type,
                "entities": focus_entities
            }
        }
    
    async def _intelligent_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """지능형 검색 (학습 기반 최적화)"""
        query = params.get("query", "")
        user_context = params.get("context", {})
        
        # 사용자 컨텍스트 기반 검색 개인화
        personalized_params = await self._personalize_search(query, user_context)
        
        # 다중 전략 검색
        strategies = ["hybrid", "semantic", "entity"]
        results = {}
        
        for strategy in strategies:
            try:
                result = await self._execute_strategy(strategy, query, personalized_params)
                results[strategy] = result
            except Exception as e:
                logging.error(f"Strategy {strategy} failed: {e}")
                results[strategy] = {"success": False, "error": str(e)}
        
        # 결과 융합 및 재랭킹
        final_result = await self._fuse_search_results(results, query)
        
        return final_result
    
    async def _select_optimal_strategy(self, query: str, question_type: str) -> str:
        """최적 검색 전략 선택 (학습 기반)"""
        # 유사한 과거 쿼리의 성능 분석
        similar_searches = [
            record for record in self.search_history
            if record["question_type"] == question_type and 
            self._calculate_query_similarity(query, record["query"]) > 0.7
        ]
        
        if similar_searches:
            # 과거 성능이 가장 좋은 전략 선택
            strategy_scores = {}
            for search in similar_searches:
                strategy = search["strategy"]
                # 성능 점수 = 1 / (실행시간 + 1) * (결과수 / 10)
                score = (1.0 / (search["execution_time"] + 1)) * min(search["results_count"] / 10, 1.0)
                strategy_scores[strategy] = strategy_scores.get(strategy, []) + [score]
            
            # 평균 점수가 가장 높은 전략 선택
            best_strategy = max(
                strategy_scores.keys(),
                key=lambda s: sum(strategy_scores[s]) / len(strategy_scores[s])
            )
            return best_strategy
        
        # 기본 전략 선택 (질문 유형별)
        default_strategies = {
            "factual": "hybrid",
            "comparison": "entity",
            "summary": "semantic",
            "procedural": "hybrid",
            "general": "hybrid"
        }
        
        return default_strategies.get(question_type, "hybrid")
    
    async def _execute_specific_strategy(self, strategy: str, query: str, 
                                       question_type: str, entities: List[str], 
                                       limit: int) -> tuple:
        """특정 검색 전략 실행"""
        if strategy == "semantic":
            results = await self.embedder.search_similar(query, limit)
            return self._format_search_results(results)
        
        elif strategy == "hybrid":
            results = await self.embedder.hybrid_search(query, limit)
            return self._format_search_results(results)
        
        elif strategy == "entity":
            # 엔티티 기반 검색은 search_strategy에서 처리
            return await self.search_strategy.execute_search(query, question_type, entities)
        
        else:
            # 기본값으로 하이브리드 검색
            results = await self.embedder.hybrid_search(query, limit)
            return self._format_search_results(results)
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> tuple:
        """검색 결과 포맷팅"""
        if not results:
            return "", []
        
        context = "\n\n".join([result.get("content", "") for result in results])
        return context, results
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """쿼리 유사도 계산 (간단한 구현)"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _optimize_search_parameters(self):
        """검색 파라미터 동적 최적화"""
        if len(self.search_history) < 10:
            return
        
        # 최근 검색 성능 분석
        recent_searches = self.search_history[-20:]
        
        # 실행 시간과 결과 품질 기반 최적화
        avg_execution_time = sum(s["execution_time"] for s in recent_searches) / len(recent_searches)
        avg_results_count = sum(s["results_count"] for s in recent_searches) / len(recent_searches)
        
        # 파라미터 조정
        if avg_execution_time > 2.0:  # 실행 시간이 너무 길면
            self.optimization_params["max_results_per_strategy"] = max(5, 
                self.optimization_params["max_results_per_strategy"] - 1)
        
        if avg_results_count < 3:  # 결과가 너무 적으면
            self.optimization_params["min_similarity_threshold"] = max(0.05,
                self.optimization_params["min_similarity_threshold"] - 0.01)
        
        logging.info(f"SearchAgent parameters optimized: {self.optimization_params}")
    
    async def _personalize_search(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 컨텍스트 기반 검색 개인화"""
        personalized_params = {}
        
        # 사용자 선호도 반영
        user_preferences = user_context.get("preferences", {})
        if "detail_level" in user_preferences:
            if user_preferences["detail_level"] == "high":
                personalized_params["limit"] = min(15, settings.DEFAULT_SEARCH_LIMIT * 2)
            elif user_preferences["detail_level"] == "low":
                personalized_params["limit"] = max(3, settings.DEFAULT_SEARCH_LIMIT // 2)
        
        # 도메인 컨텍스트 반영
        domain = user_context.get("domain", "general")
        if domain in ["technical", "research"]:
            personalized_params["semantic_weight"] = 0.8
            personalized_params["keyword_weight"] = 0.2
        elif domain in ["general", "overview"]:
            personalized_params["semantic_weight"] = 0.6
            personalized_params["keyword_weight"] = 0.4
        
        return personalized_params
    
    async def _fuse_search_results(self, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """다중 전략 검색 결과 융합"""
        successful_results = {k: v for k, v in results.items() if v.get("success")}
        
        if not successful_results:
            return {"success": False, "error": "All search strategies failed"}
        
        # 결과 점수 계산 및 융합
        all_sources = []
        for strategy, result in successful_results.items():
            sources = result.get("sources", [])
            for source in sources:
                source["strategy"] = strategy
                source["strategy_confidence"] = self.strategy_performance[strategy]["avg_relevance"]
                all_sources.append(source)
        
        # 중복 제거 및 재랭킹
        unique_sources = self._deduplicate_and_rerank(all_sources, query)
        
        # 최종 컨텍스트 구성
        final_context = "\n\n".join([source.get("content", "") for source in unique_sources])
        
        return {
            "success": True,
            "context": final_context,
            "sources": unique_sources,
            "strategies_used": list(successful_results.keys()),
            "fusion_score": len(unique_sources) / len(all_sources) if all_sources else 0
        }
    
    def _deduplicate_and_rerank(self, sources: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """중복 제거 및 재랭킹"""
        seen_content = set()
        unique_sources = []
        
        for source in sources:
            content = source.get("content", "")
            content_hash = hash(content[:100])  # 처음 100자로 중복 판단
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                # 종합 점수 계산
                relevance_score = source.get("relevance", 0.5)
                strategy_confidence = source.get("strategy_confidence", 0.7)
                combined_score = (relevance_score * 0.7) + (strategy_confidence * 0.3)
                source["final_score"] = combined_score
                unique_sources.append(source)
        
        # 점수 기준 정렬
        unique_sources.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return unique_sources[:10]  # 상위 10개만 반환
    
    async def _analyze_search_patterns(self) -> Dict[str, Any]:
        """검색 패턴 분석 및 인사이트 도출"""
        if len(self.search_history) < 10:
            return {"message": "Insufficient data for pattern analysis"}
        
        # 질문 유형별 분석
        question_types = {}
        for search in self.search_history:
            qtype = search["question_type"]
            question_types[qtype] = question_types.get(qtype, []) + [search]
        
        # 성능 분석
        analysis = {
            "total_searches": len(self.search_history),
            "avg_execution_time": sum(s["execution_time"] for s in self.search_history) / len(self.search_history),
            "question_type_distribution": {k: len(v) for k, v in question_types.items()},
            "strategy_performance": self.strategy_performance,
            "recent_trend": "improving" if len(self.search_history) > 20 and 
                           sum(s["execution_time"] for s in self.search_history[-10:]) < 
                           sum(s["execution_time"] for s in self.search_history[-20:-10]) else "stable"
        }
        
        logging.info(f"SearchAgent pattern analysis completed: {analysis}")
        return analysis
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        if not self.search_history:
            return {"message": "No search history available"}
        
        recent_searches = self.search_history[-10:]
        
        return {
            "total_searches": len(self.search_history),
            "recent_avg_time": sum(s["execution_time"] for s in recent_searches) / len(recent_searches),
            "recent_avg_results": sum(s["results_count"] for s in recent_searches) / len(recent_searches),
            "recent_avg_confidence": sum(s.get("confidence", 0.5) for s in recent_searches) / len(recent_searches),
            "optimization_params": self.optimization_params,
            "strategy_performance": self.strategy_performance,
            "agent_status": self.get_status()
        }
    
    def _get_cache_key(self, query: str, strategy: str = None) -> str:
        """캐시 키 생성"""
        if strategy:
            return f"{strategy}:{hash(query.lower().strip())}"
        return str(hash(query.lower().strip()))
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """캐시 유효성 검사"""
        return datetime.now() - timestamp < self.cache_ttl
    
    def _get_cached_entities(self, query: str) -> Optional[Dict[str, Any]]:
        """캐시된 엔티티 결과 조회"""
        cache_key = self._get_cache_key(query, "entities")
        if cache_key in self.entity_cache:
            cached = self.entity_cache[cache_key]
            if self._is_cache_valid(cached["timestamp"]):
                logging.debug(f"Using cached entities for query: {query[:50]}...")
                return cached["data"]
            else:
                del self.entity_cache[cache_key]
        return None
    
    def _cache_entities(self, query: str, entities: Dict[str, Any]):
        """엔티티 결과 캐시 저장"""
        cache_key = self._get_cache_key(query, "entities")
        self.entity_cache[cache_key] = {
            "data": entities,
            "timestamp": datetime.now()
        }
        
        # 캐시 크기 제한 (최대 100개)
        if len(self.entity_cache) > 100:
            oldest_key = min(self.entity_cache.keys(), 
                           key=lambda k: self.entity_cache[k]["timestamp"])
            del self.entity_cache[oldest_key]
    
    def _get_cached_search(self, query: str, strategy: str) -> Optional[Dict[str, Any]]:
        """캐시된 검색 결과 조회"""
        cache_key = self._get_cache_key(query, strategy)
        if cache_key in self.search_cache:
            cached = self.search_cache[cache_key]
            if self._is_cache_valid(cached["timestamp"]):
                logging.debug(f"Using cached search results for {strategy}: {query[:50]}...")
                return cached["data"]
            else:
                del self.search_cache[cache_key]
        return None
    
    def _cache_search_results(self, query: str, strategy: str, results: Dict[str, Any]):
        """검색 결과 캐시 저장"""
        cache_key = self._get_cache_key(query, strategy)
        self.search_cache[cache_key] = {
            "data": results,
            "timestamp": datetime.now()
        }
        
        # 캐시 크기 제한 (최대 200개)
        if len(self.search_cache) > 200:
            oldest_key = min(self.search_cache.keys(),
                           key=lambda k: self.search_cache[k]["timestamp"])
            del self.search_cache[oldest_key]