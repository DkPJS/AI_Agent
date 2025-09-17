"""
RAG 전용 코디네이터 Agent
각 RAG 컴포넌트를 전문적으로 관리하고 최적화하는 Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.agents.base_agent import BaseAgent, Message, Task, AgentState
from app.rag.generation.qa_system import AdvancedQASystem
from app.rag.retrieval.search_strategy import SearchStrategy
from app.rag.retrieval.graph_retriever import GraphRetriever
from app.rag.retrieval.question_analyzer import QuestionAnalyzer
from app.rag.retrieval.embedder import DocumentEmbedder
from app.core.config import settings

@dataclass
class RAGPerformanceMetrics:
    """RAG 성능 메트릭"""
    retrieval_accuracy: float = 0.0
    generation_quality: float = 0.0
    end_to_end_latency: float = 0.0
    context_relevance: float = 0.0
    answer_consistency: float = 0.0
    user_satisfaction: float = 0.0

class RAGCoordinator(BaseAgent):
    """
    RAG 전용 코디네이터 Agent
    - RAG 파이프라인 전체 최적화
    - 검색-생성 균형 조정
    - 성능 모니터링 및 자동 튜닝
    - 지식 그래프와 벡터 DB 조화
    """
    
    def __init__(self):
        super().__init__(
            name="RAGCoordinator",
            capabilities=[
                "rag_orchestration",
                "retrieval_optimization", 
                "generation_tuning",
                "context_management",
                "knowledge_synthesis",
                "performance_monitoring"
            ]
        )
        
        # RAG 컴포넌트 초기화
        self.qa_system = AdvancedQASystem()
        self.search_strategy = SearchStrategy()
        self.graph_retriever = GraphRetriever()
        self.question_analyzer = QuestionAnalyzer()
        self.embedder = DocumentEmbedder()
        
        # RAG 성능 추적
        self.rag_metrics = RAGPerformanceMetrics()
        self.rag_history = []
        
        # 적응형 파라미터
        self.adaptive_params = {
            "retrieval_threshold": 0.7,
            "context_window_size": 4,
            "reranking_enabled": True,
            "hybrid_weight_ratio": 0.7,  # semantic vs keyword
            "generation_temperature": 0.7,
            "max_context_chunks": 5
        }
        
        # 성능 최적화 상태
        self.optimization_state = {
            "last_optimization": None,
            "optimization_cycles": 0,
            "performance_trend": "stable",
            "bottleneck_component": None
        }
        
        # 실시간 모니터링 데이터
        self.real_time_stats = {
            "current_load": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
        
        logging.info("RAGCoordinator initialized with advanced optimization capabilities")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """RAG 관련 메시지 처리"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "optimize_rag_pipeline":
                result = await self._optimize_rag_pipeline(content)
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="optimization_complete",
                    content=result
                )
            
            elif message_type == "intelligent_rag_query":
                result = await self._process_intelligent_rag_query(content)
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="rag_response",
                    content=result
                )
            
            elif message_type == "rag_performance_analysis":
                result = await self._analyze_rag_performance()
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="performance_report",
                    content=result
                )
            
            elif message_type == "tune_rag_components":
                result = await self._tune_rag_components(content)
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="tuning_complete",
                    content=result
                )
            
            return None
            
        except Exception as e:
            logging.error(f"RAGCoordinator message processing error: {e}")
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """RAG 최적화 작업 실행"""
        task_name = task.name
        params = task.parameters
        
        try:
            if task_name == "adaptive_rag_processing":
                return await self._adaptive_rag_processing(params)
            
            elif task_name == "rag_component_benchmarking":
                return await self._benchmark_rag_components()
            
            elif task_name == "context_optimization":
                return await self._optimize_context_selection(params)
            
            elif task_name == "retrieval_generation_balancing":
                return await self._balance_retrieval_generation()
            
            elif task_name == "rag_pipeline_monitoring":
                return await self._monitor_rag_pipeline()
            
            else:
                raise ValueError(f"Unknown RAG task: {task_name}")
                
        except Exception as e:
            logging.error(f"RAGCoordinator task execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 최적화 의사결정"""
        # 현재 RAG 성능 분석
        current_performance = await self._assess_current_rag_performance()
        
        # 성능 임계값 기반 의사결정
        if current_performance["retrieval_accuracy"] < 0.8:
            return {
                "action": "optimize_retrieval",
                "reason": "Retrieval accuracy below threshold",
                "priority": "high"
            }
        
        if current_performance["generation_quality"] < 0.75:
            return {
                "action": "tune_generation",
                "reason": "Generation quality degradation detected",
                "priority": "medium"
            }
        
        if current_performance["end_to_end_latency"] > 5.0:
            return {
                "action": "optimize_performance",
                "reason": "Response time exceeds acceptable limits",
                "priority": "high"
            }
        
        # 주기적 최적화
        if self._should_perform_routine_optimization():
            return {
                "action": "routine_optimization",
                "reason": "Scheduled performance optimization",
                "priority": "low"
            }
        
        return {"action": "monitor", "reason": "All systems operating normally"}
    
    async def _process_intelligent_rag_query(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """지능형 RAG 쿼리 처리"""
        query = content.get("query", "")
        user_context = content.get("context", {})
        
        start_time = datetime.now()
        
        # 1. 질문 분석 및 전략 선택
        question_type, entities = self.question_analyzer.analyze_question(query)
        retrieval_strategy = await self._select_optimal_retrieval_strategy(
            query, question_type, entities
        )
        
        # 2. 적응형 검색 실행
        retrieval_start = datetime.now()
        context_data, sources = await self._execute_adaptive_retrieval(
            query, retrieval_strategy, entities
        )
        retrieval_time = (datetime.now() - retrieval_start).total_seconds()
        
        # 3. 컨텍스트 최적화
        optimized_context = await self._optimize_context_for_generation(
            context_data, query, question_type
        )
        
        # 4. 적응형 답변 생성
        generation_start = datetime.now()
        answer_result = await self._generate_adaptive_answer(
            query, optimized_context, question_type, sources
        )
        generation_time = (datetime.now() - generation_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 5. 성능 메트릭 업데이트
        await self._update_rag_metrics(
            query, answer_result, retrieval_time, generation_time, total_time
        )
        
        return {
            "success": True,
            "answer": answer_result.get("answer", ""),
            "sources": sources,
            "confidence": answer_result.get("confidence", 0.0),
            "metadata": {
                "question_type": question_type,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "context_chunks": len(sources),
                "optimization_applied": True
            }
        }
    
    async def _select_optimal_retrieval_strategy(
        self, query: str, question_type: str, entities: List[str]
    ) -> str:
        """최적 검색 전략 선택"""
        # 과거 성능 데이터 기반 전략 선택
        strategy_performance = {}
        
        for record in self.rag_history[-50:]:  # 최근 50개 기록 분석
            if record.get("question_type") == question_type:
                strategy = record.get("retrieval_strategy", "hybrid")
                performance = record.get("retrieval_accuracy", 0.5)
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(performance)
        
        # 평균 성능이 가장 높은 전략 선택
        if strategy_performance:
            best_strategy = max(
                strategy_performance.keys(),
                key=lambda s: sum(strategy_performance[s]) / len(strategy_performance[s])
            )
            return best_strategy
        
        # 기본 전략 매핑
        strategy_mapping = {
            "factual": "hybrid",
            "comparison": "graph_enhanced",
            "summary": "semantic_focused",
            "procedural": "entity_based",
            "causal": "graph_reasoning"
        }
        
        return strategy_mapping.get(question_type, "hybrid")
    
    async def _execute_adaptive_retrieval(
        self, query: str, strategy: str, entities: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """적응형 검색 실행"""
        if strategy == "hybrid":
            # 하이브리드 검색 (벡터 + 키워드)
            semantic_results = await self.embedder.search_similar_chunks(
                query, limit=self.adaptive_params["max_context_chunks"]
            )
            keyword_results = await self.search_strategy.execute_search(
                query, "factual", entities
            )
            
            # 결과 융합
            context, sources = await self._fuse_search_results(
                semantic_results, keyword_results[1] if keyword_results else []
            )
            
        elif strategy == "graph_enhanced":
            # 그래프 강화 검색
            context, sources = await self.graph_retriever.retrieve(
                query, limit=self.adaptive_params["max_context_chunks"]
            )
            
        elif strategy == "semantic_focused":
            # 시맨틱 중심 검색
            sources = await self.embedder.search_similar_chunks(
                query, limit=self.adaptive_params["max_context_chunks"]
            )
            context = "\n\n".join([source.get("content", "") for source in sources])
            
        else:
            # 기본 하이브리드
            context, sources = await self.search_strategy.execute_search(
                query, "general", entities
            )
        
        return context, sources
    
    async def _optimize_context_for_generation(
        self, context: str, query: str, question_type: str
    ) -> str:
        """생성을 위한 컨텍스트 최적화"""
        # 컨텍스트 길이 최적화
        context_chunks = context.split("\n\n")
        
        # 질문 유형별 컨텍스트 우선순위 조정
        if question_type == "factual":
            # 사실 기반 질문: 정확한 정보가 포함된 청크 우선
            optimized_chunks = context_chunks[:3]
        elif question_type == "summary":
            # 요약 질문: 더 많은 컨텍스트 포함
            optimized_chunks = context_chunks[:6]
        else:
            # 기본: 적당한 양의 컨텍스트
            optimized_chunks = context_chunks[:4]
        
        return "\n\n".join(optimized_chunks)
    
    async def _generate_adaptive_answer(
        self, query: str, context: str, question_type: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """적응형 답변 생성"""
        # QA 시스템을 통한 기본 답변 생성
        result = await self.qa_system.answer_question(query)
        
        # 답변 품질 평가
        confidence = await self._assess_answer_quality(
            query, result.get("answer", ""), context, question_type
        )
        
        # 낮은 품질의 경우 재생성 시도 (임계값을 0.5로 낮춤)
        if confidence < 0.5:
            logging.info(f"Low quality answer detected (confidence: {confidence:.3f}), regenerating...")
            # 다른 파라미터로 재생성 (실제 구현에서는 LLM 파라미터 조정)
            result = await self.qa_system.answer_question(query)
            confidence = await self._assess_answer_quality(
                query, result.get("answer", ""), context, question_type
            )
            logging.info(f"Regenerated answer quality: {confidence:.3f}")
        
        return {
            "answer": result.get("answer", ""),
            "confidence": confidence,
            "sources": sources,
            "generation_metadata": {
                "regeneration_attempts": 1 if confidence < 0.5 else 0,
                "question_type": question_type
            }
        }
    
    async def _assess_answer_quality(
        self, question: str, answer: str, context: str, question_type: str
    ) -> float:
        """개선된 답변 품질 평가"""
        if not answer or answer.strip() == "":
            return 0.1

        quality_factors = []

        # 1. 기본 답변 완성도 (0.3)
        if len(answer.strip()) < 10:
            quality_factors.append(0.1)  # 너무 짧음
        elif "오류가 발생했습니다" in answer or "정보를 찾을 수 없습니다" in answer:
            quality_factors.append(0.15)  # 오류 메시지
        elif answer.count('.') >= 2 and len(answer.split()) >= 15:
            quality_factors.append(0.3)  # 완전한 문장들
        else:
            quality_factors.append(0.2)  # 적당한 답변

        # 2. 질문 관련성 (0.25)
        import re
        question_keywords = set(re.findall(r'[가-힣a-zA-Z]+', question.lower()))
        answer_keywords = set(re.findall(r'[가-힣a-zA-Z]+', answer.lower()))

        if question_keywords:
            relevance = len(question_keywords.intersection(answer_keywords)) / len(question_keywords)
            quality_factors.append(relevance * 0.25)
        else:
            quality_factors.append(0.15)

        # 3. 컨텍스트 기반 정보 활용도 (0.25)
        if context and len(context.strip()) > 0:
            context_keywords = set(re.findall(r'[가-힣a-zA-Z]+', context.lower()))
            if context_keywords:
                context_usage = len(context_keywords.intersection(answer_keywords)) / len(context_keywords)
                quality_factors.append(min(context_usage * 1.5, 0.25))  # 최대 0.25
            else:
                quality_factors.append(0.15)
        else:
            quality_factors.append(0.1)  # 컨텍스트 없음

        # 4. 한국어 자연스러움 (0.2)
        korean_patterns = [
            r'[가-힣]',  # 한글 포함
            r'입니다|습니다|됩니다',  # 정중한 표현
            r'[.!?]',  # 적절한 문장부호
        ]
        naturalness_score = 0
        for pattern in korean_patterns:
            if re.search(pattern, answer):
                naturalness_score += 1
        quality_factors.append((naturalness_score / len(korean_patterns)) * 0.2)

        total_score = sum(quality_factors)

        # 최소/최대 값 제한
        return max(0.1, min(1.0, total_score))
    
    async def _fuse_search_results(
        self, semantic_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """검색 결과 융합"""
        all_results = []
        seen_content = set()
        
        # 시맨틱 결과 추가 (가중치 0.7)
        for result in semantic_results:
            content = result.get("content", "")
            content_hash = hash(content[:100])
            if content_hash not in seen_content:
                result["fusion_score"] = result.get("relevance", 0.5) * 0.7
                all_results.append(result)
                seen_content.add(content_hash)
        
        # 키워드 결과 추가 (가중치 0.3)
        for result in keyword_results:
            content = result.get("content", "")
            content_hash = hash(content[:100])
            if content_hash not in seen_content:
                result["fusion_score"] = result.get("relevance", 0.5) * 0.3
                all_results.append(result)
                seen_content.add(content_hash)
        
        # 융합 점수로 정렬
        all_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        
        # 상위 결과 선택
        top_results = all_results[:self.adaptive_params["max_context_chunks"]]
        context = "\n\n".join([result.get("content", "") for result in top_results])
        
        return context, top_results
    
    async def _update_rag_metrics(
        self, query: str, answer_result: Dict[str, Any], 
        retrieval_time: float, generation_time: float, total_time: float
    ):
        """RAG 메트릭 업데이트"""
        # 현재 쿼리 기록
        record = {
            "query": query,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "confidence": answer_result.get("confidence", 0.0),
            "timestamp": datetime.now()
        }
        
        self.rag_history.append(record)
        
        # 최근 기록 유지 (메모리 관리)
        if len(self.rag_history) > 1000:
            self.rag_history = self.rag_history[-500:]
        
        # 누적 메트릭 업데이트
        recent_records = self.rag_history[-20:]  # 최근 20개 기록
        
        self.rag_metrics.retrieval_accuracy = sum(
            r.get("confidence", 0.0) for r in recent_records
        ) / len(recent_records)
        
        self.rag_metrics.end_to_end_latency = sum(
            r.get("total_time", 0.0) for r in recent_records
        ) / len(recent_records)
        
        # 실시간 통계 업데이트
        self.real_time_stats["avg_retrieval_time"] = sum(
            r.get("retrieval_time", 0.0) for r in recent_records
        ) / len(recent_records)
        
        self.real_time_stats["avg_generation_time"] = sum(
            r.get("generation_time", 0.0) for r in recent_records
        ) / len(recent_records)
    
    async def _optimize_rag_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 파이프라인 최적화"""
        optimization_results = {
            "optimizations_applied": [],
            "performance_before": dict(self.rag_metrics.__dict__),
            "timestamp": datetime.now()
        }
        
        # 검색 최적화
        if self.rag_metrics.retrieval_accuracy < 0.8:
            await self._optimize_retrieval_parameters()
            optimization_results["optimizations_applied"].append("retrieval_optimization")
        
        # 생성 최적화
        if self.rag_metrics.generation_quality < 0.75:
            await self._optimize_generation_parameters()
            optimization_results["optimizations_applied"].append("generation_optimization")
        
        # 성능 최적화
        if self.rag_metrics.end_to_end_latency > 5.0:
            await self._optimize_performance_parameters()
            optimization_results["optimizations_applied"].append("performance_optimization")
        
        self.optimization_state["last_optimization"] = datetime.now()
        self.optimization_state["optimization_cycles"] += 1
        
        return optimization_results
    
    async def _optimize_retrieval_parameters(self):
        """검색 파라미터 최적화"""
        # 임계값 조정
        if self.rag_metrics.retrieval_accuracy < 0.7:
            self.adaptive_params["retrieval_threshold"] = max(0.5, 
                self.adaptive_params["retrieval_threshold"] - 0.05)
        
        # 컨텍스트 윈도우 조정
        if self.rag_metrics.context_relevance < 0.8:
            self.adaptive_params["context_window_size"] = min(8, 
                self.adaptive_params["context_window_size"] + 1)
        
        logging.info("Retrieval parameters optimized")
    
    async def _optimize_generation_parameters(self):
        """생성 파라미터 최적화"""
        # 온도 조정
        if self.rag_metrics.generation_quality < 0.7:
            self.adaptive_params["generation_temperature"] = max(0.3,
                self.adaptive_params["generation_temperature"] - 0.1)
        
        logging.info("Generation parameters optimized")
    
    async def _optimize_performance_parameters(self):
        """성능 파라미터 최적화"""
        # 최대 청크 수 조정
        if self.rag_metrics.end_to_end_latency > 5.0:
            self.adaptive_params["max_context_chunks"] = max(3,
                self.adaptive_params["max_context_chunks"] - 1)
        
        logging.info("Performance parameters optimized")
    
    async def _assess_current_rag_performance(self) -> Dict[str, float]:
        """현재 RAG 성능 평가"""
        if not self.rag_history:
            return {
                "retrieval_accuracy": 0.5,
                "generation_quality": 0.5,
                "end_to_end_latency": 5.0
            }
        
        recent_records = self.rag_history[-10:]
        
        return {
            "retrieval_accuracy": sum(r.get("confidence", 0.0) for r in recent_records) / len(recent_records),
            "generation_quality": self.rag_metrics.generation_quality,
            "end_to_end_latency": sum(r.get("total_time", 0.0) for r in recent_records) / len(recent_records)
        }
    
    def _should_perform_routine_optimization(self) -> bool:
        """정기 최적화 수행 여부 판단"""
        if not self.optimization_state["last_optimization"]:
            return True
        
        time_since_last = datetime.now() - self.optimization_state["last_optimization"]
        return time_since_last > timedelta(hours=1)  # 1시간마다 정기 최적화
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """RAG 통계 조회"""
        return {
            "rag_metrics": self.rag_metrics.__dict__,
            "adaptive_params": self.adaptive_params,
            "optimization_state": self.optimization_state,
            "real_time_stats": self.real_time_stats,
            "total_queries_processed": len(self.rag_history),
            "agent_status": self.get_status()
        }