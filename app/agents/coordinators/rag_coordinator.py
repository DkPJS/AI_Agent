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
            "generation_temperature": 0.7,
            "max_context_chunks": 5
        }

        # 동적 가중치 시스템
        self.dynamic_weights = {
            "base_weights": {
                "semantic": 0.6,
                "keyword": 0.25,
                "graph": 0.15
            },
            "question_type_adjustments": {
                "factual": {"semantic": 0.7, "keyword": 0.2, "graph": 0.1},
                "comparison": {"semantic": 0.5, "keyword": 0.25, "graph": 0.25},
                "summary": {"semantic": 0.65, "keyword": 0.3, "graph": 0.05},
                "procedural": {"semantic": 0.4, "keyword": 0.4, "graph": 0.2},
                "causal": {"semantic": 0.45, "keyword": 0.15, "graph": 0.4}
            },
            "performance_adjustments": {
                "high_quality": {"semantic": 1.0, "keyword": 1.0, "graph": 1.0},
                "medium_quality": {"semantic": 0.9, "keyword": 1.1, "graph": 1.0},
                "low_quality": {"semantic": 0.8, "keyword": 1.2, "graph": 1.1}
            }
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
        """동적 가중치 기반 적응형 검색 실행"""
        # 질문 유형 분석
        question_type, _ = self.question_analyzer.analyze_question(query)

        # 동적 가중치 계산
        search_weights = self._calculate_dynamic_weights(question_type, query)

        # 다중 검색 방법 실행
        search_results = {}

        # 시맨틱 검색
        if search_weights["semantic"] > 0.1:
            semantic_results = await self.embedder.search_similar_chunks(
                query, limit=min(self.adaptive_params["max_context_chunks"] + 2, 8)
            )
            search_results["semantic"] = semantic_results

        # 키워드 검색 (기존 search_strategy 활용)
        if search_weights["keyword"] > 0.1:
            keyword_context, keyword_sources = await self.search_strategy.execute_search(
                query, question_type, entities
            )
            search_results["keyword"] = keyword_sources

        # 그래프 검색
        if search_weights["graph"] > 0.1:
            graph_context, graph_sources = await self.graph_retriever.retrieve(
                query, limit=min(self.adaptive_params["max_context_chunks"], 5)
            )
            search_results["graph"] = graph_sources

        # 가중치 기반 결과 융합
        context, sources = await self._weighted_fusion_search_results(
            search_results, search_weights, query
        )

        return context, sources

    def _calculate_dynamic_weights(self, question_type: str, query: str) -> Dict[str, float]:
        """질문 유형과 성능 기반 동적 가중치 계산"""
        # 기본 가중치
        base_weights = self.dynamic_weights["base_weights"].copy()

        # 질문 유형별 조정
        if question_type in self.dynamic_weights["question_type_adjustments"]:
            type_weights = self.dynamic_weights["question_type_adjustments"][question_type]
            # 가중 평균으로 조정
            for method in base_weights:
                if method in type_weights:
                    base_weights[method] = (base_weights[method] * 0.4 + type_weights[method] * 0.6)

        # 성능 기반 조정
        performance_level = self._assess_current_performance_level()
        if performance_level in self.dynamic_weights["performance_adjustments"]:
            perf_adjustments = self.dynamic_weights["performance_adjustments"][performance_level]
            for method in base_weights:
                if method in perf_adjustments:
                    base_weights[method] *= perf_adjustments[method]

        # 쿼리 특성 기반 미세 조정
        base_weights = self._fine_tune_weights_by_query(base_weights, query)

        # 정규화
        total = sum(base_weights.values())
        if total > 0:
            base_weights = {k: v/total for k, v in base_weights.items()}

        logging.info(f"동적 가중치 계산: {question_type} → {base_weights}")
        return base_weights

    def _assess_current_performance_level(self) -> str:
        """현재 성능 수준 평가"""
        if not self.rag_history:
            return "medium_quality"

        recent_records = self.rag_history[-10:]
        avg_confidence = sum(r.get("confidence", 0.5) for r in recent_records) / len(recent_records)

        if avg_confidence >= 0.8:
            return "high_quality"
        elif avg_confidence >= 0.6:
            return "medium_quality"
        else:
            return "low_quality"

    def _fine_tune_weights_by_query(self, weights: Dict[str, float], query: str) -> Dict[str, float]:
        """쿼리 특성에 따른 가중치 미세 조정"""
        # 숫자/수치 정보가 많은 쿼리는 키워드 검색 강화
        import re
        if len(re.findall(r'\d+', query)) >= 2:
            weights["keyword"] *= 1.2
            weights["semantic"] *= 0.9

        # 긴 쿼리는 시맨틱 검색 강화
        if len(query.split()) >= 10:
            weights["semantic"] *= 1.1
            weights["keyword"] *= 0.95

        # 관계/연결 표현이 있는 쿼리는 그래프 검색 강화
        relation_keywords = ['관계', '연결', '영향', '원인', '결과', '비교', '차이']
        if any(keyword in query for keyword in relation_keywords):
            weights["graph"] *= 1.3
            weights["semantic"] *= 0.9

        return weights

    async def _weighted_fusion_search_results(
        self, search_results: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float], query: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """가중치 기반 검색 결과 융합"""
        all_results = []
        seen_content = set()

        # 각 검색 방법별로 가중치 적용하여 결과 처리
        for method, results in search_results.items():
            if method not in weights or weights[method] <= 0:
                continue

            method_weight = weights[method]

            for result in results:
                content = result.get("content", "")
                content_hash = hash(content[:100])  # 중복 확인용

                if content_hash not in seen_content and content.strip():
                    # 가중치가 적용된 점수 계산
                    base_relevance = result.get("relevance", 0.5)
                    weighted_score = base_relevance * method_weight

                    # 컨텍스트 품질 보너스
                    quality_bonus = self._calculate_context_quality_bonus(content, query)
                    final_score = weighted_score + quality_bonus

                    enhanced_result = {
                        **result,
                        "weighted_score": final_score,
                        "search_method": method,
                        "method_weight": method_weight,
                        "quality_bonus": quality_bonus
                    }

                    all_results.append(enhanced_result)
                    seen_content.add(content_hash)

        # 점수 기반 정렬
        all_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)

        # 상위 결과 선택
        top_results = all_results[:self.adaptive_params["max_context_chunks"]]

        # 컨텍스트 문자열 구성
        context = "\n\n".join([result.get("content", "") for result in top_results])

        logging.info(f"가중치 융합 완료: {len(search_results)} 방법 → {len(top_results)} 최종 선택")
        return context, top_results

    def _calculate_context_quality_bonus(self, content: str, query: str) -> float:
        """컨텍스트 품질 보너스 계산"""
        if not content or not query:
            return 0.0

        bonus = 0.0

        # 질문 키워드 매칭 보너스
        import re
        query_keywords = set(re.findall(r'[가-힣a-zA-Z]+', query.lower()))
        content_keywords = set(re.findall(r'[가-힣a-zA-Z]+', content.lower()))

        if query_keywords:
            keyword_match_ratio = len(query_keywords.intersection(content_keywords)) / len(query_keywords)
            bonus += keyword_match_ratio * 0.1

        # 내용 길이 적정성 보너스
        content_length = len(content)
        if 200 <= content_length <= 800:
            bonus += 0.05

        # 구조화된 내용 보너스
        if re.search(r'^\d+\.|\n\d+\.|\n-', content):
            bonus += 0.03

        return min(bonus, 0.2)  # 최대 0.2 보너스
    
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
        """고도화된 다층적 답변 품질 평가"""
        if not answer or answer.strip() == "":
            return 0.05

        # 초기 필터링 - 명백한 오류 답변
        if self._is_error_response(answer):
            return 0.1

        # 질문 유형별 평가 가중치
        type_weights = self._get_question_type_weights(question_type)

        # 다층적 품질 평가
        quality_scores = {}

        # 1. 답변 완성도 및 구조 평가
        quality_scores["completeness"] = self._evaluate_answer_completeness(answer, question)

        # 2. 정보 정확성 및 관련성 평가
        quality_scores["relevance"] = self._evaluate_answer_relevance(question, answer, context)

        # 3. 컨텍스트 활용도 평가
        quality_scores["context_utilization"] = self._evaluate_context_utilization(answer, context)

        # 4. 논리적 일관성 평가
        quality_scores["logical_consistency"] = self._evaluate_logical_consistency(answer, question_type)

        # 5. 실용성 및 유용성 평가
        quality_scores["usefulness"] = self._evaluate_answer_usefulness(answer, question, question_type)

        # 6. 언어 품질 평가
        quality_scores["language_quality"] = self._evaluate_language_quality(answer)

        # 질문 유형별 가중 평균 계산
        weighted_score = sum(
            quality_scores[factor] * type_weights[factor]
            for factor in quality_scores if factor in type_weights
        )

        # 추가 보너스/페널티 적용
        final_score = self._apply_quality_adjustments(weighted_score, answer, question, context)

        return max(0.1, min(1.0, final_score))

    def _is_error_response(self, answer: str) -> bool:
        """오류 응답 판별"""
        error_indicators = [
            "오류가 발생했습니다",
            "정보를 찾을 수 없습니다",
            "죄송합니다",
            "응답 생성 중 오류",
            "관련 정보가 없습니다"
        ]
        return any(indicator in answer for indicator in error_indicators)

    def _get_question_type_weights(self, question_type: str) -> Dict[str, float]:
        """질문 유형별 평가 가중치"""
        weight_maps = {
            "factual": {
                "completeness": 0.25,
                "relevance": 0.30,
                "context_utilization": 0.25,
                "logical_consistency": 0.10,
                "usefulness": 0.05,
                "language_quality": 0.05
            },
            "comparison": {
                "completeness": 0.20,
                "relevance": 0.25,
                "context_utilization": 0.20,
                "logical_consistency": 0.25,
                "usefulness": 0.05,
                "language_quality": 0.05
            },
            "summary": {
                "completeness": 0.30,
                "relevance": 0.20,
                "context_utilization": 0.30,
                "logical_consistency": 0.10,
                "usefulness": 0.05,
                "language_quality": 0.05
            },
            "procedural": {
                "completeness": 0.25,
                "relevance": 0.20,
                "context_utilization": 0.15,
                "logical_consistency": 0.20,
                "usefulness": 0.15,
                "language_quality": 0.05
            }
        }
        return weight_maps.get(question_type, weight_maps["factual"])

    def _evaluate_answer_completeness(self, answer: str, question: str) -> float:
        """답변 완성도 평가"""
        score = 0.0

        # 길이 적절성 (30%)
        answer_length = len(answer.strip())
        if 50 <= answer_length <= 500:
            score += 0.3
        elif 20 <= answer_length < 50 or 500 < answer_length <= 1000:
            score += 0.2
        elif answer_length > 1000:
            score += 0.15
        else:
            score += 0.05

        # 문장 구조 (25%)
        sentences = answer.split('.')
        complete_sentences = [s for s in sentences if len(s.strip()) > 5]
        if len(complete_sentences) >= 2:
            score += 0.25
        elif len(complete_sentences) == 1:
            score += 0.15
        else:
            score += 0.05

        # 정보 밀도 (25%)
        import re
        info_elements = len(re.findall(r'[가-힣]{2,}|\d+|[A-Za-z]{3,}', answer))
        info_density = info_elements / len(answer.split()) if answer.split() else 0
        if info_density > 0.7:
            score += 0.25
        elif info_density > 0.5:
            score += 0.15
        else:
            score += 0.05

        # 결론성 (20%)
        if answer.strip().endswith(('.', '다', '습니다', '됩니다')):
            score += 0.2
        else:
            score += 0.1

        return min(1.0, score)

    def _evaluate_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """답변 관련성 평가"""
        import re

        # 키워드 매칭
        question_words = set(re.findall(r'[가-힣a-zA-Z]+', question.lower()))
        answer_words = set(re.findall(r'[가-힣a-zA-Z]+', answer.lower()))

        if not question_words:
            return 0.5

        # 직접 매칭 (60%)
        direct_match_ratio = len(question_words.intersection(answer_words)) / len(question_words)
        relevance_score = direct_match_ratio * 0.6

        # 의미적 관련성 (40%) - 간접적 관련 키워드
        semantic_keywords = self._extract_semantic_keywords(question)
        semantic_match = len(semantic_keywords.intersection(answer_words)) / len(semantic_keywords) if semantic_keywords else 0
        relevance_score += semantic_match * 0.4

        return min(1.0, relevance_score)

    def _extract_semantic_keywords(self, question: str) -> set:
        """질문에서 의미적 관련 키워드 추출"""
        # 간단한 동의어/관련어 매핑
        semantic_map = {
            "방법": ["절차", "과정", "단계"],
            "기준": ["조건", "요건", "표준"],
            "평가": ["심사", "검토", "판단"],
            "지원": ["도움", "보조", "원조"],
            "결과": ["성과", "효과", "산출물"]
        }

        related_words = set()
        for word in question.split():
            if word in semantic_map:
                related_words.update(semantic_map[word])

        return related_words

    def _evaluate_context_utilization(self, answer: str, context: str) -> float:
        """컨텍스트 활용도 평가"""
        if not context or not context.strip():
            return 0.3  # 컨텍스트가 없으면 중간값

        import re
        context_words = set(re.findall(r'[가-힣a-zA-Z]+', context.lower()))
        answer_words = set(re.findall(r'[가-힣a-zA-Z]+', answer.lower()))

        if not context_words:
            return 0.3

        # 컨텍스트 활용률
        utilization_ratio = len(context_words.intersection(answer_words)) / len(context_words)

        # 과도한 복사 방지 - 너무 많이 활용하면 패널티
        if utilization_ratio > 0.8:
            return min(1.0, utilization_ratio * 0.8)
        else:
            return min(1.0, utilization_ratio * 1.2)

    def _evaluate_logical_consistency(self, answer: str, question_type: str) -> float:
        """논리적 일관성 평가"""
        score = 0.5  # 기본 점수

        # 논리적 연결어 사용
        logical_connectors = ["따라서", "그러므로", "하지만", "그러나", "또한", "또는", "즉"]
        connector_count = sum(1 for connector in logical_connectors if connector in answer)
        if connector_count > 0:
            score += min(0.2, connector_count * 0.1)

        # 모순 확인 (기본적인 패턴만)
        contradictions = ["아니다" in answer and "맞다" in answer,
                         "불가능" in answer and "가능" in answer]
        if any(contradictions):
            score -= 0.3

        # 질문 유형별 논리 구조
        if question_type == "comparison" and ("비교" in answer or "차이" in answer):
            score += 0.2

        return max(0.0, min(1.0, score))

    def _evaluate_answer_usefulness(self, answer: str, question: str, question_type: str) -> float:
        """답변 유용성 평가"""
        score = 0.5

        # 구체적 정보 제공 (숫자, 날짜, 구체적 명사)
        import re
        specific_info = len(re.findall(r'\d+|[가-힣]{2,}(?:법|규정|기준|절차)', answer))
        if specific_info > 0:
            score += min(0.3, specific_info * 0.1)

        # 실행 가능한 조언
        actionable_words = ["해야", "필요", "권장", "방법", "단계"]
        actionable_count = sum(1 for word in actionable_words if word in answer)
        if actionable_count > 0:
            score += min(0.2, actionable_count * 0.1)

        return min(1.0, score)

    def _evaluate_language_quality(self, answer: str) -> float:
        """언어 품질 평가"""
        score = 0.0

        # 한국어 자연스러움
        import re
        korean_endings = len(re.findall(r'습니다|입니다|됩니다|다$', answer))
        total_sentences = len([s for s in answer.split('.') if s.strip()])
        if total_sentences > 0:
            politeness_ratio = korean_endings / total_sentences
            score += min(0.4, politeness_ratio * 0.6)

        # 문법적 완성도 (기본적인 체크)
        if not re.search(r'[가-힣]\s+[가-힣]', answer):  # 어색한 띄어쓰기 없음
            score += 0.3

        # 적절한 문장부호 사용
        punctuation_score = min(0.3, answer.count('.') * 0.1 + answer.count(',') * 0.05)
        score += punctuation_score

        return min(1.0, score)

    def _apply_quality_adjustments(self, base_score: float, answer: str, question: str, context: str) -> float:
        """추가 품질 조정"""
        adjusted_score = base_score

        # 출처 표시 보너스
        if "[출처:" in answer or "참고:" in answer:
            adjusted_score += 0.05

        # 답변 길이 극단값 패널티
        if len(answer) < 10:
            adjusted_score *= 0.5
        elif len(answer) > 2000:
            adjusted_score *= 0.9

        # 질문 직접 답변 여부
        if self._directly_answers_question(question, answer):
            adjusted_score += 0.1

        return adjusted_score

    def _directly_answers_question(self, question: str, answer: str) -> bool:
        """질문에 직접 답변하는지 확인"""
        # 질문의 의문사와 답변의 대응 확인
        question_patterns = {
            "무엇": ["것은", "것이", "내용", "정보"],
            "어떻게": ["방법", "절차", "과정"],
            "언제": ["때", "시기", "일정"],
            "어디": ["곳", "장소", "위치"],
            "왜": ["이유", "원인", "목적"]
        }

        for pattern, responses in question_patterns.items():
            if pattern in question and any(response in answer for response in responses):
                return True

        return False
    
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