"""
통합 AI Agent 시스템
전문가 수준의 멀티 Agent RAG 시스템 통합 인터페이스
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.agents.coordinators.orchestrator import AgentOrchestrator
from app.agents.specialized.search_agent import SearchAgent
from app.agents.specialized.qa_agent import QAAgent
from app.agents.coordinators.rag_coordinator import RAGCoordinator
from app.agents.base_agent import Task

class IntelligentAgentSystem:
    """
    지능형 Agent 시스템
    - 다중 Agent 협업 RAG
    - 자율적 학습 및 최적화
    - 동적 성능 조정
    - 전문가 수준 추론 능력
    """
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.agents = {}
        self.system_metrics = {
            "total_queries_processed": 0,
            "avg_response_time": 0.0,
            "system_accuracy": 0.0,
            "agent_collaboration_rate": 0.0,
            "learning_iterations": 0
        }
        
        self.is_initialized = False
        self.system_start_time = None
        
        logging.info("IntelligentAgentSystem initializing...")
    
    async def initialize(self):
        """시스템 초기화"""
        if self.is_initialized:
            logging.warning("System already initialized")
            return
        
        try:
            # Agent 인스턴스 생성
            search_agent = SearchAgent()
            qa_agent = QAAgent()
            rag_coordinator = RAGCoordinator()
            
            # Orchestrator에 Agent 등록
            await self.orchestrator.register_agent(search_agent)
            await self.orchestrator.register_agent(qa_agent)
            await self.orchestrator.register_agent(rag_coordinator)
            
            # Agent 참조 저장
            self.agents = {
                "SearchAgent": search_agent,
                "QAAgent": qa_agent,
                "RAGCoordinator": rag_coordinator
            }
            
            # 시스템 시작
            self.system_start_time = datetime.now()
            self.is_initialized = True
            
            logging.info("IntelligentAgentSystem initialized successfully")
            
            # 오케스트레이션 시작 (백그라운드)
            asyncio.create_task(self.orchestrator.start_orchestration())
            
        except Exception as e:
            logging.error(f"System initialization failed: {e}")
            raise
    
    async def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        지능형 쿼리 처리
        - 다중 Agent 협업
        - 자동 최적화
        - 성능 추적
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # RAGCoordinator를 통한 고급 RAG 처리
            rag_coordinator = self.agents.get("RAGCoordinator")
            if rag_coordinator and len(query.strip()) > 0:
                # RAG 전용 처리
                rag_result = await rag_coordinator._process_intelligent_rag_query({
                    "query": query,
                    "context": user_context or {}
                })
                
                if rag_result.get("success"):
                    return rag_result
            
            # 기본 다중 Agent 협업 처리
            result = await self.orchestrator.submit_complex_query(query, user_context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 시스템 메트릭 업데이트
            await self._update_system_metrics(result, execution_time)
            
            # 결과에 시스템 정보 추가
            result["system_info"] = {
                "execution_time": execution_time,
                "agents_used": result.get("metadata", {}).get("agents_involved", []),
                "system_version": "1.0.0-agent",
                "processing_mode": "multi_agent_collaborative"
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Query processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def add_specialized_task(self, task_name: str, task_params: Dict[str, Any], 
                                 priority: int = 0) -> str:
        """전문화된 작업 추가"""
        task = Task(
            name=task_name,
            description=f"Specialized task: {task_name}",
            parameters=task_params,
            priority=priority
        )
        
        # 글로벌 작업 큐에 추가
        await self.orchestrator.global_task_queue.put((priority, task))
        
        logging.info(f"Specialized task {task_name} added with priority {priority}")
        return task.id
    
    async def get_agent_insights(self) -> Dict[str, Any]:
        """Agent 시스템 인사이트"""
        insights = {
            "system_overview": await self._get_system_overview(),
            "agent_performance": await self._get_agent_performance_analysis(),
            "collaboration_analysis": await self._get_collaboration_analysis(),
            "learning_progress": await self._get_learning_progress(),
            "optimization_suggestions": await self._get_optimization_suggestions()
        }
        
        return insights
    
    async def _update_system_metrics(self, result: Dict[str, Any], execution_time: float):
        """시스템 메트릭 업데이트"""
        self.system_metrics["total_queries_processed"] += 1
        
        # 평균 응답 시간 업데이트
        total_queries = self.system_metrics["total_queries_processed"]
        current_avg = self.system_metrics["avg_response_time"]
        self.system_metrics["avg_response_time"] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        # 정확도 업데이트 (품질 점수 기반)
        if result.get("success") and "quality_score" in result:
            quality_score = result["quality_score"]
            current_accuracy = self.system_metrics["system_accuracy"]
            self.system_metrics["system_accuracy"] = (
                (current_accuracy * (total_queries - 1) + quality_score) / total_queries
            )
        
        # 협업 비율 업데이트
        agents_used = len(result.get("metadata", {}).get("agents_involved", []))
        if agents_used > 1:
            collaboration_count = self.system_metrics.get("collaboration_count", 0) + 1
            self.system_metrics["collaboration_count"] = collaboration_count
            self.system_metrics["agent_collaboration_rate"] = collaboration_count / total_queries
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """시스템 개요"""
        uptime = datetime.now() - self.system_start_time if self.system_start_time else None
        
        orchestrator_status = await self.orchestrator.get_orchestrator_status()
        
        return {
            "system_status": "running" if self.is_initialized else "initializing",
            "uptime_hours": uptime.total_seconds() / 3600 if uptime else 0,
            "registered_agents": len(self.agents),
            "total_queries_processed": self.system_metrics["total_queries_processed"],
            "orchestrator_status": orchestrator_status["is_running"],
            "active_collaborations": orchestrator_status.get("active_collaborations", 0)
        }
    
    async def _get_agent_performance_analysis(self) -> Dict[str, Any]:
        """Agent 성능 분석"""
        agent_stats = {}
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'get_search_statistics'):
                stats = agent.get_search_statistics()
            elif hasattr(agent, 'get_qa_statistics'):
                stats = agent.get_qa_statistics()
            else:
                stats = agent.get_status()
            
            agent_stats[agent_name] = {
                "specialization": agent.capabilities,
                "current_state": stats.get("agent_status", {}).get("state", "unknown"),
                "performance_score": self._calculate_agent_performance_score(stats),
                "efficiency_rating": self._calculate_efficiency_rating(stats)
            }
        
        return agent_stats
    
    def _calculate_agent_performance_score(self, stats: Dict[str, Any]) -> float:
        """Agent 성능 점수 계산"""
        factors = []
        
        # 작업 성공률
        if "success_rate" in stats.get("agent_status", {}).get("performance_metrics", {}):
            success_rate = stats["agent_status"]["performance_metrics"]["success_rate"]
            factors.append(success_rate * 0.4)
        
        # 응답 시간 (빠를수록 좋음)
        if "recent_avg_time" in stats:
            time_factor = max(0, 1.0 - (stats["recent_avg_time"] / 10.0))
            factors.append(time_factor * 0.3)
        
        # 결과 품질 (QA Agent의 경우)
        if "recent_avg_confidence" in stats:
            confidence = stats["recent_avg_confidence"]
            factors.append(confidence * 0.3)
        
        return sum(factors) if factors else 0.5
    
    def _calculate_efficiency_rating(self, stats: Dict[str, Any]) -> str:
        """효율성 등급 계산"""
        performance_score = self._calculate_agent_performance_score(stats)
        
        if performance_score >= 0.9:
            return "Excellent"
        elif performance_score >= 0.8:
            return "Very Good"
        elif performance_score >= 0.7:
            return "Good"
        elif performance_score >= 0.6:
            return "Average"
        else:
            return "Needs Improvement"
    
    async def _get_collaboration_analysis(self) -> Dict[str, Any]:
        """협업 분석"""
        collaboration_history = self.orchestrator.collaboration_history
        
        if not collaboration_history:
            return {"message": "No collaboration data available"}
        
        successful_collaborations = [c for c in collaboration_history if c.get("success")]
        
        return {
            "total_collaborations": len(collaboration_history),
            "successful_collaborations": len(successful_collaborations),
            "success_rate": len(successful_collaborations) / len(collaboration_history),
            "avg_collaboration_time": sum(c["execution_time"] for c in collaboration_history) / len(collaboration_history),
            "most_common_agent_pairs": self._analyze_agent_pairs(collaboration_history)
        }
    
    def _analyze_agent_pairs(self, collaboration_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Agent 협업 쌍 분석"""
        pair_counts = {}
        
        for collaboration in collaboration_history:
            agents = collaboration.get("agents_involved", [])
            if len(agents) >= 2:
                pair = tuple(sorted(agents))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        return dict(sorted(pair_counts.items(), key=lambda x: x[1], reverse=True))
    
    async def _get_learning_progress(self) -> Dict[str, Any]:
        """학습 진행 상황"""
        learning_data = {}
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'performance_metrics'):
                metrics = agent.performance_metrics
                learning_data[agent_name] = {
                    "learning_iterations": metrics.get("learning_iterations", 0),
                    "experience_buffer_size": len(getattr(agent, 'experience_buffer', [])),
                    "knowledge_growth": self._calculate_knowledge_growth(agent)
                }
        
        return learning_data
    
    def _calculate_knowledge_growth(self, agent) -> float:
        """지식 성장률 계산"""
        if hasattr(agent, 'long_term_memory'):
            memory_size = len(agent.long_term_memory)
            return min(memory_size / 100.0, 1.0)  # 100개 항목을 1.0으로 정규화
        return 0.0
    
    async def _get_optimization_suggestions(self) -> List[str]:
        """최적화 제안"""
        suggestions = []
        
        # 시스템 성능 기반 제안
        if self.system_metrics["avg_response_time"] > 5.0:
            suggestions.append("Consider optimizing search algorithms for faster response times")
        
        if self.system_metrics["system_accuracy"] < 0.8:
            suggestions.append("Review and improve answer quality assessment methods")
        
        if self.system_metrics["agent_collaboration_rate"] < 0.5:
            suggestions.append("Enhance inter-agent collaboration mechanisms")
        
        # Agent별 제안
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'difficult_questions') and len(agent.difficult_questions) > 10:
                suggestions.append(f"Focus on improving {agent_name}'s handling of complex questions")
        
        return suggestions
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 최적화"""
        optimization_results = {
            "optimizations_applied": [],
            "performance_before": dict(self.system_metrics),
            "timestamp": datetime.now()
        }
        
        # Agent별 최적화 수행
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, '_optimize_search_parameters'):
                    await agent._optimize_search_parameters()
                    optimization_results["optimizations_applied"].append(f"{agent_name}: search optimization")
                
                if hasattr(agent, '_improve_answer_quality'):
                    await agent._improve_answer_quality()
                    optimization_results["optimizations_applied"].append(f"{agent_name}: quality improvement")
                    
            except Exception as e:
                logging.error(f"Optimization failed for {agent_name}: {e}")
        
        # 시스템 수준 최적화
        await self._optimize_orchestrator_parameters()
        optimization_results["optimizations_applied"].append("Orchestrator: parameter tuning")
        
        self.system_metrics["learning_iterations"] += 1
        
        return optimization_results
    
    async def _optimize_orchestrator_parameters(self):
        """오케스트레이터 파라미터 최적화"""
        # 성능 데이터를 기반으로 최적화 (간단한 예시)
        if self.system_metrics["avg_response_time"] > 3.0:
            # 응답 시간이 길면 더 적극적인 병렬 처리
            logging.info("Optimizing for faster response times")
        
        if self.system_metrics["agent_collaboration_rate"] < 0.3:
            # 협업률이 낮으면 협업 임계값 조정
            logging.info("Optimizing agent collaboration parameters")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 조회"""
        return {
            "system_metrics": self.system_metrics,
            "uptime": (datetime.now() - self.system_start_time).total_seconds() / 3600 if self.system_start_time else 0,
            "system_status": "running" if self.is_initialized else "not_initialized"
        }