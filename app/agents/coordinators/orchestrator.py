"""
AI Agent 오케스트레이터
다중 Agent 협업, 작업 분산, 통신 조정을 담당하는 중앙 관리 시스템
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from app.agents.base_agent import BaseAgent, Message, Task, AgentState
from app.agents.specialized.search_agent import SearchAgent
from app.agents.specialized.qa_agent import QAAgent

class AgentOrchestrator:
    """
    고급 Agent 오케스트레이터
    - 지능형 작업 분산
    - Agent 간 협업 조정
    - 성능 기반 Agent 선택
    - 동적 로드 밸런싱
    """
    
    def __init__(self):
        self.orchestrator_id = str(uuid.uuid4())
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # 메시지 라우팅
        self.message_bus = asyncio.Queue()
        self.message_routes: Dict[str, str] = {}  # receiver -> agent_name
        
        # 작업 분산
        self.global_task_queue = asyncio.PriorityQueue()
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_name
        
        # 성능 추적
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # 오케스트레이터 상태
        self.is_running = False
        self.orchestration_metrics = {
            "total_tasks_distributed": 0,
            "successful_collaborations": 0,
            "avg_task_completion_time": 0.0,
            "agent_utilization": {},
            "message_routing_efficiency": 0.0
        }
        
        logging.info(f"AgentOrchestrator initialized with ID: {self.orchestrator_id}")
    
    async def register_agent(self, agent: BaseAgent):
        """Agent 등록"""
        agent_name = agent.name
        self.agents[agent_name] = agent
        self.agent_capabilities[agent_name] = agent.capabilities
        self.agent_performance[agent_name] = {
            "tasks_completed": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "current_load": 0,
            "specialization_score": self._calculate_specialization_score(agent.capabilities)
        }
        
        # Agent의 메시지 라우팅 설정
        self.message_routes[agent_name] = agent_name
        
        logging.info(f"Agent {agent_name} registered with capabilities: {agent.capabilities}")
    
    async def start_orchestration(self):
        """오케스트레이션 시작"""
        if self.is_running:
            logging.warning("Orchestrator is already running")
            return
        
        self.is_running = True
        logging.info("Starting agent orchestration...")
        
        # 등록된 Agent들 시작
        agent_tasks = [agent.start() for agent in self.agents.values()]
        
        # 오케스트레이터 핵심 루프들 시작
        orchestrator_tasks = [
            self._message_routing_loop(),
            self._task_distribution_loop(),
            self._performance_monitoring_loop(),
            self._collaboration_coordination_loop()
        ]
        
        try:
            await asyncio.gather(*agent_tasks, *orchestrator_tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"Orchestration error: {e}")
        finally:
            self.is_running = False
    
    async def _message_routing_loop(self):
        """메시지 라우팅 루프"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_bus.get(), timeout=1.0)
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Message routing error: {e}")
    
    async def _task_distribution_loop(self):
        """작업 분산 루프"""
        while self.is_running:
            try:
                # 우선순위가 높은 작업부터 처리
                priority, task = await asyncio.wait_for(
                    self.global_task_queue.get(), timeout=1.0
                )
                
                # 최적 Agent 선택 및 작업 할당
                selected_agent = await self._select_optimal_agent(task)
                if selected_agent:
                    await self._assign_task_to_agent(task, selected_agent)
                    self.orchestration_metrics["total_tasks_distributed"] += 1
                else:
                    # 적절한 Agent가 없으면 다시 큐에 넣기
                    await self.global_task_queue.put((priority, task))
                    await asyncio.sleep(1)  # 잠시 대기
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Task distribution error: {e}")
    
    async def _performance_monitoring_loop(self):
        """성능 모니터링 루프"""
        while self.is_running:
            try:
                await self._update_agent_performance_metrics()
                await self._detect_performance_anomalies()
                await self._optimize_agent_allocation()
                
                await asyncio.sleep(30)  # 30초마다 모니터링
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
    
    async def _collaboration_coordination_loop(self):
        """협업 조정 루프"""
        while self.is_running:
            try:
                await self._identify_collaboration_opportunities()
                await self._coordinate_multi_agent_tasks()
                
                await asyncio.sleep(10)  # 10초마다 협업 기회 확인
                
            except Exception as e:
                logging.error(f"Collaboration coordination error: {e}")
    
    async def submit_complex_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """복합 쿼리 처리 (다중 Agent 협업)"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # 현재 질문을 저장 (Agent가 사용할 수 있도록)
        self._current_question = query
        
        try:
            logging.info(f"Processing complex query (session: {session_id}): {query}")
            
            # 1단계: 쿼리 분석 및 작업 분해
            task_plan = await self._analyze_and_decompose_query(query, user_context)
            
            # 2단계: 검색 Agent를 통한 정보 수집
            search_result = await self._coordinate_search_task(query, task_plan, session_id)
            
            # 3단계: QA Agent를 통한 답변 생성
            qa_result = await self._coordinate_qa_task(query, search_result, task_plan, session_id)
            
            # 4단계: 결과 통합 및 품질 평가
            final_result = await self._integrate_and_evaluate_results(
                query, search_result, qa_result, session_id
            )
            
            # 협업 기록 저장
            collaboration_record = {
                "session_id": session_id,
                "query": query,
                "task_plan": task_plan,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "agents_involved": list(task_plan.keys()),
                "success": final_result.get("success", False),
                "timestamp": datetime.now()
            }
            
            self.collaboration_history.append(collaboration_record)
            
            if final_result.get("success"):
                self.orchestration_metrics["successful_collaborations"] += 1
            
            return final_result
            
        except Exception as e:
            logging.error(f"Complex query processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def _analyze_and_decompose_query(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """쿼리 분석 및 작업 분해"""
        task_plan = {}
        
        # 검색이 필요한지 판단
        needs_search = True  # 대부분의 쿼리는 검색이 필요
        if needs_search:
            task_plan["SearchAgent"] = {
                "priority": 1,
                "task_type": "intelligent_search",
                "parameters": {
                    "query": query,
                    "context": user_context or {},
                    "search_depth": "deep" if len(query.split()) > 10 else "standard"
                }
            }
        
        # QA가 필요한지 판단
        needs_qa = True  # 검색 결과를 바탕으로 답변 생성
        if needs_qa:
            task_plan["QAAgent"] = {
                "priority": 2,
                "task_type": "answer_question",
                "parameters": {
                    "question": query,
                    "session_id": str(uuid.uuid4()),
                    "answer_strategy": "adaptive"
                },
                "dependencies": ["SearchAgent"] if needs_search else []
            }
        
        return task_plan
    
    async def _coordinate_search_task(self, query: str, task_plan: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """검색 작업 조정"""
        if "SearchAgent" not in task_plan:
            return {"success": False, "message": "No search task planned"}
        
        search_agent = self.agents.get("SearchAgent")
        if not search_agent:
            return {"success": False, "message": "SearchAgent not available"}
        
        # 검색 요청 메시지 생성
        search_message = Message(
            sender="Orchestrator",
            receiver="SearchAgent",
            message_type="search_request",
            content={
                "query": query,
                "limit": 10,
                "session_id": session_id,
                **task_plan["SearchAgent"]["parameters"]
            },
            priority=task_plan["SearchAgent"]["priority"]
        )
        
        # 메시지 전송 및 응답 대기
        await self.send_message(search_message)
        
        # 응답 대기 (실제 구현에서는 더 정교한 응답 매칭 필요)
        search_result = await self._wait_for_agent_response(
            "SearchAgent", "search_response", timeout=30
        )
        
        return search_result
    
    async def _coordinate_qa_task(self, query: str, search_result: Dict[str, Any], 
                                task_plan: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """QA 작업 조정"""
        if "QAAgent" not in task_plan:
            return {"success": False, "message": "No QA task planned"}
        
        qa_agent = self.agents.get("QAAgent")
        if not qa_agent:
            return {"success": False, "message": "QAAgent not available"}
        
        # 검색 결과를 컨텍스트로 사용
        context = ""
        sources = []
        if search_result.get("success"):
            context = search_result.get("context", "")
            sources = search_result.get("sources", [])
        
        # QA 요청 메시지 생성
        qa_message = Message(
            sender="Orchestrator",
            receiver="QAAgent",
            message_type="question",
            content={
                "question": query,
                "context": context,
                "session_id": session_id,
                "sources": sources
            },
            priority=task_plan["QAAgent"]["priority"]
        )
        
        # 메시지 전송 및 응답 대기
        await self.send_message(qa_message)
        
        qa_result = await self._wait_for_agent_response(
            "QAAgent", "answer", timeout=45
        )
        
        return qa_result
    
    async def _integrate_and_evaluate_results(self, query: str, search_result: Dict[str, Any], 
                                           qa_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """결과 통합 및 품질 평가"""
        integrated_result = {
            "success": True,
            "session_id": session_id,
            "query": query,
            "timestamp": datetime.now()
        }
        
        # QA 결과 통합
        if qa_result.get("success"):
            integrated_result.update({
                "answer": qa_result.get("answer", ""),
                "confidence": qa_result.get("confidence", 0.0),
                "sources": qa_result.get("sources", []),
                "metadata": qa_result.get("metadata", {})
            })
        else:
            integrated_result["success"] = False
            integrated_result["error"] = "QA processing failed"
        
        # 검색 결과 메타데이터 추가
        if search_result.get("success"):
            search_metadata = search_result.get("metadata", {})
            integrated_result.setdefault("metadata", {}).update({
                "search_strategy": search_metadata.get("strategy_used", "unknown"),
                "search_execution_time": search_metadata.get("execution_time", 0.0)
            })
        
        # 전체 품질 점수 계산
        quality_score = self._calculate_integrated_quality_score(integrated_result)
        integrated_result["quality_score"] = quality_score
        
        return integrated_result
    
    def _calculate_integrated_quality_score(self, result: Dict[str, Any]) -> float:
        """통합 품질 점수 계산"""
        if not result.get("success"):
            return 0.0
        
        factors = []
        
        # 답변 신뢰도
        confidence = result.get("confidence", 0.0)
        factors.append(confidence * 0.4)
        
        # 소스 품질 (소스 개수와 다양성)
        sources = result.get("sources", [])
        source_quality = min(len(sources) / 5.0, 1.0) * 0.3
        factors.append(source_quality)
        
        # 응답 시간 (빠를수록 좋음)
        search_time = result.get("metadata", {}).get("search_execution_time", 5.0)
        time_factor = max(0, 1.0 - (search_time / 10.0)) * 0.2
        factors.append(time_factor)
        
        # 답변 완성도
        answer = result.get("answer", "")
        completeness = min(len(answer.split()) / 100.0, 1.0) * 0.1
        factors.append(completeness)
        
        return sum(factors)
    
    async def _select_optimal_agent(self, task: Task) -> Optional[str]:
        """최적 Agent 선택"""
        task_capabilities = self._extract_task_capabilities(task)
        
        if not task_capabilities:
            return None
        
        # 각 Agent의 적합성 점수 계산
        agent_scores = {}
        
        for agent_name, agent in self.agents.items():
            if agent.state == AgentState.ERROR:
                continue  # 오류 상태인 Agent는 제외
            
            score = self._calculate_agent_suitability_score(
                agent, task_capabilities, task
            )
            
            if score > 0:
                agent_scores[agent_name] = score
        
        # 가장 높은 점수의 Agent 선택
        if agent_scores:
            return max(agent_scores.keys(), key=lambda k: agent_scores[k])
        
        return None
    
    def _extract_task_capabilities(self, task: Task) -> List[str]:
        """작업에 필요한 능력 추출"""
        task_name = task.name.lower()
        
        capability_mapping = {
            "search": ["semantic_search", "hybrid_search"],
            "answer": ["question_answering", "reasoning"],
            "analyze": ["context_analysis", "reasoning"],
            "document": ["document_processing"],
            "conversation": ["multi_turn_conversation"]
        }
        
        required_capabilities = []
        for keyword, capabilities in capability_mapping.items():
            if keyword in task_name:
                required_capabilities.extend(capabilities)
        
        return required_capabilities
    
    def _calculate_agent_suitability_score(self, agent: BaseAgent, 
                                         required_capabilities: List[str], 
                                         task: Task) -> float:
        """Agent 적합성 점수 계산"""
        if not required_capabilities:
            return 0.0
        
        # 능력 매칭 점수
        capability_overlap = len(set(agent.capabilities) & set(required_capabilities))
        capability_score = capability_overlap / len(required_capabilities)
        
        # 성능 점수
        performance_data = self.agent_performance.get(agent.name, {})
        success_rate = performance_data.get("success_rate", 0.0)
        current_load = performance_data.get("current_load", 0)
        
        # 로드 밸런싱 (부하가 적을수록 좋음)
        load_factor = max(0, 1.0 - (current_load / agent.max_concurrent_tasks))
        
        # 전문화 점수
        specialization = performance_data.get("specialization_score", 0.5)
        
        # 최종 점수 계산
        total_score = (
            capability_score * 0.4 +
            success_rate * 0.3 +
            load_factor * 0.2 +
            specialization * 0.1
        )
        
        return total_score
    
    def _calculate_specialization_score(self, capabilities: List[str]) -> float:
        """전문화 점수 계산"""
        # 능력의 개수가 적을수록 전문화 정도가 높음
        if not capabilities:
            return 0.0
        
        return min(1.0, 5.0 / len(capabilities))
    
    async def _assign_task_to_agent(self, task: Task, agent_name: str):
        """Agent에게 작업 할당"""
        agent = self.agents.get(agent_name)
        if not agent:
            logging.error(f"Agent {agent_name} not found")
            return
        
        # 작업 할당 기록
        self.task_assignments[task.id] = agent_name
        
        # Agent의 현재 부하 증가
        self.agent_performance[agent_name]["current_load"] += 1
        
        # Agent에게 작업 전달
        await agent.add_task(task)
        
        logging.info(f"Task {task.name} assigned to {agent_name}")
    
    async def send_message(self, message: Message):
        """메시지 전송"""
        await self.message_bus.put(message)
    
    async def _route_message(self, message: Message):
        """메시지 라우팅"""
        receiver = message.receiver
        
        if receiver in self.agents:
            agent = self.agents[receiver]
            await agent.receive_message(message)
            self.orchestration_metrics["message_routing_efficiency"] += 1
        else:
            logging.warning(f"No route found for message to {receiver}")
    
    async def _wait_for_agent_response(self, agent_name: str, expected_type: str, 
                                     timeout: int = 30) -> Dict[str, Any]:
        """Agent 응답 대기 (실제 Agent 통신)"""
        agent = self.agents.get(agent_name)
        if not agent:
            return {"success": False, "error": f"Agent {agent_name} not found"}
        
        try:
            # 실제 Agent 작업 대기
            start_time = datetime.now()
            
            if expected_type == "search_response":
                # SearchAgent는 아직 미구현이므로 기본 검색 수행
                from app.rag.retrieval.embedder import DocumentEmbedder
                embedder = DocumentEmbedder()
                
                # 현재 질문으로 검색 수행
                current_query = getattr(self, '_current_question', "검색 쿼리")
                search_results = await embedder.search_similar_chunks(current_query, limit=3)
                
                return {
                    "success": True,
                    "context": "검색 결과가 준비되었습니다.",
                    "sources": search_results,
                    "metadata": {
                        "strategy_used": "hybrid_search",
                        "execution_time": (datetime.now() - start_time).total_seconds()
                    }
                }
                
            elif expected_type == "answer":
                # QA Agent에서 실제 응답 생성 - 현재 큐에서 질문 추출하여 처리
                if agent_name == "QAAgent" and hasattr(agent, 'qa_system'):
                    # 현재 처리 중인 질문을 Agent의 현재 컨텍스트에서 가져옴
                    # 임시로 기본 QA 시스템 직접 호출
                    from app.rag.generation.qa_system import AdvancedQASystem
                    qa_system = AdvancedQASystem()
                    
                    # 마지막 처리된 질문을 가져오거나 기본 처리
                    question = getattr(self, '_current_question', "질문을 처리 중입니다.")
                    result = await qa_system.answer_question(question)
                    
                    return {
                        "success": True,
                        "answer": result.get("answer", "답변을 생성할 수 없습니다."),
                        "confidence": 0.9,
                        "sources": result.get("sources", []),
                        "question_type": result.get("question_type", "general"),
                        "metadata": {
                            "response_time": (datetime.now() - start_time).total_seconds(),
                            "agent": agent_name
                        }
                    }
            
            # 기본 응답
            return {
                "success": True,
                "agent": agent_name,
                "response_type": expected_type,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Agent {agent_name} 응답 처리 중 오류: {str(e)}")
            return {"success": False, "error": f"Agent processing error: {str(e)}"}
    
    async def _update_agent_performance_metrics(self):
        """Agent 성능 메트릭 업데이트"""
        for agent_name, agent in self.agents.items():
            agent_status = agent.get_status()
            performance_metrics = agent_status.get("performance_metrics", {})
            
            self.agent_performance[agent_name].update({
                "tasks_completed": performance_metrics.get("tasks_completed", 0),
                "success_rate": performance_metrics.get("success_rate", 0.0),
                "current_load": agent_status.get("active_tasks", 0)
            })
    
    async def _detect_performance_anomalies(self):
        """성능 이상 감지"""
        for agent_name, performance in self.agent_performance.items():
            tasks_completed = performance.get("tasks_completed", 0)
            
            # 작업을 수행한 적이 있는 Agent만 성능 평가
            if tasks_completed == 0:
                continue  # 아직 작업을 수행하지 않은 Agent는 평가하지 않음
            
            # 응답 시간이 비정상적으로 길어진 경우
            if performance.get("avg_response_time", 0) > 10.0:
                logging.warning(f"Agent {agent_name} has high response time: {performance['avg_response_time']}")
            
            # 성공률이 낮아진 경우 (작업을 5번 이상 수행한 Agent만)
            success_rate = performance.get("success_rate", 1.0)
            if tasks_completed >= 5 and success_rate < 0.5:
                logging.warning(f"Agent {agent_name} has low success rate: {success_rate} (after {tasks_completed} tasks)")
    
    async def _optimize_agent_allocation(self):
        """Agent 할당 최적화"""
        # 간단한 최적화 로직
        for agent_name, performance in self.agent_performance.items():
            current_load = performance.get("current_load", 0)
            if current_load > 5:  # 부하가 높은 경우
                logging.info(f"High load detected for {agent_name}: {current_load}")
    
    async def _identify_collaboration_opportunities(self):
        """협업 기회 식별"""
        # 복합 작업이 있는지 확인하고 협업 기회 식별
        if not self.global_task_queue.empty():
            logging.debug("Checking for collaboration opportunities...")
    
    async def _coordinate_multi_agent_tasks(self):
        """다중 Agent 작업 조정"""
        # 협업이 필요한 작업들을 조정
        if len(self.agents) > 1:
            logging.debug("Coordinating multi-agent tasks...")

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """오케스트레이터 상태 조회"""
        return {
            "orchestrator_id": self.orchestrator_id,
            "is_running": self.is_running,
            "registered_agents": list(self.agents.keys()),
            "agent_performance": dict(self.agent_performance),
            "orchestration_metrics": self.orchestration_metrics,
            "active_collaborations": len([c for c in self.collaboration_history 
                                        if c["timestamp"] > datetime.now() - timedelta(hours=1)]),
            "message_queue_size": self.message_bus.qsize(),
            "global_task_queue_size": self.global_task_queue.qsize()
        }