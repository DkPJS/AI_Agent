"""
AI Agent 기본 프레임워크
전문가 수준의 Agent 아키텍처 구현
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

class AgentState(Enum):
    """Agent 상태"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMMUNICATING = "communicating"
    LEARNING = "learning"
    ERROR = "error"

@dataclass
class Message:
    """Agent 간 메시지"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: str = "info"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0=낮음, 1=보통, 2=높음, 3=긴급

@dataclass
class Task:
    """Agent 작업"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0
    estimated_duration: Optional[timedelta] = None

class BaseAgent(ABC):
    """
    고급 AI Agent 기본 클래스
    - 자율성: 독립적 의사결정
    - 반응성: 환경 변화 감지 및 대응  
    - 사회성: 다른 Agent와 협업
    - 능동성: 목표 지향적 행동
    """
    
    def __init__(self, name: str, capabilities: List[str], max_concurrent_tasks: int = 5):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Agent 메모리 시스템
        self.short_term_memory = []  # 최근 경험
        self.long_term_memory = {}   # 학습된 패턴
        self.working_memory = {}     # 현재 작업 컨텍스트
        
        # 작업 관리
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        
        # 통신 시스템
        self.message_queue = asyncio.Queue()
        self.known_agents: Dict[str, str] = {}  # agent_name -> agent_id
        self.communication_handlers: Dict[str, Callable] = {}
        
        # 성능 메트릭
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0,  # 초기값을 1.0으로 설정 (작업이 없을 때는 완벽한 상태로 간주)
            "learning_iterations": 0
        }
        
        # 목표 시스템
        self.goals: List[Dict[str, Any]] = []
        self.current_goal = None
        
        # 학습 시스템
        self.learning_rate = 0.1
        self.experience_buffer = []
        
        logging.info(f"Agent {self.name} ({self.agent_id}) initialized with capabilities: {capabilities}")
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """메시지 처리 (각 Agent가 구현)"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """작업 실행 (각 Agent가 구현)"""
        pass
    
    @abstractmethod  
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정 (각 Agent가 구현)"""
        pass
    
    async def start(self):
        """Agent 시작 - 메인 실행 루프"""
        logging.info(f"Agent {self.name} starting...")
        
        # 동시 실행할 코루틴들
        await asyncio.gather(
            self._main_loop(),
            self._message_handler(),
            self._task_executor(),
            self._learning_loop(),
            return_exceptions=True
        )
    
    async def _main_loop(self):
        """메인 Agent 루프 - 자율적 행동"""
        while True:
            try:
                self.state = AgentState.THINKING
                
                # 1. 환경 상황 평가
                context = await self._assess_situation()
                
                # 2. 목표 확인 및 우선순위 조정
                await self._update_goals(context)
                
                # 3. 의사결정
                if self.current_goal:
                    decision = await self.make_decision(context)
                    if decision.get("action"):
                        await self._execute_decision(decision)
                
                # 4. 자기 성찰 및 개선
                await self._self_reflection()
                
                self.state = AgentState.IDLE
                await asyncio.sleep(1)  # 1초 대기
                
            except Exception as e:
                self.state = AgentState.ERROR
                logging.error(f"Agent {self.name} main loop error: {e}")
                await asyncio.sleep(5)  # 에러 시 5초 대기
    
    async def _message_handler(self):
        """메시지 처리 루프"""
        while True:
            try:
                message = await self.message_queue.get()
                self.state = AgentState.COMMUNICATING
                
                # 메시지를 단기 기억에 저장
                self.short_term_memory.append({
                    "type": "message",
                    "content": message,
                    "timestamp": datetime.now()
                })
                
                # 메시지 처리
                response = await self.process_message(message)
                
                # 응답 메시지가 있으면 전송
                if response:
                    await self._send_message(response)
                    
            except Exception as e:
                logging.error(f"Agent {self.name} message handler error: {e}")
    
    async def _task_executor(self):
        """작업 실행 루프"""
        while True:
            try:
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    # 우선순위가 높은 작업부터 처리
                    try:
                        priority, task = await asyncio.wait_for(
                            self.task_queue.get(), timeout=1.0
                        )
                        
                        # 작업 실행
                        asyncio.create_task(self._execute_task_wrapper(task))
                        
                    except asyncio.TimeoutError:
                        pass  # 대기 중인 작업이 없음
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Agent {self.name} task executor error: {e}")
    
    async def _execute_task_wrapper(self, task: Task):
        """작업 실행 래퍼"""
        start_time = datetime.now()
        self.active_tasks[task.id] = task
        task.status = "in_progress"
        
        try:
            self.state = AgentState.ACTING
            result = await self.execute_task(task)
            
            task.status = "completed"
            self.performance_metrics["tasks_completed"] += 1
            
            # 경험을 학습 버퍼에 저장
            experience = {
                "task": task,
                "result": result,
                "duration": datetime.now() - start_time,
                "success": True
            }
            self.experience_buffer.append(experience)
            
        except Exception as e:
            task.status = "failed"
            self.performance_metrics["tasks_failed"] += 1
            logging.error(f"Task {task.name} failed: {e}")
            
            # 실패 경험도 학습
            experience = {
                "task": task,
                "error": str(e),
                "duration": datetime.now() - start_time,
                "success": False
            }
            self.experience_buffer.append(experience)
            
        finally:
            # 작업 완료 후 정리
            del self.active_tasks[task.id]
            self.completed_tasks.append(task)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics()
    
    async def _learning_loop(self):
        """학습 루프"""
        while True:
            try:
                if len(self.experience_buffer) >= 10:  # 10개 경험이 쌓이면 학습
                    self.state = AgentState.LEARNING
                    await self._learn_from_experience()
                    
                await asyncio.sleep(30)  # 30초마다 학습 기회 확인
                
            except Exception as e:
                logging.error(f"Agent {self.name} learning loop error: {e}")
    
    async def _assess_situation(self) -> Dict[str, Any]:
        """현재 상황 평가"""
        return {
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize(),
            "recent_messages": len([m for m in self.short_term_memory 
                                  if m.get("timestamp", datetime.min) > datetime.now() - timedelta(minutes=5)]),
            "current_state": self.state.value,
            "performance": self.performance_metrics,
            "memory_usage": len(self.short_term_memory)
        }
    
    async def _update_goals(self, context: Dict[str, Any]):
        """목표 업데이트"""
        # 현재 목표가 없고 대기 중인 작업이 있으면 새 목표 설정
        if not self.current_goal and context["pending_tasks"] > 0:
            self.current_goal = {
                "type": "complete_tasks",
                "priority": 1,
                "created_at": datetime.now()
            }
    
    async def _execute_decision(self, decision: Dict[str, Any]):
        """의사결정 실행"""
        action = decision.get("action")
        
        if action == "process_next_task":
            # 다음 작업 처리는 task_executor에서 자동 처리됨
            pass
        elif action == "communicate":
            target = decision.get("target")
            message = decision.get("message")
            if target and message:
                await self.send_message_to_agent(target, message)
    
    async def _self_reflection(self):
        """자기 성찰 및 개선"""
        # 성능이 저하되면 학습률 조정
        if self.performance_metrics["success_rate"] < 0.7:
            self.learning_rate = min(self.learning_rate * 1.1, 0.5)
        elif self.performance_metrics["success_rate"] > 0.9:
            self.learning_rate = max(self.learning_rate * 0.9, 0.01)
    
    async def _learn_from_experience(self):
        """경험으로부터 학습"""
        if not self.experience_buffer:
            return
            
        # 성공 패턴 분석
        successful_experiences = [exp for exp in self.experience_buffer if exp.get("success")]
        failed_experiences = [exp for exp in self.experience_buffer if not exp.get("success")]
        
        # 패턴 학습 (간단한 규칙 기반)
        if successful_experiences:
            avg_success_duration = sum([exp["duration"].total_seconds() 
                                      for exp in successful_experiences]) / len(successful_experiences)
            self.long_term_memory["avg_task_duration"] = avg_success_duration
        
        # 실패 패턴 학습
        if failed_experiences:
            common_failures = {}
            for exp in failed_experiences:
                error_type = type(exp.get("error", "Unknown")).__name__
                common_failures[error_type] = common_failures.get(error_type, 0) + 1
            self.long_term_memory["common_failures"] = common_failures
        
        # 경험 버퍼 정리 (최근 50개만 유지)
        self.experience_buffer = self.experience_buffer[-50:]
        self.performance_metrics["learning_iterations"] += 1
        
        logging.info(f"Agent {self.name} completed learning iteration #{self.performance_metrics['learning_iterations']}")
    
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        total_tasks = self.performance_metrics["tasks_completed"] + self.performance_metrics["tasks_failed"]
        if total_tasks > 0:
            self.performance_metrics["success_rate"] = self.performance_metrics["tasks_completed"] / total_tasks
        # else: 작업이 없는 경우 초기값 1.0 유지 (변경하지 않음)
        
        # 최근 경험에서 응답 시간 계산
        if self.experience_buffer:
            recent_durations = [exp["duration"].total_seconds() for exp in self.experience_buffer[-10:]]
            self.performance_metrics["avg_response_time"] = sum(recent_durations) / len(recent_durations)
        
        logging.debug(f"Agent {self.name} metrics updated: success_rate={self.performance_metrics['success_rate']:.2f}, total_tasks={total_tasks}")
    
    async def add_task(self, task: Task):
        """작업 추가"""
        await self.task_queue.put((task.priority, task))
        logging.info(f"Task {task.name} added to agent {self.name}")
    
    async def send_message_to_agent(self, target_agent: str, content: Dict[str, Any], 
                                  message_type: str = "info", priority: int = 0):
        """다른 Agent에게 메시지 전송"""
        message = Message(
            sender=self.name,
            receiver=target_agent,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        # 실제 구현에서는 Agent Registry나 Message Bus를 통해 전송
        logging.info(f"Message sent from {self.name} to {target_agent}: {message_type}")
        return message
    
    async def receive_message(self, message: Message):
        """메시지 수신"""
        await self.message_queue.put(message)
    
    async def _send_message(self, message: Message):
        """메시지 전송 (실제 전송 로직)"""
        # 여기서는 로깅만 수행, 실제로는 Message Bus나 Agent Registry 사용
        logging.info(f"Agent {self.name} sending message to {message.receiver}")
    
    def get_status(self) -> Dict[str, Any]:
        """Agent 상태 조회"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize(),
            "performance_metrics": self.performance_metrics,
            "memory_stats": {
                "short_term": len(self.short_term_memory),
                "long_term_keys": len(self.long_term_memory),
                "working": len(self.working_memory)
            }
        }
    
    def add_capability(self, capability: str):
        """능력 추가"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            logging.info(f"Agent {self.name} acquired new capability: {capability}")
    
    def register_communication_handler(self, message_type: str, handler: Callable):
        """통신 핸들러 등록"""
        self.communication_handlers[message_type] = handler