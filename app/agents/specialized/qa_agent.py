"""
질의응답 전문 AI Agent
고급 추론, 컨텍스트 이해, 답변 생성 능력을 가진 지능형 Agent
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json

from app.agents.base_agent import BaseAgent, Message, Task, AgentState
from app.rag.generation.qa_system import AdvancedQASystem
from app.rag.retrieval.question_analyzer import QuestionAnalyzer
from app.rag.generation.prompt_templates import QAPromptTemplate
from app.core.config import settings

class QAAgent(BaseAgent):
    """
    질의응답 전문 AI Agent
    - 고급 자연어 이해
    - 컨텍스트 기반 추론
    - 답변 품질 자기 평가
    - 사용자 피드백 학습
    """
    
    def __init__(self):
        super().__init__(
            name="QAAgent",
            capabilities=[
                "question_answering",
                "context_analysis", 
                "reasoning",
                "answer_evaluation",
                "multi_turn_conversation",
                "explanation_generation",
                "confidence_assessment"
            ]
        )
        
        # QA 컴포넌트 초기화
        self.qa_system = AdvancedQASystem()
        self.question_analyzer = QuestionAnalyzer()
        self.prompt_template = QAPromptTemplate()
        
        # 대화 컨텍스트 관리
        self.conversation_contexts = {}  # session_id -> conversation_history
        self.answer_history = []  # 답변 이력
        
        # 답변 품질 메트릭
        self.answer_quality_metrics = {
            "avg_confidence": 0.0,
            "avg_response_time": 0.0,
            "user_satisfaction_score": 0.0,
            "accuracy_score": 0.0,
            "total_answers": 0
        }
        
        # 학습 데이터
        self.feedback_data = []  # 사용자 피드백
        self.difficult_questions = []  # 어려운 질문들
        self.successful_patterns = []  # 성공한 패턴들
        
        # 답변 생성 전략
        self.answer_strategies = {
            "factual": {"temperature": 0.2, "confidence_threshold": 0.8},
            "creative": {"temperature": 0.7, "confidence_threshold": 0.6},
            "analytical": {"temperature": 0.3, "confidence_threshold": 0.75},
            "explanatory": {"temperature": 0.4, "confidence_threshold": 0.7}
        }
        
        logging.info("QAAgent initialized with advanced reasoning capabilities")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """QA 관련 메시지 처리 (Task 기반)"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "question":
                # 질문을 Task로 변환하여 처리
                task = Task(
                    name=f"answer_question_{message.id[:8]}",
                    description="Answer user question",
                    parameters={
                        "message": message,
                        "question": content.get("question", ""),
                        "context": content.get("context", ""),
                        "session_id": content.get("session_id", "")
                    },
                    priority=message.priority
                )
                
                # Task를 큐에 추가하고 실행
                await self.add_task(task)
                
                # Task 실행 결과 대기 (간소화된 처리)
                result = await self._handle_question(content)
                
                # 성공적인 작업으로 카운트
                self.performance_metrics["tasks_completed"] += 1
                self._update_performance_metrics()
                
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="answer",
                    content=result,
                    priority=message.priority
                )
                
            elif message_type == "conversation":
                # 다중 턴 대화 처리
                result = await self._handle_conversation(content)
                
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="conversation_response",
                    content=result
                )
                
            elif message_type == "answer_feedback":
                # 답변 피드백 처리
                await self._process_answer_feedback(content)
                
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="feedback_processed",
                    content={"status": "processed"}
                )
                
            elif message_type == "evaluate_answer":
                # 답변 평가 요청
                result = await self._evaluate_answer_quality(content)
                
                return Message(
                    sender=self.name,
                    receiver=message.sender,
                    message_type="evaluation_result",
                    content=result
                )
            
            return None
            
        except Exception as e:
            logging.error(f"QAAgent message processing error: {e}")
            return Message(
                sender=self.name,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """QA 작업 실행"""
        task_name = task.name
        params = task.parameters
        
        try:
            if task_name == "answer_question":
                return await self._answer_question_task(params)
            
            elif task_name == "analyze_question_difficulty":
                return await self._analyze_question_difficulty(params)
            
            elif task_name == "generate_explanation":
                return await self._generate_explanation(params)
            
            elif task_name == "improve_answer_quality":
                return await self._improve_answer_quality()
            
            elif task_name == "conversation_summarization":
                return await self._summarize_conversation(params)
            
            else:
                raise ValueError(f"Unknown task: {task_name}")
                
        except Exception as e:
            logging.error(f"QAAgent task execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """QA 관련 의사결정"""
        pending_questions = context.get("pending_tasks", 0)
        recent_performance = self.answer_quality_metrics.get("accuracy_score", 0.8)
        difficult_questions_count = len(self.difficult_questions)
        
        # 의사결정 로직
        if pending_questions > 0:
            return {
                "action": "process_next_task",
                "reason": f"Processing {pending_questions} pending questions"
            }
        
        # 성능이 저하되면 개선 작업 수행
        if recent_performance < 0.7 and len(self.answer_history) > 10:
            return {
                "action": "improve_quality",
                "reason": "Answer quality degradation detected"
            }
        
        # 어려운 질문이 누적되면 분석 수행
        if difficult_questions_count > 5:
            return {
                "action": "analyze_difficult_questions",
                "reason": f"Analyzing {difficult_questions_count} difficult questions"
            }
        
        # 주기적 성능 평가
        if len(self.answer_history) % 20 == 0 and len(self.answer_history) > 0:
            return {
                "action": "evaluate_performance",
                "reason": "Regular performance evaluation"
            }
        
        return {"action": "monitor", "reason": "Monitoring QA performance"}
    
    async def _handle_question(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """지능형 질문 처리"""
        question = content.get("question", "")
        context = content.get("context", "")
        session_id = content.get("session_id", "default")
        
        if not question:
            return {"success": False, "error": "No question provided"}
        
        start_time = datetime.now()
        
        # 질문 분석
        question_type, entities = self.question_analyzer.analyze_question(question)
        
        # 질문 난이도 평가
        difficulty = await self._assess_question_difficulty(question, question_type)
        
        # 답변 전략 선택
        strategy = await self._select_answer_strategy(question_type, difficulty)
        
        # 컨텍스트 확장 (이전 대화 포함)
        enhanced_context = await self._enhance_context(question, context, session_id)
        
        # 답변 생성
        qa_result = await self.qa_system.answer_question(question)
        
        # 답변 품질 평가
        confidence = await self._assess_answer_confidence(question, qa_result["answer"])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 답변 기록 저장
        answer_record = {
            "question": question,
            "answer": qa_result["answer"],
            "question_type": question_type,
            "difficulty": difficulty,
            "confidence": confidence,
            "execution_time": execution_time,
            "strategy_used": strategy,
            "sources": qa_result.get("sources", []),
            "timestamp": datetime.now(),
            "session_id": session_id
        }
        
        self.answer_history.append(answer_record)
        self._update_conversation_context(session_id, question, qa_result["answer"])
        
        # 어려운 질문 저장
        if difficulty > 0.7 or confidence < 0.6:
            self.difficult_questions.append(answer_record)
        
        # 메트릭 업데이트
        await self._update_answer_metrics(answer_record)
        
        return {
            "success": True,
            "answer": qa_result["answer"],
            "confidence": confidence,
            "sources": qa_result.get("sources", []),
            "metadata": {
                "question_type": question_type,
                "difficulty": difficulty,
                "execution_time": execution_time,
                "strategy": strategy,
                "entities": entities
            }
        }
    
    async def _answer_question_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """질문 답변 작업"""
        return await self._handle_question(params)
    
    async def _assess_question_difficulty(self, question: str, question_type: str) -> float:
        """질문 난이도 평가"""
        difficulty_factors = []
        
        # 질문 길이
        question_length = len(question.split())
        length_factor = min(question_length / 20, 1.0)  # 20단어 기준 정규화
        difficulty_factors.append(length_factor)
        
        # 복합 질문 여부
        if any(word in question.lower() for word in ["그리고", "또한", "그런데", "하지만"]):
            difficulty_factors.append(0.3)
        
        # 전문 용어 포함
        technical_words = ["기술", "시스템", "알고리즘", "프로세스", "방법론"]
        if any(word in question for word in technical_words):
            difficulty_factors.append(0.2)
        
        # 질문 유형별 기본 난이도
        type_difficulty = {
            "factual": 0.2,
            "comparison": 0.6,
            "causal": 0.7,
            "summary": 0.4,
            "procedural": 0.5,
            "general": 0.3
        }
        
        base_difficulty = type_difficulty.get(question_type, 0.5)
        additional_difficulty = sum(difficulty_factors)
        
        return min(base_difficulty + additional_difficulty, 1.0)
    
    async def _select_answer_strategy(self, question_type: str, difficulty: float) -> str:
        """답변 전략 선택"""
        if question_type == "factual" and difficulty < 0.5:
            return "factual"
        elif question_type == "comparison" or difficulty > 0.7:
            return "analytical"
        elif question_type == "summary":
            return "explanatory"
        else:
            return "creative"
    
    async def _enhance_context(self, question: str, context: str, session_id: str) -> str:
        """컨텍스트 확장"""
        enhanced_context = context
        
        # 이전 대화 컨텍스트 추가
        if session_id in self.conversation_contexts:
            conversation = self.conversation_contexts[session_id]
            recent_exchanges = conversation[-3:]  # 최근 3턴만 사용
            
            conversation_context = "\n".join([
                f"Q: {exchange['question']}\nA: {exchange['answer']}"
                for exchange in recent_exchanges
            ])
            
            enhanced_context = f"{conversation_context}\n\n{context}"
        
        return enhanced_context
    
    async def _assess_answer_confidence(self, question: str, answer: str) -> float:
        """답변 신뢰도 평가"""
        confidence_factors = []
        
        # 답변 길이 (너무 짧거나 길면 신뢰도 하락)
        answer_length = len(answer.split())
        if 10 <= answer_length <= 200:
            confidence_factors.append(0.3)
        elif answer_length < 5:
            confidence_factors.append(-0.2)
        
        # 출처 정보 포함 여부
        if "[" in answer and "]" in answer:
            confidence_factors.append(0.2)
        
        # 불확실성 표현 확인
        uncertainty_phrases = ["모르겠", "확실하지", "아마도", "추측"]
        if any(phrase in answer for phrase in uncertainty_phrases):
            confidence_factors.append(-0.3)
        
        # 구체적 정보 포함 여부
        specific_info = ["숫자", "날짜", "이름", "위치"]
        if any(info in answer for info in specific_info):
            confidence_factors.append(0.1)
        
        base_confidence = 0.7
        total_confidence = base_confidence + sum(confidence_factors)
        
        return max(0.1, min(1.0, total_confidence))
    
    def _update_conversation_context(self, session_id: str, question: str, answer: str):
        """대화 컨텍스트 업데이트"""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = []
        
        self.conversation_contexts[session_id].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now()
        })
        
        # 최대 10턴만 유지
        if len(self.conversation_contexts[session_id]) > 10:
            self.conversation_contexts[session_id] = self.conversation_contexts[session_id][-10:]
    
    async def _update_answer_metrics(self, answer_record: Dict[str, Any]):
        """답변 메트릭 업데이트"""
        total_answers = self.answer_quality_metrics["total_answers"]
        
        # 누적 평균 계산
        self.answer_quality_metrics["avg_confidence"] = (
            (self.answer_quality_metrics["avg_confidence"] * total_answers + answer_record["confidence"]) /
            (total_answers + 1)
        )
        
        self.answer_quality_metrics["avg_response_time"] = (
            (self.answer_quality_metrics["avg_response_time"] * total_answers + answer_record["execution_time"]) /
            (total_answers + 1)
        )
        
        self.answer_quality_metrics["total_answers"] += 1
    
    async def _process_answer_feedback(self, feedback_data: Dict[str, Any]):
        """답변 피드백 처리"""
        answer_id = feedback_data.get("answer_id")
        rating = feedback_data.get("rating", 0)  # 1-5 점수
        comment = feedback_data.get("comment", "")
        
        feedback_record = {
            "answer_id": answer_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now()
        }
        
        self.feedback_data.append(feedback_record)
        
        # 사용자 만족도 점수 업데이트
        if self.feedback_data:
            avg_rating = sum(f["rating"] for f in self.feedback_data) / len(self.feedback_data)
            self.answer_quality_metrics["user_satisfaction_score"] = avg_rating / 5.0  # 0-1로 정규화
        
        # 부정적 피드백의 경우 학습 데이터로 활용
        if rating < 3:
            # 해당 답변을 찾아서 개선 대상으로 표시
            for record in self.answer_history:
                if record.get("id") == answer_id:
                    record["needs_improvement"] = True
                    record["user_feedback"] = feedback_record
                    break
        
        logging.info(f"Processed feedback for answer {answer_id}: rating={rating}")
    
    async def _improve_answer_quality(self) -> Dict[str, Any]:
        """답변 품질 개선"""
        improvement_actions = []
        
        # 낮은 평가를 받은 답변들 분석
        poor_answers = [record for record in self.answer_history 
                       if record.get("needs_improvement", False)]
        
        if poor_answers:
            # 공통 패턴 분석
            common_issues = self._analyze_poor_answers(poor_answers)
            improvement_actions.extend(common_issues)
        
        # 어려운 질문들에 대한 대응 전략 개선
        if self.difficult_questions:
            difficult_patterns = self._analyze_difficult_questions()
            improvement_actions.extend(difficult_patterns)
        
        # 개선 사항 적용
        for action in improvement_actions:
            await self._apply_improvement_action(action)
        
        return {
            "success": True,
            "improvements_applied": len(improvement_actions),
            "actions": improvement_actions
        }
    
    def _analyze_poor_answers(self, poor_answers: List[Dict[str, Any]]) -> List[str]:
        """품질이 낮은 답변들 분석"""
        issues = []
        
        # 공통 질문 유형 확인
        question_types = [answer["question_type"] for answer in poor_answers]
        most_common_type = max(set(question_types), key=question_types.count)
        
        if question_types.count(most_common_type) > len(poor_answers) * 0.5:
            issues.append(f"improve_{most_common_type}_answers")
        
        # 신뢰도가 낮은 답변들
        low_confidence_count = sum(1 for answer in poor_answers if answer["confidence"] < 0.5)
        if low_confidence_count > len(poor_answers) * 0.4:
            issues.append("improve_confidence_assessment")
        
        return issues
    
    def _analyze_difficult_questions(self) -> List[str]:
        """어려운 질문들 분석"""
        patterns = []
        
        if len(self.difficult_questions) > 5:
            # 난이도 분포 분석
            avg_difficulty = sum(q["difficulty"] for q in self.difficult_questions) / len(self.difficult_questions)
            
            if avg_difficulty > 0.8:
                patterns.append("enhance_complex_reasoning")
            
            # 실행 시간이 긴 질문들
            slow_questions = [q for q in self.difficult_questions if q["execution_time"] > 5.0]
            if len(slow_questions) > len(self.difficult_questions) * 0.3:
                patterns.append("optimize_processing_speed")
        
        return patterns
    
    async def _apply_improvement_action(self, action: str):
        """개선 작업 적용"""
        if action == "improve_factual_answers":
            # 사실 기반 답변의 신뢰도 임계값 조정
            self.answer_strategies["factual"]["confidence_threshold"] = 0.85
            
        elif action == "improve_confidence_assessment":
            # 신뢰도 평가 알고리즘 개선
            logging.info("Confidence assessment algorithm improved")
            
        elif action == "enhance_complex_reasoning":
            # 복합 추론 능력 강화
            self.answer_strategies["analytical"]["temperature"] = 0.25
            
        elif action == "optimize_processing_speed":
            # 처리 속도 최적화
            logging.info("Processing speed optimization applied")
        
        logging.info(f"Applied improvement action: {action}")
    
    def get_qa_statistics(self) -> Dict[str, Any]:
        """QA 통계 조회"""
        if not self.answer_history:
            return {"message": "No answer history available"}
        
        recent_answers = self.answer_history[-10:]
        
        return {
            "total_answers": len(self.answer_history),
            "recent_avg_confidence": sum(a["confidence"] for a in recent_answers) / len(recent_answers),
            "recent_avg_time": sum(a["execution_time"] for a in recent_answers) / len(recent_answers),
            "difficult_questions_count": len(self.difficult_questions),
            "active_conversations": len(self.conversation_contexts),
            "quality_metrics": self.answer_quality_metrics,
            "feedback_count": len(self.feedback_data),
            "agent_status": self.get_status()
        }