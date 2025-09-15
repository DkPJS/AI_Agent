"""
Agent 기반 RAG API 엔드포인트
각 Agent가 RAG 컴포넌트를 활용할 수 있도록 하는 전용 API
"""

from fastapi import APIRouter, Depends, HTTPException, Body, status
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime

from app.agents.agent_system import IntelligentAgentSystem
from app.agents.coordinators.rag_coordinator import RAGCoordinator
from app.agents.specialized.enhanced_search_agent import EnhancedSearchAgent
from app.agents.specialized.qa_agent import QAAgent

router = APIRouter(prefix="/api/agent-rag", tags=["agent-rag"])

# 전역 Agent 시스템 인스턴스
agent_system = IntelligentAgentSystem()

class AgentRAGRequest(BaseModel):
    """Agent RAG 요청 모델"""
    query: str
    agent_type: Optional[str] = "auto"  # auto, search, qa, rag_coordinator
    optimization_level: Optional[str] = "standard"  # basic, standard, advanced
    user_context: Optional[Dict[str, Any]] = {}

class AgentRAGResponse(BaseModel):
    """Agent RAG 응답 모델"""
    success: bool
    answer: str
    sources: List[Dict[str, Any]]
    agent_used: str
    performance_metrics: Dict[str, Any]
    optimization_applied: bool

class MultiAgentRequest(BaseModel):
    """다중 Agent 협업 요청"""
    query: str
    agents: List[str]  # 사용할 Agent 목록
    collaboration_mode: Optional[str] = "sequential"  # sequential, parallel, hierarchical

@router.post("/query", response_model=AgentRAGResponse)
async def agent_rag_query(request: AgentRAGRequest):
    """Agent 기반 지능형 RAG 쿼리 처리"""
    try:
        # Agent 시스템 초기화 확인
        if not agent_system.is_initialized:
            await agent_system.initialize()
        
        start_time = datetime.now()
        
        # Agent 타입별 처리
        if request.agent_type == "rag_coordinator":
            result = await _process_with_rag_coordinator(request)
        elif request.agent_type == "search":
            result = await _process_with_search_agent(request)
        elif request.agent_type == "qa":
            result = await _process_with_qa_agent(request)
        else:
            # 자동 Agent 선택
            result = await _process_with_auto_agent_selection(request)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 메트릭 추가
        if result.get("success"):
            result["performance_metrics"] = {
                "execution_time": execution_time,
                "agent_selection_time": 0.1,  # 실제 측정 필요
                "optimization_overhead": 0.05
            }
        
        return result
        
    except Exception as e:
        logging.error(f"Agent RAG query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent RAG 처리 중 오류 발생: {str(e)}"
        )

async def _process_with_rag_coordinator(request: AgentRAGRequest) -> Dict[str, Any]:
    """RAGCoordinator를 통한 처리"""
    rag_coordinator = agent_system.agents.get("RAGCoordinator")
    if not rag_coordinator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAGCoordinator가 사용 불가능합니다."
        )
    
    result = await rag_coordinator._process_intelligent_rag_query({
        "query": request.query,
        "context": request.user_context,
        "optimization_level": request.optimization_level
    })
    
    return {
        "success": result.get("success", False),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "agent_used": "RAGCoordinator",
        "optimization_applied": True,
        "metadata": result.get("metadata", {})
    }

async def _process_with_search_agent(request: AgentRAGRequest) -> Dict[str, Any]:
    """SearchAgent를 통한 처리"""
    search_agent = agent_system.agents.get("SearchAgent")
    if not search_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SearchAgent가 사용 불가능합니다."
        )
    
    # 검색만 수행
    search_result = await search_agent._handle_search_request({
        "query": request.query,
        "limit": 5,
        "type": "advanced" if request.optimization_level == "advanced" else "auto"
    })
    
    if not search_result.get("success"):
        return {
            "success": False,
            "answer": "검색 결과를 찾을 수 없습니다.",
            "sources": [],
            "agent_used": "SearchAgent",
            "optimization_applied": False
        }
    
    # 간단한 답변 생성 (검색 컨텍스트 기반)
    context = search_result.get("context", "")
    sources = search_result.get("sources", [])
    
    answer = f"검색된 정보에 따르면, {context[:200]}..." if context else "관련 정보를 찾았습니다."
    
    return {
        "success": True,
        "answer": answer,
        "sources": sources,
        "agent_used": "SearchAgent",
        "optimization_applied": request.optimization_level in ["standard", "advanced"],
        "metadata": search_result.get("metadata", {})
    }

async def _process_with_qa_agent(request: AgentRAGRequest) -> Dict[str, Any]:
    """QAAgent를 통한 처리"""
    qa_agent = agent_system.agents.get("QAAgent")
    if not qa_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QAAgent가 사용 불가능합니다."
        )
    
    # QA 처리
    qa_result = await qa_agent._handle_question({
        "question": request.query,
        "context": "",  # 컨텍스트는 QA Agent 내부에서 검색
        "session_id": "api_session"
    })
    
    return {
        "success": qa_result.get("success", False),
        "answer": qa_result.get("answer", ""),
        "sources": qa_result.get("sources", []),
        "agent_used": "QAAgent",
        "optimization_applied": True,
        "metadata": qa_result.get("metadata", {})
    }

async def _process_with_auto_agent_selection(request: AgentRAGRequest) -> Dict[str, Any]:
    """자동 Agent 선택 및 처리"""
    query = request.query
    
    # 쿼리 분석으로 최적 Agent 선택
    optimal_agent = _select_optimal_agent(query, request.optimization_level)
    
    if optimal_agent == "rag_coordinator":
        return await _process_with_rag_coordinator(request)
    elif optimal_agent == "search":
        return await _process_with_search_agent(request)
    elif optimal_agent == "qa":
        return await _process_with_qa_agent(request)
    else:
        # 기본 시스템 처리
        result = await agent_system.process_query(query, request.user_context)
        
        return {
            "success": result.get("success", False),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "agent_used": "IntelligentAgentSystem",
            "optimization_applied": True,
            "metadata": result
        }

def _select_optimal_agent(query: str, optimization_level: str) -> str:
    """최적 Agent 선택 로직"""
    query_lower = query.lower()
    
    # 검색 중심 질문
    if any(keyword in query_lower for keyword in ["찾아", "검색", "어디", "무엇"]):
        if optimization_level == "advanced":
            return "rag_coordinator"
        return "search"
    
    # 분석/추론 중심 질문
    elif any(keyword in query_lower for keyword in ["왜", "어떻게", "분석", "비교"]):
        return "qa"
    
    # 복합 질문
    elif len(query.split()) > 10:
        return "rag_coordinator"
    
    # 기본값
    return "qa"

@router.post("/multi-agent")
async def multi_agent_collaboration(request: MultiAgentRequest):
    """다중 Agent 협업 처리"""
    try:
        if not agent_system.is_initialized:
            await agent_system.initialize()
        
        start_time = datetime.now()
        
        if request.collaboration_mode == "parallel":
            result = await _parallel_agent_collaboration(request)
        elif request.collaboration_mode == "hierarchical":
            result = await _hierarchical_agent_collaboration(request)
        else:
            result = await _sequential_agent_collaboration(request)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        result["collaboration_metrics"] = {
            "execution_time": execution_time,
            "agents_used": request.agents,
            "collaboration_mode": request.collaboration_mode
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Multi-agent collaboration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"다중 Agent 협업 중 오류 발생: {str(e)}"
        )

async def _parallel_agent_collaboration(request: MultiAgentRequest) -> Dict[str, Any]:
    """병렬 Agent 협업"""
    tasks = []
    
    for agent_name in request.agents:
        if agent_name == "search":
            task = _process_with_search_agent(AgentRAGRequest(
                query=request.query,
                agent_type="search"
            ))
        elif agent_name == "qa":
            task = _process_with_qa_agent(AgentRAGRequest(
                query=request.query,
                agent_type="qa"
            ))
        elif agent_name == "rag_coordinator":
            task = _process_with_rag_coordinator(AgentRAGRequest(
                query=request.query,
                agent_type="rag_coordinator"
            ))
        else:
            continue
        
        tasks.append(task)
    
    # 병렬 실행
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 통합
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
    
    if not successful_results:
        return {
            "success": False,
            "answer": "모든 Agent에서 처리 실패",
            "sources": [],
            "collaboration_mode": "parallel"
        }
    
    # 최고 점수 결과 선택
    best_result = max(successful_results, key=lambda x: x.get("confidence", 0))
    
    return {
        **best_result,
        "collaboration_mode": "parallel",
        "agents_participated": len(successful_results)
    }

async def _sequential_agent_collaboration(request: MultiAgentRequest) -> Dict[str, Any]:
    """순차 Agent 협업"""
    accumulated_context = ""
    final_result = None
    
    for agent_name in request.agents:
        agent_request = AgentRAGRequest(
            query=request.query,
            agent_type=agent_name,
            user_context={"previous_context": accumulated_context}
        )
        
        if agent_name == "search":
            result = await _process_with_search_agent(agent_request)
            accumulated_context += f"\n검색 결과: {result.get('answer', '')}"
        elif agent_name == "qa":
            result = await _process_with_qa_agent(agent_request)
            final_result = result
        elif agent_name == "rag_coordinator":
            result = await _process_with_rag_coordinator(agent_request)
            final_result = result
    
    return {
        **(final_result or {}),
        "collaboration_mode": "sequential",
        "context_accumulated": bool(accumulated_context)
    }

async def _hierarchical_agent_collaboration(request: MultiAgentRequest) -> Dict[str, Any]:
    """계층적 Agent 협업"""
    # 1단계: 검색 Agent로 정보 수집
    search_result = await _process_with_search_agent(AgentRAGRequest(
        query=request.query,
        agent_type="search",
        optimization_level="advanced"
    ))
    
    # 2단계: QA Agent로 답변 생성
    qa_result = await _process_with_qa_agent(AgentRAGRequest(
        query=request.query,
        agent_type="qa",
        user_context={"search_context": search_result.get("answer", "")}
    ))
    
    # 3단계: RAG Coordinator로 최종 최적화
    if "rag_coordinator" in request.agents:
        final_result = await _process_with_rag_coordinator(AgentRAGRequest(
            query=request.query,
            agent_type="rag_coordinator",
            optimization_level="advanced"
        ))
    else:
        final_result = qa_result
    
    return {
        **final_result,
        "collaboration_mode": "hierarchical",
        "stages_completed": 3 if "rag_coordinator" in request.agents else 2
    }

@router.get("/agent-status")
async def get_agent_rag_status():
    """Agent RAG 시스템 상태 조회"""
    try:
        if not agent_system.is_initialized:
            return {"status": "not_initialized", "agents": []}
        
        agent_status = {}
        for agent_name, agent in agent_system.agents.items():
            if hasattr(agent, 'get_rag_statistics'):
                stats = agent.get_rag_statistics()
            elif hasattr(agent, 'get_search_statistics'):
                stats = agent.get_search_statistics()
            elif hasattr(agent, 'get_qa_statistics'):
                stats = agent.get_qa_statistics()
            else:
                stats = agent.get_status()
            
            agent_status[agent_name] = {
                "status": stats.get("agent_status", {}).get("state", "unknown"),
                "capabilities": agent.capabilities,
                "rag_integration": True,
                "performance": stats
            }
        
        return {
            "system_status": "running",
            "agents": agent_status,
            "total_agents": len(agent_system.agents),
            "rag_optimization_enabled": True
        }
        
    except Exception as e:
        logging.error(f"Agent status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent 상태 조회 중 오류 발생: {str(e)}"
        )

@router.post("/optimize")
async def optimize_agent_rag_system():
    """Agent RAG 시스템 최적화"""
    try:
        if not agent_system.is_initialized:
            await agent_system.initialize()
        
        optimization_result = await agent_system.optimize_system_performance()
        
        return {
            "success": True,
            "optimization_result": optimization_result,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logging.error(f"System optimization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"시스템 최적화 중 오류 발생: {str(e)}"
        )