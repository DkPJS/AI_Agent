from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import os
import uvicorn
import json
import asyncio

from app.core.config import settings
from app.infrastructure.database.sql_client import init_db
from app.api.v1.documents import router as documents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.qa import router as qa_router
from app.api.v1.synonyms import router as synonyms_router
from app.agents.agent_system import IntelligentAgentSystem
from app.utils.gpu_utils import get_gpu_info, check_gpu_health, setup_gpu_device
from app.utils.model_manager import get_model_memory_info

# 지능형 Agent 시스템 초기화
agent_system = IntelligentAgentSystem()

# FastAPI 앱 생성
app = FastAPI(
    title="Intelligent AI Agent RAG System",
    description="Multi-Agent 협업 기반 지능형 문서 검색 및 질의응답 시스템",
    version="2.0.0-agent"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 디렉토리 설정
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API 라우터 등록
app.include_router(documents_router, tags=["documents"])
app.include_router(chat_router, tags=["chat"])
app.include_router(qa_router, tags=["qa"])
app.include_router(synonyms_router, tags=["synonyms"])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU information"""
    try:
        gpu_info = get_gpu_info()
        gpu_healthy = check_gpu_health()
        model_info = get_model_memory_info()

        return {
            "status": "healthy",
            "service": "intelligent-rag-chatbot",
            "version": "2.0.0-agent",
            "gpu_available": gpu_info["cuda_available"],
            "gpu_healthy": gpu_healthy,
            "gpu_info": {
                "device_name": gpu_info["device_name"],
                "memory_total": f"{gpu_info['memory_total']:.1f}GB",
                "memory_allocated": f"{gpu_info['memory_allocated']:.3f}GB",
                "memory_reserved": f"{gpu_info['memory_reserved']:.3f}GB"
            } if gpu_info["cuda_available"] else None,
            "model_manager": model_info,
            "agent_system": "active"
        }
    except Exception as e:
        return {
            "status": "healthy",
            "service": "intelligent-rag-chatbot",
            "version": "2.0.0-agent",
            "gpu_available": False,
            "gpu_error": str(e),
            "agent_system": "active"
        }

@app.post("/generate-answer")
async def generate_answer_endpoint(request: Request):
    """AI Agent 기반 지능형 답변 생성"""
    # JSON 데이터 파싱
    data = await request.json()
    question = data.get("question", "")
    user_context = data.get("context", {})
    
    # Agent 시스템을 통한 지능형 처리
    result = await agent_system.process_query(question, user_context)
    
    # StreamingResponse로 변환하여 프론트엔드와 호환되게 응답
    async def generate():
        # Agent 시스템 상태 알림
        yield f"data: {json.dumps({'status': 'agent_initialization', 'message': 'AI Agent 시스템 활성화'})}\n\n"
        
        # 질문 분석 알림
        metadata = result.get("metadata", {})
        question_type = metadata.get("question_type", "general")
        yield f"data: {json.dumps({'status': 'intelligent_analysis', 'message': f'지능형 분석 완료: {question_type}'})}\n\n"
        
        # 다중 Agent 협업 상태
        agents_used = result.get("system_info", {}).get("agents_used", [])
        if len(agents_used) > 1:
            yield f"data: {json.dumps({'status': 'multi_agent_collaboration', 'message': f'{len(agents_used)}개 Agent 협업 중'})}\n\n"
        
        # 검색 정보 전송
        sources_count = len(result.get("sources", []))
        yield f"data: {json.dumps({'status': 'knowledge_search', 'message': f'고급 검색: {sources_count}개 관련 문서 발견'})}\n\n"
        
        # 응답 전송
        answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")
        yield f"data: {json.dumps({'token': answer})}\n\n"
        
        # 품질 점수 및 신뢰도
        confidence = result.get("confidence", 0.0)
        quality_score = result.get("quality_score", 0.0)
        yield f"data: {json.dumps({'status': 'quality_assessment', 'message': f'답변 품질: {quality_score:.2f}, 신뢰도: {confidence:.2f}'})}\n\n"
        
        # 완료 상태 전송
        execution_time = result.get("system_info", {}).get("execution_time", 0.0)
        yield f"data: {json.dumps({'status': 'completed', 'execution_time': execution_time})}\n\n"
        yield f"data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/get_map_url")
async def get_map_url_endpoint():
    # 간단한 지도 URL 반환
    map_url = "https://map.naver.com/p/search/관련장소"
    return {"map_url": map_url}

# Agent 시스템 API 엔드포인트
@app.get("/agent-status")
async def get_agent_status():
    """Agent 시스템 상태 조회"""
    return await agent_system.get_agent_insights()

@app.get("/system-metrics")
async def get_system_metrics():
    """시스템 메트릭 조회"""
    return agent_system.get_system_metrics()

@app.post("/optimize-system")
async def optimize_system():
    """시스템 성능 최적화 실행"""
    return await agent_system.optimize_system_performance()

@app.post("/agent-task")
async def add_agent_task(request: Request):
    """전문화된 Agent 작업 추가"""
    data = await request.json()
    task_name = data.get("task_name", "")
    task_params = data.get("parameters", {})
    priority = data.get("priority", 0)
    
    task_id = await agent_system.add_specialized_task(task_name, task_params, priority)
    return {"task_id": task_id, "status": "added"}

@app.on_event("startup")
async def startup_event():
    # GPU 초기화
    try:
        device = setup_gpu_device()
        gpu_info = get_gpu_info()
        print(f"GPU 설정 완료: {device}")
        if gpu_info["cuda_available"]:
            print(f"GPU 메모리: {gpu_info['memory_total']:.1f}GB 사용 가능")
    except Exception as e:
        print(f"GPU 초기화 중 오류: {e}")

    # 데이터베이스 초기화
    init_db()

    # 업로드 디렉토리 생성
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # AI Agent 시스템 초기화
    try:
        await agent_system.initialize()
        print("AI Agent 시스템이 성공적으로 초기화되었습니다!")
    except Exception as e:
        print(f"Agent 시스템 초기화 실패: {e}")

    print(f"지능형 Agent 서버가 시작되었습니다. (환경: {'개발' if settings.DEBUG else '운영'})")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # GPU 메모리 부족 방지를 위해 reload 비활성화
        workers=1      # 단일 워커로 메모리 사용량 제한
    )