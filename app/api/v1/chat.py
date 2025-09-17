from fastapi import APIRouter, Depends, HTTPException, Body, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Tuple, Dict, Any, Optional, List
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid
from fastapi.responses import StreamingResponse
import json
import asyncio
import logging
import re

from app.infrastructure.database.sql_client import get_db
from app.models.chat import ChatSession, ChatMessage, MessageSource
from app.rag.retrieval.graph_retriever import GraphRetriever
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.rag.retrieval.embedder import DocumentEmbedder
from app.core.config import settings
import httpx

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    message_id: str


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db)
):
    """채팅 메시지 처리 및 응답 생성"""
    try:
        # 1. 세션 처리
        session_id = request.session_id
        if not session_id:
            # 새 세션 생성
            session = ChatSession(id=str(uuid.uuid4()))
            db.add(session)
            db.commit()
            session_id = session.id
        else:
            # 기존 세션 조회
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="세션을 찾을 수 없습니다."
                )
        
        # 2. 사용자 메시지 저장
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=request.message
        )
        db.add(user_message)
        db.commit()
        
        # 3. 문서 검색
        retriever = GraphRetriever()
        context, sources = await retriever.retrieve(request.message)
        
        if not context:
            answer = "죄송합니다. 질문에 관련된 문서를 찾을 수 없습니다."
        else:
            # 4. 프롬프트 구성
            prompt = f"""다음 정보를 바탕으로 질문에 답변해주세요. 정보에 관련된 내용이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요.

### 문서 정보:
{context}

### 질문:
{request.message}

### 답변:
"""
            
            # 5. LLM API 호출
            try:
                payload = {
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": False
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        settings.LLM_API_URL,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("response", "")
                    else:
                        raise Exception(f"LLM API 오류: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"LLM 호출 중 오류 발생: {e}")
                answer = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
        
        # 6. 응답 메시지 저장
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=answer
        )
        db.add(assistant_message)
        
        # 7. 소스 정보 저장
        for source in sources:
            message_source = MessageSource(
                id=str(uuid.uuid4()),
                message_id=assistant_message.id,
                document_id=source.get("document_id"),
                chunk_id=source.get("chunk_id"),
                relevance_score=int(source.get("relevance", 0.5) * 100)
            )
            db.add(message_source)
        
        db.commit()
        
        # 8. 응답 반환
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "message_id": assistant_message.id
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"채팅 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/sessions")
async def list_chat_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """채팅 세션 목록 조회"""
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).offset(skip).limit(limit).all()
        return {"sessions": [session.to_dict() for session in sessions]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """특정 채팅 세션 상세 정보 조회"""
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없습니다."
            )
        
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
        
        return {
            "session": session.to_dict(),
            "messages": [message.to_dict() for message in messages]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """채팅 세션 삭제"""
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없습니다."
            )
        
        db.delete(session)
        db.commit()
        
        return {"message": "세션이 성공적으로 삭제되었습니다."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/generate-answer")
async def generate_answer(request: Request):
    """질문에 답변 생성 (스트리밍)"""
    try:
        data = await request.json()
        question = data.get("question", data.get("query", ""))
        context = data.get("context", "")
        sources = data.get("sources", [])
        
        if not question:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "질문이 필요합니다."}
            )
        
        # LLM 호출용 프롬프트 생성
        prompt = create_prompt(question, context, sources)
        logging.debug(f"생성된 프롬프트: {prompt}")
        
        # LLM 호출 및 응답 스트리밍
        async def generate():
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "system": "너는 한국어로만 대답하는 AI 비서입니다. 답변은 항상 한국어로만 작성하세요."
                }
                
                logging.debug(f"LLM API 호출: {settings.LLM_API_URL}, 모델: {settings.LLM_MODEL}")
                
                try:
                    async with client.stream("POST", settings.LLM_API_URL, json=payload) as response:
                        logging.debug(f"LLM API 응답 상태 코드: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_text = await response.aread()
                            logging.error(f"LLM API 오류: {response.status_code} - {error_text}")
                            yield f"data: {json.dumps({'token': '오류가 발생했습니다. 응답 생성에 실패했습니다.'})}"
                            yield f"data: {json.dumps({'status': 'completed'})}"
                            yield f"data: [DONE]"
                            return
                        
                        buffer = ""
                        async for chunk in response.aiter_text():
                            logging.debug(f"LLM 응답 청크: {chunk}")
                            if chunk.strip():
                                try:
                                    data = json.loads(chunk)
                                    if "response" in data:
                                        token = data["response"]
                                        buffer += token
                                        yield f"data: {json.dumps({'token': token})}"
                                except json.JSONDecodeError as e:
                                    logging.error(f"JSON 파싱 오류: {e}, 원본 청크: {chunk}")
                                    continue
                        
                        # 답변이 없으면 에러 메시지 전송
                        if not buffer:
                            logging.warning("LLM에서 빈 응답 반환")
                            yield f"data: {json.dumps({'token': '죄송합니다. 답변을 생성할 수 없습니다.'})}"
                except Exception as e:
                    logging.error(f"LLM API 스트림 처리 중 예외 발생: {e}")
                    yield f"data: {json.dumps({'token': f'오류가 발생했습니다: {str(e)}'})}"
            
            # 완료 이벤트 먼저 전송
            yield f"data: {json.dumps({'status': 'completed'})}"
            # 그 다음 DONE 메시지 전송
            yield f"data: [DONE]"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logging.error(f"답변 생성 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
        )

def create_prompt(question: str, context: str, sources: List[Dict[str, Any]]) -> str:
    """최적화된 LLM 호출용 프롬프트 생성"""
    # 소스 정보 간단히 추출
    source_files = []
    for source in sources:
        filename = source.get("filename", "문서")
        content = source.get("content", "")
        page_match = re.search(r'\[페이지\s*(\d+)\]', content)
        if page_match:
            source_files.append(f"{filename}(p.{page_match.group(1)})")
        else:
            source_files.append(filename)

    # 중복 제거
    unique_sources = list(dict.fromkeys(source_files))

    # 간결하고 효과적인 프롬프트
    prompt = f"""질문에 대해 제공된 문서 내용을 바탕으로 정확하고 유용한 답변을 제공하세요.

질문: {question}

관련 문서 내용:
{context}

답변 가이드라인:
- 문서 내용에 기반한 정확한 정보 제공
- 구체적이고 실용적인 답변 구성
- 문서에 없는 내용은 "관련 정보를 찾을 수 없습니다"로 명시
- 답변 마지막에 "[출처: {', '.join(unique_sources[:3])}]" 형식으로 출처 표시

답변:"""

    return prompt