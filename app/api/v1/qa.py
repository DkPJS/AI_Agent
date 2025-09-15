"""Question-Answering API endpoints.

This module provides REST API endpoints for the RAG-based question-answering
system, including chat session management, document search, and file upload.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List, Union

import httpx
from fastapi import APIRouter, Depends, HTTPException, Body, status, File, Form, UploadFile
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.database.sql_client import get_db
from app.models.chat import ChatSession, ChatMessage, MessageSource
from app.rag.generation.qa_system import AdvancedQASystem
from app.rag.retrieval.graph_retriever import GraphRetriever
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.rag.retrieval.embedder import DocumentEmbedder
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize router with comprehensive configuration
router = APIRouter(
    prefix="/api/qa",
    tags=["qa"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"}
    }
)


# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    """Request model for question submission.

    Attributes:
        question: The user's question text
        session_id: Optional existing chat session ID
        context: Optional additional context for the question
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to be answered"
    )
    session_id: Optional[str] = Field(
        None,
        description="Existing chat session ID"
    )
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional context for the question"
    )

    @validator('question')
    def validate_question(cls, v: str) -> str:
        """Validate question content.

        Args:
            v: The question string to validate

        Returns:
            The validated question string

        Raises:
            ValueError: If question is invalid
        """
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class QuestionResponse(BaseModel):
    """Response model for question answers.

    Attributes:
        answer: The generated answer text
        sources: List of source documents used
        question_type: Classified type of the question
        session_id: Chat session identifier
        message_id: Message identifier
        confidence: Confidence score of the answer
    """
    answer: str
    sources: List[Dict[str, Any]]
    question_type: str
    session_id: str
    message_id: str
    confidence: Optional[float] = None


class SearchResult(BaseModel):
    """Model for individual search results.

    Attributes:
        content: The text content of the result
        document_id: Source document identifier
        filename: Original filename
        chunk_id: Document chunk identifier
        relevance: Relevance score (0-1)
    """
    content: str
    document_id: str
    filename: str
    chunk_id: str
    relevance: float = Field(ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    """Response model for document-based answers.

    Attributes:
        answer: The generated answer
        context: Context used for answer generation
        sources: Source documents and chunks
        processing_time: Time taken to process the question
    """
    answer: str
    context: str
    sources: List[Dict[str, Any]]
    processing_time: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standardized error response model.

    Attributes:
        error: Error message
        detail: Detailed error information
        error_code: Application-specific error code
    """
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None

@router.post("/", response_model=QuestionResponse)
async def answer_question(
    request: QuestionRequest = Body(...),
    db: Session = Depends(get_db)
) -> QuestionResponse:
    """Process question using advanced RAG system.

    This endpoint handles question submission, retrieves relevant context
    from documents, and generates comprehensive answers using the RAG pipeline.

    Args:
        request: Question request containing the question and optional session info
        db: Database session dependency

    Returns:
        Comprehensive answer with sources and metadata

    Raises:
        HTTPException: If question processing fails
    """
    try:
        # Handle session management
        session_id = await _handle_session(request.session_id, db)

        # Store user message
        user_message = await _store_user_message(session_id, request.question, db)

        # Generate answer using QA system
        qa_system = AdvancedQASystem()
        result = await qa_system.answer_question(request.question)

        # Store assistant response and sources
        assistant_message = await _store_assistant_message(
            session_id, result["answer"], db
        )
        await _store_message_sources(assistant_message.id, result["sources"], db)

        db.commit()
        logger.info(f"Successfully processed question in session {session_id}")

        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            question_type=result.get("question_type", "general"),
            session_id=session_id,
            message_id=assistant_message.id,
            confidence=result.get("confidence")
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error during question processing: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )
    except Exception as e:
        logger.error(f"Unexpected error during question processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your question"
        )


async def _handle_session(session_id: Optional[str], db: Session) -> str:
    """Handle chat session creation or retrieval.

    Args:
        session_id: Optional existing session ID
        db: Database session

    Returns:
        Session ID (new or existing)

    Raises:
        HTTPException: If existing session not found
    """
    if not session_id:
        # Create new session
        session = ChatSession(id=str(uuid.uuid4()))
        db.add(session)
        db.flush()  # Get the ID without committing
        return session.id
    else:
        # Validate existing session
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        return session_id


async def _store_user_message(session_id: str, question: str, db: Session) -> ChatMessage:
    """Store user message in database.

    Args:
        session_id: Chat session ID
        question: User question text
        db: Database session

    Returns:
        Created ChatMessage instance
    """
    user_message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=question
    )
    db.add(user_message)
    db.flush()
    return user_message


async def _store_assistant_message(session_id: str, answer: str, db: Session) -> ChatMessage:
    """Store assistant message in database.

    Args:
        session_id: Chat session ID
        answer: Generated answer text
        db: Database session

    Returns:
        Created ChatMessage instance
    """
    assistant_message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=answer
    )
    db.add(assistant_message)
    db.flush()
    return assistant_message


async def _store_message_sources(
    message_id: str,
    sources: List[Dict[str, Any]],
    db: Session
) -> None:
    """Store message source references in database.

    Args:
        message_id: Assistant message ID
        sources: List of source document information
        db: Database session
    """
    for source in sources:
        if source.get("document_id") and source.get("chunk_id"):
            message_source = MessageSource(
                id=str(uuid.uuid4()),
                message_id=message_id,
                document_id=source.get("document_id"),
                chunk_id=source.get("chunk_id"),
                relevance_score=int(source.get("relevance", 0.5) * 100)
            )
            db.add(message_source)
    db.flush()

@router.get("/sessions/{session_id}")
async def get_qa_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """QA 세션 및 메시지 조회"""
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

@router.get("/sessions")
async def list_qa_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """QA 세션 목록 조회"""
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).offset(skip).limit(limit).all()
        return {"sessions": [session.to_dict() for session in sessions]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/ask", response_model=AnswerResponse)
async def answer_question_graph(question: QuestionRequest):
    """질문에 대한 답변 생성"""
    try:
        # 1. 문서 검색
        retriever = GraphRetriever()
        context, sources = await retriever.retrieve(question.question, limit=4)
        
        if not context:
            return {
                "answer": "질문에 관련된 문서를 찾을 수 없습니다.",
                "context": "",
                "sources": []
            }
        
        # 2. LLM을 통한 답변 생성
        prompt = f"""다음 정보를 바탕으로 질문에 답변해주세요. 정보에 관련된 내용이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요.

### 문서 정보:
{context}

### 질문:
{question.question}

### 답변:
"""
        
        response = await generate_answer(prompt)
        
        # 3. 응답 반환
        return {
            "answer": response,
            "context": context,
            "sources": sources
        }
        
    except Exception as e:
        logging.error(f"질문 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"질문 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/search")
async def search_documents(query: str = Body(..., embed=True), limit: int = 5):
    """문서 검색"""
    try:
        # 1. 벡터 검색
        embedder = DocumentEmbedder()
        vector_results = await embedder.search_similar_chunks(query, limit=limit)
        
        # 2. 키워드 검색 (Neo4j 풀텍스트 인덱스)
        with Neo4jClient() as neo4j_client:
            keyword_results = neo4j_client.search_related_chunks(query, limit=limit)
        
        # 결과 합치기 (중복 제거)
        combined_results = []
        seen_chunks = set()
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                combined_results.append(result)
        
        # 키워드 검색 결과 추가
        for result in keyword_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                combined_results.append(result)
        
        # 결과가 없는 경우
        if not combined_results:
            return {"results": [], "message": "검색 결과가 없습니다."}
        
        # 상위 N개 결과만 반환
        return {
            "results": combined_results[:limit],
            "message": f"{len(combined_results[:limit])}개의 결과를 찾았습니다."
        }
        
    except Exception as e:
        logging.error(f"문서 검색 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 검색 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/upload")
async def upload_document(
    file: Optional[UploadFile] = File(None),  # None이 허용됨
    description: str = Form(""),
    auto_detect_synonyms: str = Form("false"),
    db: Session = Depends(get_db)
):
    # 파일 필드 명시적 검증
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일이 제공되지 않았습니다."
        )
    
    # 나머지 코드...

async def generate_answer(prompt: str) -> str:
    """LLM API를 통한 답변 생성"""
    try:
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                settings.LLM_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logging.error(f"LLM API 오류: {response.status_code}, {response.text}")
                return "죄송합니다. 답변을 생성하는 동안 오류가 발생했습니다."
                
    except Exception as e:
        logging.error(f"LLM 호출 중 오류 발생: {e}")
        return "죄송합니다. 답변을 생성하는 동안 오류가 발생했습니다."