"""
Main FastAPI application module for RAG Document Chatbot.
This module provides the main FastAPI application with CORS configuration,
static file serving, and basic endpoints for file upload.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIRECTORY = Path("./uploads")
ALLOWED_FILE_TYPES = {".pdf", ".docx", ".txt", ".hwp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create FastAPI app with enhanced configuration
application = FastAPI(
    title="RAG Document Chatbot",
    description="Advanced RAG-based document Q&A system with graph knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS with more specific settings
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIRECTORY.mkdir(exist_ok=True)
application.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIRECTORY)), name="uploads")

@application.get("/", response_model=Dict[str, str])
async def read_root() -> Dict[str, str]:
    """Root endpoint providing basic API information.

    Returns:
        Dict containing API welcome message and status.
    """
    return {
        "message": "RAG Document Chatbot API",
        "status": "operational",
        "version": "1.0.0"
    }


@application.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring.

    Returns:
        Dict containing health status information.
    """
    try:
        import sys
        sys.path.append('.')
        from app.utils.gpu_utils import get_gpu_info, check_gpu_health
        gpu_info = get_gpu_info()
        gpu_healthy = check_gpu_health()

        return {
            "status": "healthy",
            "service": "rag-chatbot",
            "gpu_available": gpu_info["cuda_available"],
            "gpu_healthy": gpu_healthy,
            "gpu_info": gpu_info
        }
    except Exception as e:
        return {
            "status": "healthy",
            "service": "rag-chatbot",
            "gpu_available": False,
            "gpu_error": str(e)
        }

def _validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size.

    Args:
        file: The uploaded file to validate.

    Raises:
        HTTPException: If file validation fails.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. "
                   f"Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
        )


@application.post("/upload/", response_model=Dict[str, Any])
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """Upload and process document file.

    Args:
        file: The document file to upload.
        description: Optional description of the document.

    Returns:
        Dict containing upload result information.

    Raises:
        HTTPException: If file upload or processing fails.
    """
    try:
        # Validate file
        _validate_file(file)

        # Prepare file path
        file_path = UPLOAD_DIRECTORY / file.filename

        # Check if file already exists
        if file_path.exists():
            logger.warning(f"File {file.filename} already exists, overwriting")

        # Save file
        contents = await file.read()

        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // 1024 // 1024}MB"
            )

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        logger.info(f"Successfully uploaded file: {file.filename}")

        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "description": description,
            "file_path": str(file_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

def create_app() -> FastAPI:
    """Application factory function.

    Returns:
        Configured FastAPI application instance.
    """
    return application

# app 별칭 생성 (uvicorn 호환성을 위해)
app = application


def init_gpu():
    """GPU 초기화 함수"""
    try:
        import sys
        sys.path.append('.')
        from app.utils.gpu_utils import get_gpu_info, setup_gpu_device
        device = setup_gpu_device()
        gpu_info = get_gpu_info()
        logger.info(f"애플리케이션 시작 - GPU 설정: {device}")
        if gpu_info["cuda_available"]:
            logger.info(f"GPU 메모리: {gpu_info['memory_total']:.1f}GB 사용 가능")
    except Exception as e:
        logger.warning(f"GPU 초기화 중 오류: {e}")

if __name__ == "__main__":
    # GPU 초기화
    init_gpu()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )