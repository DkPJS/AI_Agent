"""Document-related SQLAlchemy models.

This module defines the database models for document storage and management,
including document metadata and chunked content for efficient retrieval.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship, Mapped

from app.infrastructure.database.sql_client import Base

class Document(Base):
    """Document metadata model for storing file information.

    This model stores essential metadata about uploaded documents including
    file information, content type, and relationships to document chunks.

    Attributes:
        id: Unique document identifier (UUID)
        filename: Original filename of the uploaded document
        file_path: Storage path of the document file
        content_type: MIME type of the document
        size: File size in bytes
        description: Optional description provided during upload
        upload_date: Timestamp when the document was uploaded
        chunks: Related document chunks for this document
    """
    __tablename__ = "documents"

    id: Mapped[str] = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = Column(String(255), nullable=False, index=True)
    file_path: Mapped[str] = Column(String(512), nullable=False)
    content_type: Mapped[str] = Column(String(100), nullable=False)
    size: Mapped[int] = Column(Integer, nullable=False)
    description: Mapped[Optional[str]] = Column(Text, nullable=True)
    upload_date: Mapped[datetime] = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    def __repr__(self) -> str:
        """String representation of the Document.

        Returns:
            Human-readable representation of the document.
        """
        return f"<Document(id='{self.id}', filename='{self.filename}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation.

        Returns:
            Dictionary containing document metadata and statistics.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "description": self.description,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "chunk_count": self.chunks.count() if self.chunks else 0
        }

    @property
    def is_text_document(self) -> bool:
        """Check if document is a text-based format.

        Returns:
            True if document is text-based, False otherwise.
        """
        text_types = {
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        }
        return self.content_type in text_types

class DocumentChunk(Base):
    """Document chunk model for storing segmented document content.

    This model stores individual chunks of documents that have been split
    for processing and embedding. Each chunk maintains a reference to its
    parent document and position within the document.

    Attributes:
        id: Unique chunk identifier (UUID)
        document_id: Foreign key reference to parent document
        content: The actual text content of this chunk
        chunk_index: Sequential position of chunk within the document
        embedding_id: Reference ID for vector embeddings (e.g., Weaviate)
        document: Back-reference to parent document
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    content: Mapped[str] = Column(Text, nullable=False)
    chunk_index: Mapped[int] = Column(Integer, nullable=False)
    embedding_id: Mapped[Optional[str]] = Column(String(36), nullable=True, index=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        """String representation of the DocumentChunk.

        Returns:
            Human-readable representation of the chunk.
        """
        return (
            f"<DocumentChunk(id='{self.id}', "
            f"document_id='{self.document_id}', "
            f"index={self.chunk_index})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation.

        Returns:
            Dictionary containing chunk data and metadata.
        """
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "embedding_id": self.embedding_id,
            "content_length": len(self.content) if self.content else 0
        }

    @property
    def content_preview(self) -> str:
        """Get a preview of the chunk content.

        Returns:
            First 100 characters of content with ellipsis if truncated.
        """
        if not self.content:
            return ""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content

    @property
    def word_count(self) -> int:
        """Get approximate word count of the chunk.

        Returns:
            Number of words in the chunk content.
        """
        if not self.content:
            return 0
        return len(self.content.split())