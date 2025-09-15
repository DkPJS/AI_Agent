"""Chat-related SQLAlchemy models.

This module defines database models for chat session management,
message storage, and source tracking for the RAG chatbot system.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship, Mapped

from app.infrastructure.database.sql_client import Base

class ChatSession(Base):
    """Chat session model for managing conversation contexts.

    This model represents individual chat sessions that group related
    messages together, providing conversation history and context management.

    Attributes:
        id: Unique session identifier (UUID)
        title: Optional human-readable session title
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        messages: Related chat messages in this session
    """
    __tablename__ = "chat_sessions"

    id: Mapped[str] = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    title: Mapped[Optional[str]] = Column(String(255), nullable=True)
    created_at: Mapped[datetime] = Column(
        DateTime,
        default=datetime.now,
        nullable=False
    )
    updated_at: Mapped[datetime] = Column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False
    )

    # Relationships
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="ChatMessage.created_at"
    )

    def __repr__(self) -> str:
        """String representation of the ChatSession.

        Returns:
            Human-readable representation of the session.
        """
        return f"<ChatSession(id='{self.id}', title='{self.title}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary representation.

        Returns:
            Dictionary containing session metadata and statistics.
        """
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.messages.count() if self.messages else 0
        }

    @property
    def last_message_time(self) -> Optional[datetime]:
        """Get timestamp of the most recent message.

        Returns:
            Timestamp of last message or None if no messages exist.
        """
        if self.messages:
            last_message = self.messages.order_by(
                ChatMessage.created_at.desc()
            ).first()
            return last_message.created_at if last_message else None
        return None

    def generate_title_from_first_message(self) -> Optional[str]:
        """Generate session title from first user message.

        Returns:
            Generated title or None if no suitable message found.
        """
        if self.messages:
            first_user_message = self.messages.filter_by(role="user").first()
            if first_user_message and first_user_message.content:
                # Take first 50 characters as title
                content = first_user_message.content.strip()
                return content[:50] + "..." if len(content) > 50 else content
        return None

class ChatMessage(Base):
    """채팅 메시지 모델"""
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' 또는 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # 관계 설정
    session = relationship("ChatSession", back_populates="messages")
    sources = relationship("MessageSource", back_populates="message", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatMessage(id='{self.id}', session_id='{self.session_id}', role='{self.role}')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "sources": [source.to_dict() for source in self.sources] if self.sources else []
        }

class MessageSource(Base):
    """메시지 소스(출처) 모델"""
    __tablename__ = "message_sources"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String(36), ForeignKey("chat_messages.id"), nullable=False)
    document_id = Column(String(36), nullable=False)
    chunk_id = Column(String(36), nullable=False)
    relevance_score = Column(Integer, nullable=True)  # 관련성 점수 (0-100)
    
    # 관계 설정
    message = relationship("ChatMessage", back_populates="sources")
    
    def __repr__(self):
        return f"<MessageSource(id='{self.id}', message_id='{self.message_id}', document_id='{self.document_id}')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "relevance_score": self.relevance_score
        }