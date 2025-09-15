"""SQLAlchemy database client and session management.

This module provides database connection, session management, and initialization
functionality for the RAG chatbot application using SQLAlchemy ORM.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy configuration
DATABASE_URL: str = getattr(settings, 'DATABASE_URL', 'sqlite:///./chatbot.db')

# Create database engine with appropriate configuration
def _create_engine() -> Engine:
    """Create and configure database engine.

    Returns:
        Configured SQLAlchemy engine instance.
    """
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    return create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=False,  # Set to True for SQL query logging
        pool_pre_ping=True,  # Verify connections before use
    )

# Initialize engine and session factory
engine: Engine = _create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Database session dependency for FastAPI.

    Yields:
        Database session that will be automatically closed.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Yields:
        Database session with automatic transaction management.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db() -> None:
    """Initialize database tables and schema.

    Creates all tables defined in the models if they don't exist.

    Raises:
        SQLAlchemyError: If database initialization fails.
    """
    try:
        # Import models to ensure they are registered with the Base
        from app.models.document import Document, DocumentChunk
        from app.models.chat import ChatSession, ChatMessage, MessageSource

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    except ImportError as e:
        logger.error(f"Failed to import models: {e}")
        raise


def check_db_connection() -> bool:
    """Check if database connection is working.

    Returns:
        True if connection is successful, False otherwise.
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        logger.info("Database connection check successful")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def drop_all_tables() -> None:
    """Drop all database tables. Use with caution!

    This function is primarily for testing and development.

    Raises:
        SQLAlchemyError: If table dropping fails.
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except SQLAlchemyError as e:
        logger.error(f"Failed to drop tables: {e}")
        raise