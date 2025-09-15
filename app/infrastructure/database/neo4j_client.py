"""Neo4j graph database client for advanced document relationships.

This module provides a comprehensive Neo4j client for storing and querying
document relationships, entity connections, and semantic graph structures.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Generator

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j graph database client for document relationship management.

    This client provides comprehensive functionality for:
    - Document and chunk node management
    - Entity extraction and linking
    - Semantic relationship mapping
    - Advanced graph-based search capabilities
    - Synonym and ontology management

    Attributes:
        uri: Neo4j database URI
        username: Database username
        password: Database password
        driver: Neo4j driver instance
    """

    def __init__(self) -> None:
        """Initialize Neo4j client with connection and schema setup.

        Raises:
            Neo4jError: If connection or schema setup fails.
        """
        self.uri: str = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        self.username: str = getattr(settings, 'NEO4J_USERNAME', 'neo4j')
        self.password: str = getattr(settings, 'NEO4J_PASSWORD', 'password')
        self.driver: Optional[Driver] = None

        self.connect()
        self._setup_schema()
    
    def __enter__(self):
        """컨텍스트 매니저 시작"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
    
    def connect(self):
        """Neo4j 데이터베이스 연결"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # 연결 테스트
            self.driver.verify_connectivity()
            logging.info("Neo4j 데이터베이스 연결 성공")
        except Exception as e:
            logging.error(f"Neo4j 연결 실패: {str(e)}")
            raise
    
    def close(self):
        """드라이버 연결 종료"""
        if self.driver:
            self.driver.close()
    
    def _setup_schema(self):
        """그래프 데이터베이스 스키마 설정"""
        # 인덱스 및 제약 조건 생성
        with self.driver.session() as session:
            # Document 노드 ID 고유성 제약
            session.run("""
                CREATE CONSTRAINT document_id IF NOT EXISTS
                FOR (d:Document) REQUIRE d.id IS UNIQUE
            """)
            
            # Chunk 노드 ID 고유성 제약
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            
            # Entity 노드 이름 고유성 제약
            session.run("""
                CREATE CONSTRAINT entity_name IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """)
            
            # 텍스트 검색을 위한 전체 텍스트 인덱스
            session.run("""
                CREATE FULLTEXT INDEX chunk_content IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.content]
            """)
    
    def create_document(self, document_data: Dict[str, Any]) -> str:
        """문서 노드 생성"""
        query = """
        CREATE (d:Document {
            id: $id,
            filename: $filename,
            content_type: $content_type,
            size: $size,
            description: $description,
            upload_date: datetime($upload_date)
        })
        RETURN d.id AS document_id
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                id=document_data["id"],
                filename=document_data["filename"],
                content_type=document_data["content_type"],
                size=document_data["size"],
                description=document_data.get("description", ""),
                upload_date=document_data["upload_date"].isoformat()
            )
            return result.single()["document_id"]
    
    def create_chunk(self, chunk_data: Dict[str, Any], document_id: str) -> str:
        """청크 노드 생성 및 문서에 연결"""
        query = """
        MATCH (d:Document {id: $document_id})
        CREATE (c:Chunk {
            id: $id,
            content: $content,
            chunk_index: $chunk_index,
            embedding_id: $embedding_id
        })
        CREATE (d)-[:CONTAINS]->(c)
        RETURN c.id AS chunk_id
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                document_id=document_id,
                id=chunk_data["id"],
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                embedding_id=chunk_data.get("embedding_id", "")
            )
            return result.single()["chunk_id"]
    
    def extract_and_link_entities(self, chunk_id: str, entities: List[Dict[str, Any]]):
        """청크에서 추출한 엔티티를 노드로 생성하고 연결"""
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {name: $entity_name})
        ON CREATE SET e.type = $entity_type
        CREATE (c)-[:MENTIONS {confidence: $confidence}]->(e)
        """
        
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    query,
                    chunk_id=chunk_id,
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    confidence=entity.get("score", 0.0)
                )
    
    def search_by_entity(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """엔티티로 청크 검색"""
        query = """
        MATCH (e:Entity {name: $entity_name})<-[m:MENTIONS]-(chunk:Chunk)
        MATCH (doc:Document)-[:CONTAINS]->(chunk)
        RETURN 
            chunk.id AS chunk_id,
            chunk.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            m.confidence AS relevance
        ORDER BY m.confidence DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                entity_name=entity_name,
                limit=limit
            )
            return [dict(record) for record in result]
    
    def search_related_chunks(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """키워드 기반 관련 청크 검색"""
        cypher_query = """
        CALL db.index.fulltext.queryNodes("chunk_content", $query_text) 
        YIELD node, score
        WITH node as chunk, score
        MATCH (doc:Document)-[:CONTAINS]->(chunk)
        RETURN 
            chunk.id AS chunk_id, 
            chunk.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            score AS relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(
                cypher_query,
                query_text=query_text,
                limit=limit
            )
            return [dict(record) for record in result]
    
    def add_synonym_relation(self, term1: str, term2: str):
        """두 엔티티 간에 동의어 관계 설정"""
        query = """
        MERGE (e1:Entity {name: $term1})
        MERGE (e2:Entity {name: $term2})
        MERGE (e1)-[r:SYNONYM_OF]-(e2)
        RETURN e1.name, e2.name
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                term1=term1,
                term2=term2
            )
            return result.single()
            
    def create_section_relationship(self, document_id: str, parent_title: str, child_title: str, relationship_type: str = "CONTAINS"):
        """문서 섹션 간 계층 관계 생성"""
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c1:Chunk)-[:BELONGS_TO]->(d)
        MATCH (c2:Chunk)-[:BELONGS_TO]->(d)
        WHERE c1.section_title = $parent_title AND c2.section_title = $child_title
        MERGE (c1)-[r:`" + relationship_type + "`]->(c2)
        RETURN c1.section_title as parent, c2.section_title as child, type(r) as relationship
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    document_id=document_id,
                    parent_title=parent_title,
                    child_title=child_title
                )
                return result.single()
        except Exception as e:
            logging.error(f"섹션 관계 생성 중 오류: {str(e)}")
            # 섹션 관계 생성 실패하더라도 계속 진행할 수 있도록 None 반환
            return None

    def enhanced_search(self, query_text, limit=5):
        """컬럼 이름에 구애받지 않는 향상된 검색"""
        # 기존 전체 텍스트 검색 활용
        results = self.search_related_chunks(query_text, limit)
        
        # 동의어 및 유사 용어 확장 쿼리
        expanded_query = """
        MATCH (e:Entity)-[:SYNONYM_OF]-(related:Entity)
        WHERE e.name IN $terms
        WITH collect(distinct related.name) AS synonyms
        CALL db.index.fulltext.queryNodes("chunk_content", apoc.text.join(synonyms, " OR ")) 
        YIELD node, score
        WITH node as chunk, score
        MATCH (doc:Document)-[:CONTAINS]->(chunk)
        RETURN 
            chunk.id AS chunk_id, 
            chunk.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            score AS relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """
        
        # 쿼리에서 핵심 용어 추출 (간단한 예시)
        terms = [term.strip() for term in query_text.split() if len(term.strip()) > 3]
        
        with self.driver.session() as session:
            expanded_results = session.run(
                expanded_query,
                terms=terms,
                limit=limit
            )
            return results + [dict(record) for record in expanded_results]

