"""Neo4j Cypher 쿼리 모음"""

class CypherQueries:
    """Neo4j 그래프 검색용 Cypher 쿼리"""
    
    @staticmethod
    def get_related_chunks_query() -> str:
        """관련 청크 검색 쿼리"""
        return """
        // 벡터 검색으로 찾은 청크의 엔티티가 언급된 다른 청크 찾기
        MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
        WHERE c1.content CONTAINS $query_text AND c1 <> c2
        WITH c2, count(DISTINCT e) AS common_entities
        ORDER BY common_entities DESC
        LIMIT $limit
        
        // 결과 청크의 문서 정보 가져오기
        MATCH (doc:Document)-[:CONTAINS]->(c2)
        RETURN 
            c2.id AS chunk_id,
            c2.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            common_entities AS relevance
        """
    
    @staticmethod
    def get_entity_chunks_query() -> str:
        """엔티티별 청크 검색 쿼리"""
        return """
        MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)
        MATCH (doc:Document)-[:CONTAINS]->(c)
        RETURN 
            c.id AS chunk_id,
            c.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            1.0 AS relevance
        LIMIT $limit
        """
    
    @staticmethod
    def get_document_info_query() -> str:
        """문서 정보 조회 쿼리"""
        return """
        MATCH (d:Document {id: $document_id})
        RETURN d.filename AS filename
        """
    
    @staticmethod
    def build_multi_entity_query(entity_count: int) -> str:
        """여러 엔티티를 모두 포함하는 청크 검색 쿼리 생성"""
        conditions = [
            f"EXISTS ((:Entity {{name: $entity{i}}})<-[:MENTIONS]-(c))"
            for i in range(entity_count)
        ]
        
        return f"""
        MATCH (c:Chunk)
        WHERE {' AND '.join(conditions)}
        MATCH (doc:Document)-[:CONTAINS]->(c)
        RETURN 
            c.id AS chunk_id,
            c.content AS content,
            doc.id AS document_id,
            doc.filename AS filename,
            1.0 AS relevance
        LIMIT 3
        """