"""임베딩 관련 설정 상수"""

class EmbeddingConstants:
    """임베딩 시스템 상수"""
    
    # TF-IDF 설정
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # 한국어 불용어
    KOREAN_STOPWORDS = {
        '이', '그', '저', '것', '들', '를', '을', '가', '이', '는', '은', 
        '에', '의', '와', '과', '으로', '로', '에서', '까지', '부터', 
        '하지만', '그러나', '그런데'
    }
    
    # 임베딩 기본값
    DEFAULT_RELEVANCE_SCORE = 0.8
    MIN_KEYWORD_SIMILARITY = 0.01
    SEMANTIC_SCORE_BOOST = 1.25
    KEYWORD_SCORE_BOOST = 2.0
    DIRECT_MATCH_BOOST = 1.2
    
    # Weaviate 스키마
    WEAVIATE_CLASS_NAME = "DocumentChunk"
    WEAVIATE_CLASS_SCHEMA = {
        "class": "DocumentChunk",
        "description": "Document chunks for RAG system",
        "vectorizer": "none",
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The text content of the chunk"
            },
            {
                "name": "document_id",
                "dataType": ["string"],
                "description": "ID of the parent document"
            },
            {
                "name": "chunk_id",
                "dataType": ["string"],
                "description": "ID of the chunk in the database"
            },
            {
                "name": "metadata",
                "dataType": ["text"],
                "description": "JSON metadata about the document"
            }
        ]
    }