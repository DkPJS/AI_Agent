from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.infrastructure.external.weaviate_client import WeaviateClient
from app.infrastructure.external.text_processor import TextProcessor
from app.infrastructure.external.embedding_config import EmbeddingConstants
from app.utils.model_manager import get_shared_embedding_model

class DocumentEmbedder:
    """문서 임베딩 생성 및 관리"""
    
    def __init__(self):
        self.weaviate_client = WeaviateClient()
        self.text_processor = TextProcessor()
        
        # TF-IDF 벡터라이저 (하이브리드 검색용)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=EmbeddingConstants.TFIDF_MAX_FEATURES,
            stop_words=None,
            ngram_range=EmbeddingConstants.TFIDF_NGRAM_RANGE,
            analyzer='word'
        )
        self.tfidf_matrix = None
        self.chunk_contents = []
    def create_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 벡터 생성 (캐시 활용)"""
        from app.rag.retrieval.batch_embedder import get_batch_embedder
        batch_embedder = get_batch_embedder()

        # 단일 텍스트도 배치 처리를 통해 캐시 활용
        embeddings = batch_embedder.embed_texts([text], show_progress=False)
        return embeddings[0] if embeddings else []

    def _generate_embedding(self, text: str) -> List[float]:
        """내부 임베딩 생성 메서드 (테스트용)"""
        return self.create_embedding(text)

    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """배치 텍스트 임베딩 벡터 생성 (최적화된 버전)"""
        from app.rag.retrieval.batch_embedder import get_batch_embedder
        batch_embedder = get_batch_embedder()

        if batch_size:
            batch_embedder.batch_size = batch_size

        return batch_embedder.embed_texts(texts, show_progress=True)
    
    def store_chunk(self, chunk_id: str, document_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """청크 텍스트 임베딩하여 Weaviate에 저장"""
        embedding = self.create_embedding(content)
        return self.weaviate_client.store_chunk(chunk_id, document_id, content, embedding, metadata)
    
    def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서 청크 검색"""
        try:
            query_embedding = self.create_embedding(query)
            
            if not query_embedding:
                logging.warning("쿼리 임베딩 생성 실패")
                return []
            
            logging.info(f"쿼리 '{query}'에 대한 임베딩 생성 성공, 벡터 검색 시작")
            
            # Weaviate 클라이언트를 통한 검색
            return self.weaviate_client.search_similar(query_embedding, limit)
            
        except Exception as e:
            logging.error(f"유사 청크 검색 중 오류: {str(e)}")
            return []
            
    def _enhance_chunk_info(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """청크 정보 확장 (Neo4j에서 추가 정보 가져오기)"""
        try:
            chunk_id = chunk.get("chunk_id")
            document_id = chunk.get("document_id")
            
            if not chunk_id or not document_id:
                return chunk
                
            # 파일명 정보가 없으면 Neo4j에서 가져옴
            if "filename" not in chunk or not chunk["filename"]:
                try:
                    from app.infrastructure.database.neo4j_client import Neo4jClient
                    
                    with Neo4jClient() as neo4j_client:
                        with neo4j_client.driver.session() as session:
                            # 문서 정보 조회
                            result = session.run(
                                """
                                MATCH (d:Document {id: $document_id})
                                RETURN d.filename AS filename
                                """,
                                document_id=document_id
                            )
                            
                            record = result.single()
                            if record and "filename" in record:
                                chunk["filename"] = record["filename"]
                except Exception as e:
                    logging.error(f"Neo4j에서 파일명 조회 실패: {str(e)}")
            
            return chunk
        except Exception as e:
            logging.error(f"청크 정보 확장 중 오류: {str(e)}")
            return chunk
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """문서의 모든 청크 삭제"""
        return self.weaviate_client.delete_document_chunks(document_id)

    def hybrid_search(self, query: str, limit: int = 10, 
                     semantic_weight: float = 0.7, 
                     keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """하이브리드 검색: 시맨틱 + 키워드 검색 결합"""
        try:
            # 1. 시맨틱 검색
            semantic_results = self.search_similar(query, limit=limit*2)
            
            # 2. 키워드 검색 (TF-IDF)
            keyword_results = self._keyword_search(query, limit=limit*2)
            
            # 3. 결과 통합 및 재순위화
            combined_results = self._combine_and_rerank(
                query, semantic_results, keyword_results,
                semantic_weight, keyword_weight
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            logging.error(f"하이브리드 검색 중 오류: {str(e)}")
            return self.search_similar(query, limit)  # 폴백
    
    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """TF-IDF 기반 키워드 검색"""
        try:
            if self.tfidf_matrix is None:
                self._build_tfidf_index()
            
            if self.tfidf_matrix is None:
                return []
            
            # 쿼리 벡터화
            query_preprocessed = self.text_processor.preprocess_korean_text(query)
            query_vector = self.tfidf_vectorizer.transform([query_preprocessed])
            
            # 유사도 계산
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # 상위 결과 선택
            top_indices = similarities.argsort()[-limit:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # 임계값 설정
                    result = {
                        'content': self.chunk_contents[idx],
                        'relevance': float(similarities[idx]),
                        'search_type': 'keyword'
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"키워드 검색 중 오류: {str(e)}")
            return []
    
    def _build_tfidf_index(self):
        """TF-IDF 인덱스 구축"""
        try:
            # Weaviate 클라이언트를 통해 모든 문서 내용 가져오기
            self.chunk_contents = self.weaviate_client.get_all_contents()
            
            if self.chunk_contents:
                # 한국어 텍스트 전처리
                processed_contents = [
                    self.text_processor.preprocess_korean_text(content) 
                    for content in self.chunk_contents
                ]
                
                # TF-IDF 매트릭스 생성
                if processed_contents:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_contents)
                    logging.info(f"TF-IDF 인덱스 구축 완료: {len(self.chunk_contents)}개 청크")
                
        except Exception as e:
            logging.error(f"TF-IDF 인덱스 구축 중 오류: {str(e)}")
    
    
    def _combine_and_rerank(self, query: str, semantic_results: List[Dict], 
                           keyword_results: List[Dict], 
                           semantic_weight: float, keyword_weight: float) -> List[Dict[str, Any]]:
        """검색 결과 통합 및 재순위화"""
        try:
            # 결과를 content 기준으로 통합
            combined = {}
            
            # 시맨틱 검색 결과 추가
            for result in semantic_results:
                content = result.get('content', '')
                if content:
                    combined[content] = {
                        **result,
                        'semantic_score': result.get('relevance', 0.0),
                        'keyword_score': 0.0,
                        'combined_score': 0.0
                    }
            
            # 키워드 검색 결과 추가/업데이트
            for result in keyword_results:
                content = result.get('content', '')
                if content:
                    if content in combined:
                        combined[content]['keyword_score'] = result.get('relevance', 0.0)
                    else:
                        combined[content] = {
                            **result,
                            'semantic_score': 0.0,
                            'keyword_score': result.get('relevance', 0.0),
                            'combined_score': 0.0
                        }
            
            # 최종 점수 계산
            for content, result in combined.items():
                # 정규화된 가중 합계
                semantic_norm = min(result['semantic_score'] * 1.25, 1.0)  # 시맨틱 점수 부스트
                keyword_norm = min(result['keyword_score'] * 2.0, 1.0)     # 키워드 점수 부스트
                
                result['combined_score'] = (
                    semantic_weight * semantic_norm + 
                    keyword_weight * keyword_norm
                )
                
                # 추가 점수 조정 (질문과 직접 매칭되는 키워드가 있으면 부스트)
                if self.text_processor.has_direct_keyword_match(query, content):
                    result['combined_score'] *= 1.2
            
            # 점수순 정렬
            sorted_results = sorted(
                combined.values(), 
                key=lambda x: x['combined_score'], 
                reverse=True
            )
            
            return sorted_results
            
        except Exception as e:
            logging.error(f"결과 통합 중 오류: {str(e)}")
            return semantic_results
    

    async def search_similar_chunks(self, query: str, limit: int = 5,
                                   use_hybrid: bool = True,
                                   search_strategy: str = "auto",
                                   filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """고급 유사 문서 검색 (메타데이터 인식 하이브리드 검색)"""
        try:
            logging.info(f"질문 '{query}'에 대한 고급 검색 시작 (전략: {search_strategy})")

            if use_hybrid:
                # 새로운 고급 하이브리드 검색 사용
                from app.rag.retrieval.advanced_search import AdvancedHybridSearch
                advanced_searcher = AdvancedHybridSearch()

                search_results = advanced_searcher.search(
                    query=query,
                    limit=limit,
                    search_strategy=search_strategy,
                    filters=filters
                )

                # SearchResult 객체를 Dict로 변환
                results = []
                for result in search_results:
                    results.append({
                        "content": result.content,
                        "document_id": result.document_id,
                        "chunk_id": result.chunk_id,
                        "relevance": result.relevance_score,
                        "search_method": result.search_method,
                        "metadata": result.metadata,
                        "scores": {
                            "semantic": result.semantic_score,
                            "keyword": result.keyword_score,
                            "structure": result.structure_score,
                            "recency": result.recency_score,
                            "popularity": result.popularity_score
                        }
                    })
            else:
                results = self.search_similar(query, limit=limit)

            logging.info(f"고급 검색 결과: {len(results)} 개의 청크 발견")
            return results

        except Exception as e:
            logging.error(f"고급 검색 오류: {str(e)}, 기본 검색으로 fallback")
            # 기본 검색으로 fallback
            return self.search_similar(query, limit=limit)