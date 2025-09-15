"""Weaviate 클라이언트 관리"""
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
import logging
import time
import uuid
import json
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.infrastructure.external.embedding_config import EmbeddingConstants

class WeaviateClient:
    """Weaviate 벡터 데이터베이스 클라이언트"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.client:
            self.client.close()
        return False
    
    def _initialize_client(self):
        """Weaviate v4.x 클라이언트 초기화"""
        max_retries = settings.EMBEDDING_MAX_RETRIES
        retry_delay = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Weaviate v4.x 방식
                self.client = weaviate.connect_to_local(
                    host=settings.WEAVIATE_URL.replace('http://', '').replace('https://', '').split(':')[0],
                    port=int(settings.WEAVIATE_URL.split(':')[-1]),
                    grpc_port=50051
                )
                
                # 연결 확인
                if self.client.is_ready():
                    logging.info(f"Weaviate v4 연결 성공: {settings.WEAVIATE_URL}")
                    self._ensure_schema()
                    break
                else:
                    raise ConnectionError("Weaviate가 준비되지 않음")
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Weaviate 연결 실패 (최대 재시도 횟수 초과): {str(e)}")
                    raise
                logging.warning(f"Weaviate 연결 실패 ({retry_count}/{max_retries}), {retry_delay}초 후 재시도: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 1.5
    
    def _ensure_schema(self):
        """Weaviate v4.x 스키마 초기화"""
        try:
            # v4.x에서는 컬렉션 존재 확인이 다름
            if not self.client.collections.exists(EmbeddingConstants.WEAVIATE_CLASS_NAME):
                # v4.x 스타일로 컬렉션 생성
                self.client.collections.create(
                    name=EmbeddingConstants.WEAVIATE_CLASS_NAME,
                    description="Document chunks for RAG system",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="document_id", data_type=DataType.TEXT),
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="metadata", data_type=DataType.TEXT)
                    ],
                    vectorizer_config=Configure.Vectorizer.none()  # 외부 임베딩 사용
                )
                logging.info("Weaviate v4 컬렉션 생성 완료")
        except Exception as e:
            logging.error(f"Weaviate 스키마 설정 중 오류: {str(e)}")
            raise
    
    def store_chunk(self, chunk_id: str, document_id: str, content: str, 
                   embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """청크 벡터 저장 (v4.x)"""
        try:
            if not embedding:
                logging.warning(f"청크 {chunk_id}의 임베딩이 비어있음")
                return ""
            
            meta_dict = metadata or {}
            meta_json = json.dumps(meta_dict)
            
            weaviate_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{document_id}_{chunk_id}"))
            
            # v4.x 방식으로 객체 삽입
            collection = self.client.collections.get(EmbeddingConstants.WEAVIATE_CLASS_NAME)
            collection.data.insert(
                properties={
                    "content": content,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "metadata": meta_json
                },
                uuid=weaviate_id,
                vector=embedding
            )
            
            logging.info(f"청크 {chunk_id} 벡터 저장 완료 (Weaviate ID: {weaviate_id})")
            return weaviate_id
        except Exception as e:
            logging.error(f"벡터 저장 중 오류: {str(e)}")
            return ""
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """벡터 유사도 검색 (v4.x)"""
        try:
            if not query_embedding:
                return []
            
            # v4.x 방식으로 검색
            collection = self.client.collections.get(EmbeddingConstants.WEAVIATE_CLASS_NAME)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_properties=["content", "document_id", "chunk_id", "metadata"]
            )
            
            # 결과를 기존 형식으로 변환
            chunks = []
            for obj in response.objects:
                chunks.append({
                    "content": obj.properties.get("content", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "chunk_id": obj.properties.get("chunk_id", ""),
                    "metadata": obj.properties.get("metadata", "")
                })
            
            return self._process_search_results(chunks)
        except Exception as e:
            logging.error(f"벡터 검색 중 오류: {str(e)}")
            return []
    
    def get_all_contents(self, limit: int = 10000) -> List[str]:
        """모든 문서 내용 가져오기 (TF-IDF 인덱스용, v4.x)"""
        try:
            collection = self.client.collections.get(EmbeddingConstants.WEAVIATE_CLASS_NAME)
            response = collection.query.fetch_objects(
                limit=limit,
                return_properties=["content"]
            )
            
            return [obj.properties.get("content", "") for obj in response.objects]
        except Exception as e:
            logging.error(f"문서 내용 조회 중 오류: {str(e)}")
            return []
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """문서의 모든 청크 삭제 (v4.x)"""
        try:
            logging.info(f"문서 청크 삭제 시작: document_id={document_id}")
            collection = self.client.collections.get(EmbeddingConstants.WEAVIATE_CLASS_NAME)

            # 삭제 전 해당 문서의 청크 수 확인
            count_response = collection.query.fetch_objects(
                where=wvc.query.Filter.by_property("document_id").equal(document_id),
                limit=10000,
                return_properties=["chunk_id"]
            )
            chunk_count = len(count_response.objects)
            logging.info(f"삭제 대상 청크 수: {chunk_count}개")

            # 실제 삭제 수행
            delete_response = collection.data.delete_many(
                where=wvc.query.Filter.by_property("document_id").equal(document_id)
            )

            logging.info(f"문서 {document_id}의 청크 벡터 삭제 완료 - 삭제된 청크: {chunk_count}개")
            return True
        except Exception as e:
            logging.error(f"문서 청크 삭제 중 오류 (document_id: {document_id}): {str(e)}")
            return False
    
    def _process_search_results(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """검색 결과 처리"""
        processed_chunks = []
        
        for chunk in chunks:
            try:
                if "metadata" in chunk and chunk["metadata"]:
                    try:
                        metadata = json.loads(chunk["metadata"])
                        chunk["metadata"] = metadata
                        
                        if "filename" not in chunk and "filename" in metadata:
                            chunk["filename"] = metadata["filename"]
                    except:
                        chunk["metadata"] = {}
                
                chunk["relevance"] = EmbeddingConstants.DEFAULT_RELEVANCE_SCORE
                processed_chunks.append(chunk)
            except Exception as e:
                logging.error(f"청크 정보 처리 중 오류: {str(e)}")
                processed_chunks.append(chunk)
        
        return processed_chunks