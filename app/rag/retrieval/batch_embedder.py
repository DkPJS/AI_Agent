"""배치 임베딩 최적화 시스템"""
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import defaultdict

from app.utils.model_manager import get_shared_embedding_model
from app.infrastructure.external.weaviate_client import WeaviateClient


class EmbeddingCache:
    """임베딩 캐시 시스템"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    def _get_cache_key(self, text: str) -> str:
        """텍스트에 대한 캐시 키 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회"""
        key = self._get_cache_key(text)
        current_time = time.time()

        with self._lock:
            if key in self.cache:
                cached_time, embedding = self.cache[key]

                # TTL 확인
                if current_time - cached_time < self.ttl_seconds:
                    self.access_times[key] = current_time
                    return embedding
                else:
                    # TTL 만료된 항목 제거
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]

        return None

    def put(self, text: str, embedding: List[float]) -> None:
        """임베딩을 캐시에 저장"""
        key = self._get_cache_key(text)
        current_time = time.time()

        with self._lock:
            # 캐시 크기 관리
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            self.cache[key] = (current_time, embedding)
            self.access_times[key] = current_time

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거 (LRU)"""
        if not self.access_times:
            return

        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def clear(self) -> None:
        """캐시 전체 삭제"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        with self._lock:
            current_time = time.time()
            valid_items = sum(
                1 for cached_time, _ in self.cache.values()
                if current_time - cached_time < self.ttl_seconds
            )

            return {
                "total_items": len(self.cache),
                "valid_items": valid_items,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }


class BatchEmbedder:
    """배치 임베딩 처리 최적화"""

    def __init__(self,
                 batch_size: int = 32,
                 max_workers: int = 4,
                 cache_enabled: bool = True):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache = EmbeddingCache() if cache_enabled else None
        self.weaviate_client = WeaviateClient()

        # 성능 통계
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "total_time": 0.0,
            "batch_count": 0
        }

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """배치로 텍스트 임베딩 생성 (캐시 활용)"""
        start_time = time.time()

        if not texts:
            return []

        # 1. 캐시에서 기존 임베딩 확인
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text)
                if cached:
                    cached_embeddings[i] = cached
                    self.stats["cache_hits"] += 1
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)
            self.stats["cache_misses"] += 1

        # 2. 캐시되지 않은 텍스트들을 배치로 처리
        new_embeddings = {}
        if uncached_texts:
            logging.info(f"배치 임베딩 생성: {len(uncached_texts)}개 텍스트")

            try:
                embedding_model = get_shared_embedding_model()
                batch_embeddings = embedding_model.create_embeddings_batch(
                    uncached_texts,
                    batch_size=self.batch_size
                )

                # 새 임베딩을 캐시에 저장
                for i, (text, embedding) in enumerate(zip(uncached_texts, batch_embeddings)):
                    idx = uncached_indices[i]
                    new_embeddings[idx] = embedding

                    if self.cache and embedding:
                        self.cache.put(text, embedding)

                self.stats["batch_count"] += 1

            except Exception as e:
                logging.error(f"배치 임베딩 생성 실패: {e}")
                # 개별 처리로 fallback
                new_embeddings = self._fallback_individual_embedding(uncached_texts, uncached_indices)

        # 3. 결과 재구성
        result = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            elif i in new_embeddings:
                result.append(new_embeddings[i])
            else:
                result.append([])  # 실패한 경우

        # 통계 업데이트
        elapsed_time = time.time() - start_time
        self.stats["total_embeddings"] += len(texts)
        self.stats["total_time"] += elapsed_time

        if show_progress:
            cache_hit_rate = self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            logging.info(f"배치 임베딩 완료: {len(texts)}개, {elapsed_time:.2f}초, 캐시 적중률: {cache_hit_rate:.2%}")

        return result

    def _fallback_individual_embedding(self, texts: List[str], indices: List[int]) -> Dict[int, List[float]]:
        """개별 임베딩 생성 (배치 실패 시 fallback)"""
        logging.warning("배치 임베딩 실패, 개별 처리로 fallback")

        embeddings = {}
        embedding_model = get_shared_embedding_model()

        for i, text in enumerate(texts):
            try:
                embedding = embedding_model.create_embedding(text)
                embeddings[indices[i]] = embedding

                if self.cache and embedding:
                    self.cache.put(text, embedding)

            except Exception as e:
                logging.error(f"개별 임베딩 생성 실패 ({text[:50]}...): {e}")
                embeddings[indices[i]] = []

        return embeddings

    async def embed_and_store_chunks_async(self,
                                          chunks: List[Dict[str, Any]],
                                          document_id: str) -> List[str]:
        """청크들을 비동기로 임베딩하고 Weaviate에 저장"""
        start_time = time.time()

        if not chunks:
            return []

        # 1. 모든 청크 텍스트 추출
        chunk_texts = [chunk.get("content", "") for chunk in chunks]

        # 2. 배치로 임베딩 생성
        embeddings = self.embed_texts(chunk_texts, show_progress=True)

        # 3. 비동기로 Weaviate에 저장
        tasks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # 임베딩이 성공한 경우만
                chunk_id = f"{document_id}_chunk_{i}"
                task = self._store_chunk_async(chunk_id, document_id, chunk, embedding)
                tasks.append(task)

        # 4. 모든 저장 작업 완료 대기
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_ids = []
            for result in results:
                if not isinstance(result, Exception) and result:
                    successful_ids.append(result)

            elapsed_time = time.time() - start_time
            logging.info(f"비동기 청크 저장 완료: {len(successful_ids)}/{len(chunks)}개, {elapsed_time:.2f}초")

            return successful_ids

        return []

    async def _store_chunk_async(self, chunk_id: str, document_id: str,
                               chunk: Dict[str, Any], embedding: List[float]) -> Optional[str]:
        """개별 청크를 비동기로 저장"""
        try:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            # ThreadPoolExecutor를 사용하여 동기 함수를 비동기로 실행
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.weaviate_client.store_chunk,
                    chunk_id, document_id, content, embedding, metadata
                )

            return result

        except Exception as e:
            logging.error(f"청크 저장 실패 ({chunk_id}): {e}")
            return None

    def embed_documents_batch(self,
                            documents: List[Dict[str, Any]],
                            chunking_config: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """여러 문서를 배치로 처리 (청킹 + 임베딩 + 저장)"""
        if not documents:
            return {}

        chunking_config = chunking_config or {
            "max_chunk_size": 1000,
            "use_semantic": True,
            "document_type": "general"
        }

        # 1. 모든 문서를 청킹
        all_chunks = []
        document_chunk_mapping = {}  # 문서별 청크 인덱스 추적

        chunk_index = 0
        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")
            doc_type = doc.get("type", chunking_config["document_type"])

            # 문서 청킹
            from app.services.document.base_processor import BaseDocumentProcessor
            doc_chunks = BaseDocumentProcessor.chunk_text(
                content,
                max_chunk_size=chunking_config["max_chunk_size"],
                use_semantic=chunking_config["use_semantic"],
                document_type=doc_type
            )

            # 청크에 문서 정보 추가
            for chunk in doc_chunks:
                chunk["document_id"] = doc_id
                chunk["chunk_id"] = f"{doc_id}_chunk_{len(all_chunks)}"
                all_chunks.append(chunk)

            document_chunk_mapping[doc_id] = list(range(chunk_index, chunk_index + len(doc_chunks)))
            chunk_index += len(doc_chunks)

        # 2. 모든 청크를 배치로 임베딩
        chunk_texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self.embed_texts(chunk_texts, show_progress=True)

        # 3. 임베딩과 청크 매칭하여 저장
        results = {}
        stored_count = 0

        for doc_id, chunk_indices in document_chunk_mapping.items():
            doc_results = []

            for chunk_idx in chunk_indices:
                chunk = all_chunks[chunk_idx]
                embedding = embeddings[chunk_idx]

                if embedding:
                    try:
                        weaviate_id = self.weaviate_client.store_chunk(
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["content"],
                            embedding,
                            chunk.get("metadata", {})
                        )

                        if weaviate_id:
                            doc_results.append(weaviate_id)
                            stored_count += 1

                    except Exception as e:
                        logging.error(f"청크 저장 실패 ({chunk['chunk_id']}): {e}")

            results[doc_id] = doc_results

        logging.info(f"배치 문서 처리 완료: {len(documents)}개 문서, {stored_count}개 청크 저장")
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = self.stats["cache_hits"] / max(1, total_requests)
        avg_time_per_embedding = self.stats["total_time"] / max(1, self.stats["total_embeddings"])

        stats = {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "avg_time_per_embedding": avg_time_per_embedding,
            "total_requests": total_requests
        }

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    def clear_cache(self) -> None:
        """캐시 초기화"""
        if self.cache:
            self.cache.clear()
            logging.info("임베딩 캐시 초기화 완료")


# 전역 배치 임베더 인스턴스
_batch_embedder = None

def get_batch_embedder() -> BatchEmbedder:
    """전역 배치 임베더 인스턴스 반환"""
    global _batch_embedder
    if _batch_embedder is None:
        _batch_embedder = BatchEmbedder()
    return _batch_embedder