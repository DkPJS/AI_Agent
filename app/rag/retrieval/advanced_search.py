"""고급 하이브리드 검색 시스템"""
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.rag.retrieval.batch_embedder import get_batch_embedder
from app.infrastructure.external.weaviate_client import WeaviateClient
from app.infrastructure.external.text_processor import TextProcessor


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    document_id: str
    chunk_id: str
    relevance_score: float
    search_method: str
    metadata: Dict[str, Any]

    # 세부 점수들
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    structure_score: float = 0.0
    recency_score: float = 0.0
    popularity_score: float = 0.0


class AdvancedHybridSearch:
    """개선된 하이브리드 검색 시스템"""

    def __init__(self):
        self.weaviate_client = WeaviateClient()
        self.text_processor = TextProcessor()
        self.batch_embedder = get_batch_embedder()

        # 동적 가중치 (쿼리 유형에 따라 조정)
        self.weight_profiles = {
            "factual": {
                "semantic": 0.6,
                "keyword": 0.25,
                "structure": 0.1,
                "recency": 0.03,
                "popularity": 0.02
            },
            "conceptual": {
                "semantic": 0.75,
                "keyword": 0.15,
                "structure": 0.05,
                "recency": 0.03,
                "popularity": 0.02
            },
            "procedural": {
                "semantic": 0.5,
                "keyword": 0.3,
                "structure": 0.15,
                "recency": 0.03,
                "popularity": 0.02
            },
            "exploratory": {
                "semantic": 0.4,
                "keyword": 0.35,
                "structure": 0.1,
                "recency": 0.08,
                "popularity": 0.07
            }
        }

    def search(self, query: str,
               limit: int = 10,
               search_strategy: str = "auto",
               filters: Dict[str, Any] = None) -> List[SearchResult]:
        """
        고급 하이브리드 검색

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수
            search_strategy: 검색 전략 ("auto", "factual", "conceptual", "procedural", "exploratory")
            filters: 추가 필터 조건

        Returns:
            List[SearchResult]: 정렬된 검색 결과
        """
        try:
            # 1. 쿼리 분석 및 전략 결정
            if search_strategy == "auto":
                search_strategy = self._analyze_query_type(query)

            logging.info(f"하이브리드 검색 시작: '{query}' (전략: {search_strategy})")

            # 2. 다중 검색 실행
            search_results = self._execute_multiple_searches(query, limit * 3)

            # 3. 결과 융합 및 재순위화
            fused_results = self._fuse_and_rerank(
                query, search_results, search_strategy, limit
            )

            # 4. 필터 적용
            if filters:
                fused_results = self._apply_filters(fused_results, filters)

            logging.info(f"하이브리드 검색 완료: {len(fused_results)}개 결과")
            return fused_results[:limit]

        except Exception as e:
            logging.error(f"하이브리드 검색 중 오류: {e}")
            # 기본 시맨틱 검색으로 fallback
            return self._fallback_search(query, limit)

    def _analyze_query_type(self, query: str) -> str:
        """쿼리 유형 자동 분석"""
        query_lower = query.lower()

        # 사실 기반 질문 패턴
        factual_patterns = [
            r'^(누가|무엇|언제|어디서|얼마나|몇)',
            r'(정의|의미|개념).*무엇',
            r'(몇|얼마).*인가',
            r'^.*(은|는).*무엇',
            r'.*란 무엇',
            r'.*이란$',
            r'.*란$',
            r'.*무엇인가'
        ]

        # 절차/방법 기반 질문 패턴
        procedural_patterns = [
            r'^(어떻게|어떤 방법)',
            r'(방법|절차|단계|과정)',
            r'^.*하려면 어떻게',
            r'(설치|설정|구성).*방법',
            r'.*어떻게.*하나',
            r'.*구현.*방법',
            r'.*만드는.*방법'
        ]

        # 개념적/설명 질문 패턴
        conceptual_patterns = [
            r'^(왜|어떤 이유)',
            r'(원리|이유|배경|목적)',
            r'^.*인 이유',
            r'(장점|단점|특징|차이점)'
        ]

        # 탐색적 질문 패턴
        exploratory_patterns = [
            r'^(어떤.*있는지|어떤.*종류)',
            r'(예시|사례|경우)',
            r'^.*에 대해 알려줘',
            r'(관련된|연관된).*무엇'
        ]

        # 패턴 매칭
        for pattern in factual_patterns:
            if re.search(pattern, query):
                return "factual"

        for pattern in procedural_patterns:
            if re.search(pattern, query):
                return "procedural"

        for pattern in conceptual_patterns:
            if re.search(pattern, query):
                return "conceptual"

        for pattern in exploratory_patterns:
            if re.search(pattern, query):
                return "exploratory"

        # 기본값
        return "conceptual"

    def _execute_multiple_searches(self, query: str, limit: int) -> Dict[str, List[Dict]]:
        """다중 검색 방법 실행"""
        results = {}

        try:
            # 1. 시맨틱 검색 (기본)
            results["semantic"] = self._semantic_search(query, limit)
        except Exception as e:
            logging.error(f"시맨틱 검색 실패: {e}")
            results["semantic"] = []

        try:
            # 2. 키워드 검색 (BM25 기반)
            results["keyword"] = self._bm25_search(query, limit)
        except Exception as e:
            logging.error(f"키워드 검색 실패: {e}")
            results["keyword"] = []

        try:
            # 3. 구조 기반 검색 (헤더, 메타데이터 활용)
            results["structure"] = self._structure_search(query, limit)
        except Exception as e:
            logging.error(f"구조 검색 실패: {e}")
            results["structure"] = []

        return results

    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """개선된 시맨틱 검색"""
        try:
            query_embedding = self.batch_embedder.embed_texts([query], show_progress=False)[0]
            if not query_embedding:
                return []

            raw_results = self.weaviate_client.search_similar(query_embedding, limit)

            # 시맨틱 점수 정규화 및 메타데이터 추가
            results = []
            for result in raw_results:
                results.append({
                    **result,
                    "search_method": "semantic",
                    "raw_score": result.get("relevance", 0.0)
                })

            return results

        except Exception as e:
            logging.error(f"시맨틱 검색 오류: {e}")
            return []

    def _bm25_search(self, query: str, limit: int) -> List[Dict]:
        """BM25 기반 키워드 검색"""
        try:
            # 전체 문서 컨텐츠 가져오기
            all_contents = self.weaviate_client.get_all_contents(limit=5000)
            if not all_contents:
                return []

            # 쿼리 전처리
            processed_query = self.text_processor.preprocess_korean_text(query)
            processed_contents = [
                self.text_processor.preprocess_korean_text(content)
                for content in all_contents
            ]

            # TF-IDF 벡터화 (BM25 유사)
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,  # BM25 유사한 효과
                norm='l2'
            )

            try:
                tfidf_matrix = vectorizer.fit_transform(processed_contents)
                query_vector = vectorizer.transform([processed_query])

                # 코사인 유사도 계산
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

                # 상위 결과 선택
                top_indices = similarities.argsort()[-limit:][::-1]

                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.01:  # 임계값
                        results.append({
                            "content": all_contents[idx],
                            "relevance": float(similarities[idx]),
                            "search_method": "keyword",
                            "raw_score": float(similarities[idx])
                        })

                return results

            except Exception as ve:
                logging.error(f"벡터화 오류: {ve}")
                return []

        except Exception as e:
            logging.error(f"BM25 검색 오류: {e}")
            return []

    def _structure_search(self, query: str, limit: int) -> List[Dict]:
        """구조 기반 검색 (헤더, 섹션, 메타데이터 활용)"""
        try:
            # 구조적 키워드 추출
            structure_keywords = self._extract_structure_keywords(query)
            if not structure_keywords:
                return []

            # 메타데이터 필터링된 시맨틱 검색
            query_embedding = self.batch_embedder.embed_texts([query], show_progress=False)[0]
            if not query_embedding:
                return []

            # Weaviate에서 메타데이터 조건을 포함한 검색
            # (실제 구현에서는 Weaviate의 where 조건 사용)
            raw_results = self.weaviate_client.search_similar(query_embedding, limit * 2)

            # 구조 점수 계산
            results = []
            for result in raw_results:
                structure_score = self._calculate_structure_score(
                    result, structure_keywords
                )

                if structure_score > 0.1:  # 구조 관련성이 있는 경우만
                    results.append({
                        **result,
                        "search_method": "structure",
                        "structure_score": structure_score,
                        "raw_score": result.get("relevance", 0.0)
                    })

            # 구조 점수 기준 정렬
            results.sort(key=lambda x: x["structure_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logging.error(f"구조 검색 오류: {e}")
            return []

    def _extract_structure_keywords(self, query: str) -> List[str]:
        """쿼리에서 구조적 키워드 추출"""
        structure_patterns = [
            r'(제\d+장|제\d+절|제\d+조)',  # 장, 절, 조
            r'(\d+\.\d+|\d+\.)',           # 번호 매기기
            r'(서론|결론|요약|개요)',        # 섹션명
            r'(목차|차례|목록)',            # 구조 관련
            r'(표|그림|도표)',              # 시각 요소
        ]

        keywords = []
        for pattern in structure_patterns:
            matches = re.findall(pattern, query)
            keywords.extend(matches)

        return keywords

    def _calculate_structure_score(self, result: Dict, structure_keywords: List[str]) -> float:
        """구조 점수 계산"""
        if not structure_keywords:
            return 0.0

        content = result.get("content", "").lower()
        metadata_str = str(result.get("metadata", "")).lower()
        combined_text = content + " " + metadata_str

        score = 0.0
        for keyword in structure_keywords:
            if isinstance(keyword, tuple):
                keyword = keyword[0] if keyword else ""

            if keyword.lower() in combined_text:
                score += 0.2

        return min(score, 1.0)

    def _fuse_and_rerank(self,
                        query: str,
                        search_results: Dict[str, List[Dict]],
                        strategy: str,
                        limit: int) -> List[SearchResult]:
        """결과 융합 및 재순위화"""
        try:
            # 전략별 가중치 가져오기
            weights = self.weight_profiles.get(strategy, self.weight_profiles["conceptual"])

            # 결과 통합 (컨텐츠 기준으로 중복 제거)
            unified_results = {}

            # 각 검색 방법의 결과 통합
            for method, results in search_results.items():
                for result in results:
                    content = result.get("content", "")
                    content_hash = hash(content[:200])  # 첫 200자로 해시

                    if content_hash not in unified_results:
                        unified_results[content_hash] = SearchResult(
                            content=content,
                            document_id=result.get("document_id", ""),
                            chunk_id=result.get("chunk_id", ""),
                            relevance_score=0.0,
                            search_method=method,
                            metadata=result.get("metadata", {})
                        )

                    # 각 방법별 점수 설정
                    search_result = unified_results[content_hash]
                    raw_score = result.get("raw_score", 0.0)

                    if method == "semantic":
                        search_result.semantic_score = raw_score
                    elif method == "keyword":
                        search_result.keyword_score = raw_score
                    elif method == "structure":
                        search_result.structure_score = result.get("structure_score", 0.0)

            # 최종 점수 계산 및 정렬
            final_results = []
            for search_result in unified_results.values():
                # 각 점수 정규화 (0-1 범위)
                normalized_semantic = min(search_result.semantic_score * 1.2, 1.0)
                normalized_keyword = min(search_result.keyword_score * 2.0, 1.0)
                normalized_structure = min(search_result.structure_score, 1.0)

                # 추가 점수 계산 (향후 확장용)
                recency_score = self._calculate_recency_score(search_result)
                popularity_score = self._calculate_popularity_score(search_result)

                # 가중 합계
                final_score = (
                    weights["semantic"] * normalized_semantic +
                    weights["keyword"] * normalized_keyword +
                    weights["structure"] * normalized_structure +
                    weights["recency"] * recency_score +
                    weights["popularity"] * popularity_score
                )

                # 쿼리 매칭 보너스
                if self._has_exact_match(query, search_result.content):
                    final_score *= 1.15

                search_result.relevance_score = final_score
                search_result.recency_score = recency_score
                search_result.popularity_score = popularity_score

                final_results.append(search_result)

            # 점수 기준 정렬
            final_results.sort(key=lambda x: x.relevance_score, reverse=True)

            return final_results

        except Exception as e:
            logging.error(f"결과 융합 중 오류: {e}")
            return []

    def _calculate_recency_score(self, result: SearchResult) -> float:
        """최신성 점수 계산 (향후 구현)"""
        # 메타데이터에서 날짜 정보 추출하여 계산
        # 현재는 기본값 반환
        return 0.5

    def _calculate_popularity_score(self, result: SearchResult) -> float:
        """인기도 점수 계산 (향후 구현)"""
        # 클릭 수, 참조 횟수 등을 기반으로 계산
        # 현재는 기본값 반환
        return 0.5

    def _has_exact_match(self, query: str, content: str) -> bool:
        """정확한 매칭 여부 확인"""
        query_words = set(self.text_processor.preprocess_korean_text(query).split())
        content_words = set(self.text_processor.preprocess_korean_text(content).split())

        if not query_words:
            return False

        match_ratio = len(query_words.intersection(content_words)) / len(query_words)
        return match_ratio >= 0.6  # 60% 이상 매칭

    def _apply_filters(self,
                      results: List[SearchResult],
                      filters: Dict[str, Any]) -> List[SearchResult]:
        """필터 적용"""
        filtered_results = []

        for result in results:
            should_include = True

            # 문서 유형 필터
            if "document_type" in filters:
                expected_type = filters["document_type"]
                actual_type = result.metadata.get("document_type", "")
                if actual_type != expected_type:
                    should_include = False

            # 점수 임계값 필터
            if "min_score" in filters:
                if result.relevance_score < filters["min_score"]:
                    should_include = False

            # 날짜 범위 필터 (향후 구현)
            # if "date_range" in filters:
            #     # 날짜 범위 체크

            if should_include:
                filtered_results.append(result)

        return filtered_results

    def _fallback_search(self, query: str, limit: int) -> List[SearchResult]:
        """기본 검색 (오류 시 fallback)"""
        try:
            logging.warning("기본 시맨틱 검색으로 fallback")
            results = self._semantic_search(query, limit)

            fallback_results = []
            for result in results:
                fallback_results.append(SearchResult(
                    content=result.get("content", ""),
                    document_id=result.get("document_id", ""),
                    chunk_id=result.get("chunk_id", ""),
                    relevance_score=result.get("relevance", 0.0),
                    search_method="fallback_semantic",
                    metadata=result.get("metadata", {}),
                    semantic_score=result.get("relevance", 0.0)
                ))

            return fallback_results

        except Exception as e:
            logging.error(f"Fallback 검색도 실패: {e}")
            return []