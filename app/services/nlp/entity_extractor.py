"""Entity extraction service for document processing.

This module provides comprehensive entity extraction capabilities using
multiple NLP models including spaCy and Hugging Face transformers.
"""

import logging
from typing import List, Dict, Any, Optional, Set

import spacy
from transformers import pipeline, Pipeline
from spacy.lang.en import English

from app.core.config import settings

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Advanced entity extraction from document chunks.

    This class provides multi-model entity extraction capabilities with
    fallback mechanisms and confidence scoring. It supports:

    - spaCy-based named entity recognition
    - Hugging Face transformer models
    - Custom entity type mapping
    - Confidence thresholding
    - Duplicate entity handling

    Attributes:
        nlp: spaCy language model instance
        ner_pipeline: Hugging Face NER pipeline
        ner_enabled: Flag indicating if HF NER is available
        confidence_threshold: Minimum confidence score for entities
        supported_entity_types: Set of supported entity types
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialize the entity extractor with model loading.

        Args:
            confidence_threshold: Minimum confidence score for entity acceptance
        """
        self.nlp: Optional[English] = None
        self.ner_pipeline: Optional[Pipeline] = None
        self.ner_enabled: bool = False
        self.confidence_threshold: float = confidence_threshold
        self.supported_entity_types: Set[str] = {
            "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
            "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
        }

        self._initialize_spacy_model()
        self._initialize_hf_model()

    def _initialize_spacy_model(self) -> None:
        """Initialize spaCy model with error handling."""
        try:
            logger.info("Loading spaCy English model...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy English model loaded successfully")
        except OSError as e:
            logger.warning(f"spaCy model loading failed: {e}")
            logger.info("Falling back to basic English model")
            try:
                self.nlp = English()
                logger.info("Basic English model loaded")
            except Exception as fallback_error:
                logger.error(f"Failed to load any spaCy model: {fallback_error}")
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy model: {e}")

    def _initialize_hf_model(self) -> None:
        """Initialize Hugging Face NER model with error handling."""
        try:
            logger.info("Loading Hugging Face NER model...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            self.ner_enabled = True
            logger.info("Hugging Face NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Hugging Face NER model loading failed: {e}")
            self.ner_enabled = False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 엔티티 추출"""
        entities = []
        
        # 1. spaCy로 기본 엔티티 추출
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "name": ent.text,
                        "type": ent.label_,
                        "score": 0.8,  
                        "source": "spacy"
                    })
                logging.info(f"spaCy로 {len(doc.ents)}개 엔티티 추출 성공")
            except Exception as e:
                logging.error(f"spaCy 엔티티 추출 실패: {str(e)}")
        
        # 2. Hugging Face 모델로 추가 엔티티 추출
        if self.ner_enabled:
            try:
                hf_entities = self.ner_pipeline(text)
                logging.info(f"Hugging Face로 {len(hf_entities)}개 엔티티 추출 성공")

                # 디버깅: 첫 번째 엔티티 구조 확인
                if hf_entities:
                    logging.debug(f"HF 엔티티 구조 예시: {hf_entities[0]}")

                # 결과 통합
                for ent in hf_entities:
                    try:
                        # 안전한 키 접근 - entity 또는 entity_group 키 확인
                        entity_type = ent.get("entity") or ent.get("entity_group") or ent.get("label", "UNKNOWN")
                        entity_name = ent.get("word") or ent.get("text", "")
                        entity_score = ent.get("score", 0.0)

                        if not entity_name:  # 엔티티 이름이 없으면 스킵
                            continue

                        # 중복 제거를 위한 확인
                        is_duplicate = False
                        for existing in entities:
                            if existing["name"] == entity_name:
                                is_duplicate = True
                                # 신뢰도가 더 높은 결과로 업데이트
                                if entity_score > existing["score"]:
                                    existing["score"] = entity_score
                                    existing["type"] = entity_type
                                    existing["source"] = "huggingface"
                                break

                        if not is_duplicate:
                            entities.append({
                                "name": entity_name,
                                "type": entity_type,
                                "score": entity_score,
                                "source": "huggingface"
                            })
                    except Exception as ent_error:
                        logging.warning(f"개별 엔티티 처리 실패: {ent_error}, 엔티티: {ent}")
                        continue

            except Exception as e:
                logging.error(f"Huggingface 엔티티 추출 실패: {str(e)}")
                logging.debug(f"Huggingface 오류 상세: {type(e).__name__}: {e}")
        
        # 엔티티가 하나도 없을 경우 빈 리스트 반환
        if not entities:
            logging.warning("추출된 엔티티가 없습니다.")
            return []
        
        # 3. 결과 정제 - 신뢰도 임계값 적용 및 정렬
        threshold = 0.5
        filtered_entities = [e for e in entities if e["score"] >= threshold]
        filtered_entities.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered_entities

    def debug_hf_pipeline(self, text: str) -> Dict[str, Any]:
        """Huggingface 파이프라인 디버깅을 위한 함수"""
        if not self.ner_enabled:
            return {"error": "HF NER pipeline not enabled"}

        try:
            raw_result = self.ner_pipeline(text)
            return {
                "success": True,
                "raw_result": raw_result,
                "result_type": type(raw_result).__name__,
                "first_entity_keys": list(raw_result[0].keys()) if raw_result else [],
                "count": len(raw_result)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }