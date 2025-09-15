from typing import Dict, List, Set
import spacy
import logging
import os
import json
from app.core.config import settings

class SynonymMapper:
    """동의어 및 용어 매핑 처리"""
    
    def __init__(self):
        # spaCy 모델 로드
        try:
            self.nlp = spacy.load("ko_core_news_lg")
            logging.info("spaCy 한국어 모델 로드 성공")
        except:
            logging.warning("spaCy 한국어 모델 로드 실패, 기본 모델 사용")
            self.nlp = spacy.load("en_core_web_sm")
        
        # 사전 정의된 동의어 사전 로드
        self.synonym_dict = self._load_synonym_dict()
        
        # Word2Vec 모델 로드 (선택적)
        self.word_vectors = None
        if settings.ENABLE_WORD_VECTORS:
            try:
                from gensim.models import KeyedVectors
                if os.path.exists(settings.WORD_VECTORS_PATH):
                    self.word_vectors = KeyedVectors.load(settings.WORD_VECTORS_PATH)
                    logging.info(f"단어 벡터 모델 로드 성공: {settings.WORD_VECTORS_PATH}")
                else:
                    logging.warning(f"단어 벡터 파일을 찾을 수 없음: {settings.WORD_VECTORS_PATH}")
            except Exception as e:
                logging.error(f"단어 벡터 모델 로드 실패: {str(e)}")
    
    def _load_synonym_dict(self) -> Dict[str, Set[str]]:
        """저장된 동의어 사전 로드"""
        try:
            if os.path.exists(settings.SYNONYM_DICT_PATH):
                with open(settings.SYNONYM_DICT_PATH, 'r', encoding='utf-8') as f:
                    raw_dict = json.load(f)
                
                # Dict[str, List[str]]를 Dict[str, Set[str]]로 변환
                logging.info(f"동의어 사전 로드 성공: {len(raw_dict)} 항목")
                return {k: set(v) for k, v in raw_dict.items()}
            else:
                logging.info("동의어 사전 파일이 없어 새로 생성합니다.")
                return {}
        except Exception as e:
            logging.error(f"동의어 사전 로드 실패: {str(e)}")
            return {}
    
    def save_synonym_dict(self):
        """동의어 사전 저장"""
        try:
            # Set을 JSON 직렬화 가능한 List로 변환
            save_dict = {k: list(v) for k, v in self.synonym_dict.items()}
            
            # 디렉토리 확인 및 생성
            os.makedirs(os.path.dirname(settings.SYNONYM_DICT_PATH), exist_ok=True)
            
            with open(settings.SYNONYM_DICT_PATH, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=2)
            
            logging.info(f"동의어 사전 저장 완료: {len(save_dict)} 항목")
        except Exception as e:
            logging.error(f"동의어 사전 저장 실패: {str(e)}")
    
    def detect_synonyms_from_document(self, text: str, threshold: float = 0.85) -> Dict[str, Set[str]]:
        """문서에서 자동으로 동의어 후보 탐지"""
        if not self.word_vectors:
            return {}
        
        try:
            # 명사 추출
            doc = self.nlp(text)
            nouns = [token.text for token in doc if token.pos_ == "NOUN" and len(token.text) > 1]
            
            # 중복 제거
            unique_nouns = list(set(nouns))
            logging.info(f"문서에서 추출된 고유 명사: {len(unique_nouns)}개")
            
            # 동의어 후보 탐지
            synonyms = {}
            for i, noun1 in enumerate(unique_nouns):
                for noun2 in unique_nouns[i+1:]:
                    try:
                        # Word2Vec 모델로 유사도 계산
                        if noun1 in self.word_vectors and noun2 in self.word_vectors:
                            similarity = self.word_vectors.similarity(noun1, noun2)
                            if similarity > threshold:
                                # 동의어 관계 저장
                                if noun1 not in synonyms:
                                    synonyms[noun1] = set()
                                synonyms[noun1].add(noun2)
                    except:
                        continue
            
            logging.info(f"탐지된 동의어 관계: {len(synonyms)}쌍")
            return synonyms
        except Exception as e:
            logging.error(f"동의어 탐지 중 오류: {str(e)}")
            return {}
    
    def add_synonym_pair(self, term: str, synonym: str):
        """동의어 쌍 수동 추가"""
        # 공백 제거
        term = term.strip()
        synonym = synonym.strip()
        
        if not term or not synonym or term == synonym:
            return False
        
        try:
            # 첫 번째 방향 관계 설정
            if term not in self.synonym_dict:
                self.synonym_dict[term] = set()
            self.synonym_dict[term].add(synonym)
            
            # 양방향 관계 설정
            if synonym not in self.synonym_dict:
                self.synonym_dict[synonym] = set()
            self.synonym_dict[synonym].add(term)
            
            # 변경사항 저장
            self.save_synonym_dict()
            logging.info(f"동의어 쌍 추가: {term} <-> {synonym}")
            return True
        except Exception as e:
            logging.error(f"동의어 쌍 추가 실패: {str(e)}")
            return False
    
    def get_synonyms(self, term: str) -> Set[str]:
        """용어의 모든 동의어 가져오기"""
        direct_synonyms = self.synonym_dict.get(term, set())
        
        # 재귀적으로 연결된 모든 동의어 수집 (선택적)
        all_synonyms = set(direct_synonyms)
        for syn in direct_synonyms:
            all_synonyms.update(self.synonym_dict.get(syn, set()))
        
        # 원래 용어 자신은 결과에서 제외
        if term in all_synonyms:
            all_synonyms.remove(term)
            
        return all_synonyms
    
    def expand_query(self, query: str) -> str:
        """쿼리 확장 - 동의어 포함"""
        doc = self.nlp(query)
        expanded_terms = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 1:
                synonyms = self.get_synonyms(token.text)
                if synonyms:
                    synonym_str = " OR ".join([token.text] + list(synonyms))
                    expanded_terms.append(f"({synonym_str})")
                else:
                    expanded_terms.append(token.text)
            else:
                expanded_terms.append(token.text)
        
        return " ".join(expanded_terms)