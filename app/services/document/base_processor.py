from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

class BaseDocumentProcessor(ABC):
    """문서 처리기 기본 클래스"""
    
    @abstractmethod
    def process(self, file_path: str) -> List[str]:
        """
        파일을 처리하여 텍스트 청크 목록을 반환
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            List[str]: 추출된 텍스트 청크 목록
        """
        pass
    
    @classmethod
    def get_processor(cls, file_path: str) -> 'BaseDocumentProcessor':
        """
        파일 확장자에 적합한 프로세서 반환
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            BaseDocumentProcessor: 적절한 프로세서 인스턴스
        """
        from app.processors.pdf_processor import PDFProcessor
        from app.processors.docx_processor import DocxProcessor
        from app.processors.excel_processor import ExcelProcessor
        from app.processors.hwp_processor import HwpProcessor
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return PDFProcessor()
        elif ext == '.docx':
            return DocxProcessor()
        elif ext in ['.xlsx', '.xls']:
            return ExcelProcessor()
        elif ext == '.hwp':
            return HwpProcessor()
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    
    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        긴 텍스트를 오버랩이 있는 청크로 분할
        
        Args:
            text (str): 분할할 텍스트
            max_chunk_size (int): 최대 청크 크기 (문자 단위)
            overlap (int): 청크 간 오버랩 크기
            
        Returns:
            List[str]: 분할된 텍스트 청크 목록
        """
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # 문장 또는 단락 경계에서 끊기 위해 적절한 위치 찾기
            if end < len(text):
                # 단락 경계 찾기 (더 우선)
                paragraph_boundary = text.rfind('\n\n', start, end)
                
                # 문장 경계 찾기
                sentence_boundary = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                
                if paragraph_boundary > start + max_chunk_size // 2:
                    end = paragraph_boundary + 2  # '\n\n' 포함
                elif sentence_boundary > start + max_chunk_size // 2:
                    end = sentence_boundary + 2  # '. ' 포함
            
            chunks.append(text[start:end].strip())
            
            # 다음 시작 위치 (오버랩 고려)
            start = end - overlap if end < len(text) else len(text)
        
        return [chunk for chunk in chunks if chunk]  # 빈 청크 제거
    
    def classify_document_type(self, text: str, structure: Dict) -> str:
        """
        문서 유형 자동 분류
        
        Args:
            text (str): 문서 텍스트
            structure (Dict): 문서 구조 정보
            
        Returns:
            str: 예측된 문서 유형 (보고서, 계약서, 매뉴얼 등)
        """
        # 특징 추출 (헤더, 포맷 패턴, 키워드 등)
        features = self._extract_document_features(text, structure)
        
        # 사전 훈련된 분류기로 문서 유형 예측
        doc_type = self._document_classifier(features)
        return doc_type
    
    def _extract_document_features(self, text: str, structure: Dict) -> Dict[str, Any]:
        """
        문서 특징 추출 (문서 분류를 위한 특징)
        
        Args:
            text (str): 문서 텍스트
            structure (Dict): 문서 구조 정보
            
        Returns:
            Dict: 추출된 문서 특징
        """
        # 텍스트 기반 특징
        features = {}
        
        # 텍스트 길이 및 기본 통계
        features['text_length'] = len(text)
        features['avg_sentence_length'] = np.mean([len(s.strip()) for s in text.split('.') if s.strip()])
        
        # 키워드 빈도 (자주 나오는 중요 단어)
        words = re.findall(r'\b\w+\b', text.lower())
        features['word_count'] = len(words)
        
        # 숫자 포함 비율
        features['number_ratio'] = len(re.findall(r'\d+', text)) / (features['word_count'] + 1)
        
        # 구조 기반 특징
        if 'headers' in structure:
            features['header_count'] = len(structure['headers'])
        
        if 'sections' in structure:
            features['section_count'] = len(structure['sections'])
        
        # 문서 유형별 키워드 검색
        features['is_contract'] = any(kw in text.lower() for kw in ['계약', '계약서', '합의', '당사자', '위약금'])
        features['is_report'] = any(kw in text.lower() for kw in ['보고서', '결과', '분석', '요약', '조사'])
        features['is_manual'] = any(kw in text.lower() for kw in ['매뉴얼', '지침', '설명서', '사용법', '단계'])
        
        return features
    
    def _document_classifier(self, features: Dict[str, Any]) -> str:
        """
        특징에 기반한 간단한 규칙 기반 문서 유형 분류기
        (향후 머신러닝 모델로 대체 가능)
        
        Args:
            features (Dict): 문서 특징 정보
            
        Returns:
            str: 문서 유형
        """
        # 간단한 규칙 기반 분류
        if features.get('is_contract', False):
            return '계약서'
        elif features.get('is_report', False):
            return '보고서'
        elif features.get('is_manual', False):
            return '매뉴얼'
        elif features.get('section_count', 0) > 5:
            return '보고서'
        else:
            return '일반문서'


class DocumentStructureExtractor:
    """문서 구조 추출 클래스"""
    
    def extract_structure(self, document: str) -> Dict:
        """
        문서 섹션, 헤더, 목차 등 구조 자동 추출
        
        Args:
            document (str): 문서 텍스트
            
        Returns:
            Dict: 문서 구조 정보 (헤더, 섹션, 메타데이터)
        """
        # 정규표현식, 레이아웃 분석 등 통해 구조 추출
        headers = self._extract_headers(document)
        sections = self._segment_sections(document, headers)
        metadata = self._extract_metadata(document)
        
        return {
            "headers": headers,
            "sections": sections,
            "metadata": metadata
        }
    
    def _extract_headers(self, document: str) -> List[Dict[str, Any]]:
        """
        문서에서 헤더 추출
        
        Args:
            document (str): 문서 텍스트
            
        Returns:
            List[Dict]: 추출된 헤더 목록 (텍스트, 레벨, 위치 등)
        """
        headers = []
        
        # 헤더 패턴 정의 (기본적인 형식 찾기)
        header_patterns = [
            r'^제(\d+)장\s+(.+)$',    # 제1장 머리말
            r'^(\d+)\.\s+(.+)$',      # 1. 서론
            r'^(\d+\.\d+)\s+(.+)$',   # 1.1 배경
            r'^(\d+\.\d+\.\d+)\s+(.+)$' # 1.1.1 목적
        ]
        
        lines = document.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    header_id = match.group(1)
                    header_text = match.group(2).strip()
                    
                    # 헤더 레벨 추정
                    if '장' in header_id:
                        level = 1
                    else:
                        level = len(header_id.split('.'))
                    
                    headers.append({
                        'id': header_id,
                        'text': header_text,
                        'level': level,
                        'line_number': i
                    })
                    break
                    
        return headers
    
    def _segment_sections(self, document: str, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        헤더 기반으로 문서를 섹션으로 분할
        
        Args:
            document (str): 문서 텍스트
            headers (List[Dict]): 추출된 헤더 목록
            
        Returns:
            List[Dict]: 분할된 섹션 목록 (제목, 내용, 레벨 등)
        """
        if not headers:
            return [{'title': '전체 문서', 'content': document, 'level': 0}]
            
        sections = []
        lines = document.split('\n')
        
        for i in range(len(headers)):
            start_line = headers[i]['line_number']
            end_line = len(lines) - 1
            
            # 현재 헤더 다음에 오는 헤더가 있으면 해당 위치까지
            if i < len(headers) - 1:
                end_line = headers[i+1]['line_number'] - 1
            
            section_content = '\n'.join(lines[start_line+1:end_line+1]).strip()
            
            sections.append({
                'title': headers[i]['text'],
                'content': section_content,
                'level': headers[i]['level'],
                'header_id': headers[i]['id']
            })
            
        return sections
    
    def _extract_metadata(self, document: str) -> Dict[str, Any]:
        """
        문서에서 메타데이터 추출 (날짜, 작성자 등)
        
        Args:
            document (str): 문서 텍스트
            
        Returns:
            Dict: 추출된 메타데이터
        """
        metadata = {}
        
        # 날짜 패턴 찾기
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, document)
            if match:
                year, month, day = match.groups()
                metadata['date'] = f"{year}-{month:0>2}-{day:0>2}"
                break
                
        # 작성자 추정 (예: "작성자: 홍길동" 패턴)
        author_match = re.search(r'작성자\s*[:|]\s*([\w\s]+)', document)
        if author_match:
            metadata['author'] = author_match.group(1).strip()
            
        # 문서 제목 추정 (첫 번째 큰 헤더나 PDF 제목)
        title_match = re.search(r'^(.+)$', document.split('\n')[0])
        if title_match:
            metadata['title'] = title_match.group(1).strip()
            
        return metadata

# 호환성을 위한 별칭
BaseProcessor = BaseDocumentProcessor