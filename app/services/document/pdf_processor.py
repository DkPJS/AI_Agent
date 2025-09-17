from app.services.document.base_processor import BaseDocumentProcessor
from typing import List, Dict, Any
from pypdf import PdfReader
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
    
import re
import logging
import os
from datetime import datetime

class PDFProcessor(BaseDocumentProcessor):
    """PDF 파일 처리기"""
    
    def process(self, file_path: str) -> List[str]:
        """
        PDF 파일에서 향상된 텍스트 추출
        
        Args:
            file_path (str): PDF 파일 경로
            
        Returns:
            List[str]: 추출된 텍스트 청크 목록
        """
        try:
            # 파일 존재 및 크기 확인
            if not os.path.exists(file_path):
                logging.error(f"PDF 파일이 존재하지 않습니다: {file_path}")
                return [f"PDF 파일이 존재하지 않습니다: {os.path.basename(file_path)}"]
                
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.error(f"PDF 파일이 비어 있습니다: {file_path}")
                return [f"PDF 파일이 비어 있습니다: {os.path.basename(file_path)}"]
                
            logging.info(f"PDF 파일 처리 시작: {file_path}, 크기: {file_size} 바이트")
            
            # PDF 파일 열기
            try:
                reader = PdfReader(file_path)
                
                # 암호화 확인 및 처리 시도
                if reader.is_encrypted:
                    # 빈 암호로 시도
                    try:
                        reader.decrypt("")
                        logging.info("암호화된 PDF를 빈 암호로 복호화했습니다.")
                    except:
                        logging.error(f"PDF 파일이 암호화되어 있습니다: {file_path}")
                        return ["PDF 파일이 암호화되어 있어 처리할 수 없습니다."]
                
                # 페이지 수 확인
                if len(reader.pages) == 0:
                    logging.error(f"PDF 파일에 페이지가 없습니다: {file_path}")
                    return ["PDF 파일에 페이지가 없습니다."]
                
                logging.info(f"PDF 페이지 수: {len(reader.pages)}")
                
                # 메타데이터 추출
                chunks = []
                metadata_info = self._extract_pdf_metadata(reader)
                if metadata_info:
                    chunks.append(f"[문서 정보]\n{metadata_info}")
                
                # 북마크/목차 추출
                outline_info = self._extract_pdf_outline(reader)
                if outline_info:
                    chunks.append(f"[목차 정보]\n{outline_info}")
                
                # 페이지별 텍스트 추출 (구조화)
                page_texts = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_content = self._extract_page_content(page, i + 1)
                        if page_content:
                            page_texts.append(page_content)
                        else:
                            logging.warning(f"페이지 {i+1}에서 텍스트를 추출할 수 없습니다.")
                    except Exception as page_error:
                        logging.error(f"페이지 {i+1} 처리 중 오류 발생: {page_error}")
                
                # 전체 텍스트 결합
                if page_texts:
                    full_text = "\n\n".join(page_texts)
                    # 스마트 청킹
                    text_chunks = self._smart_chunk_pdf_text(full_text)
                    chunks.extend(text_chunks)
                else:
                    logging.error(f"PDF 파일에서 텍스트를 추출할 수 없습니다: {file_path}")
                    return ["PDF 파일에서 텍스트를 추출할 수 없습니다. 이미지나 스캔된 문서일 수 있습니다."]
                
                if not chunks:
                    logging.error(f"PDF 텍스트 청킹 실패: {file_path}")
                    return ["PDF 파일에서 텍스트를 추출했으나 청킹에 실패했습니다."]
                
                logging.info(f"PDF 처리 완료: {file_path}, {len(chunks)}개 청크 생성")
                return chunks
                
            except Exception as reader_error:
                logging.error(f"PDF 파일 읽기 중 오류 발생: {reader_error}")
                return [f"PDF 파일 읽기 중 오류가 발생했습니다: {str(reader_error)}"]
            
        except Exception as e:
            logging.error(f"PDF 처리 중 예외 발생: {e}")
            return [f"PDF 파일 처리 중 예외가 발생했습니다: {str(e)}"]
    
    def _extract_pdf_metadata(self, reader: PdfReader) -> str:
        """PDF 메타데이터 추출"""
        metadata_parts = []
        
        try:
            if reader.metadata:
                metadata = reader.metadata
                
                # 제목
                if '/Title' in metadata:
                    title = metadata['/Title']
                    if title:
                        metadata_parts.append(f"제목: {title}")
                
                # 작성자
                if '/Author' in metadata:
                    author = metadata['/Author']
                    if author:
                        metadata_parts.append(f"작성자: {author}")
                
                # 주제
                if '/Subject' in metadata:
                    subject = metadata['/Subject']
                    if subject:
                        metadata_parts.append(f"주제: {subject}")
                
                # 키워드
                if '/Keywords' in metadata:
                    keywords = metadata['/Keywords']
                    if keywords:
                        metadata_parts.append(f"키워드: {keywords}")
                
                # 생성 프로그램
                if '/Creator' in metadata:
                    creator = metadata['/Creator']
                    if creator:
                        metadata_parts.append(f"생성 프로그램: {creator}")
                
                # 생성일
                if '/CreationDate' in metadata:
                    try:
                        creation_date = metadata['/CreationDate']
                        if creation_date:
                            metadata_parts.append(f"생성일: {creation_date}")
                    except:
                        pass
                
                # 수정일
                if '/ModDate' in metadata:
                    try:
                        mod_date = metadata['/ModDate']
                        if mod_date:
                            metadata_parts.append(f"수정일: {mod_date}")
                    except:
                        pass
        
        except Exception as e:
            logging.warning(f"메타데이터 추출 중 오류: {e}")
        
        return "\n".join(metadata_parts) if metadata_parts else ""
    
    def _extract_pdf_outline(self, reader: PdfReader) -> str:
        """PDF 북마크/목차 추출"""
        outline_parts = []
        
        try:
            if reader.outline:
                outline_parts.append("목차:")
                self._process_outline_items(reader.outline, outline_parts, level=0)
        except Exception as e:
            logging.warning(f"목차 추출 중 오류: {e}")
        
        return "\n".join(outline_parts) if len(outline_parts) > 1 else ""
    
    def _process_outline_items(self, items, outline_parts, level=0):
        """북마크 항목 재귀 처리"""
        indent = "  " * level
        
        for item in items:
            if isinstance(item, list):
                self._process_outline_items(item, outline_parts, level + 1)
            else:
                try:
                    title = str(item.title) if hasattr(item, 'title') else str(item)
                    outline_parts.append(f"{indent}- {title}")
                except:
                    pass
    
    def _extract_page_content(self, page, page_num: int) -> str:
        """개별 페이지 내용 추출"""
        try:
            text = page.extract_text()
            
            if not text or not text.strip():
                return ""
            
            # 페이지 정보 추가
            content = f"[페이지 {page_num}]\n"
            
            # 텍스트 정리
            cleaned_text = self._clean_text(text)
            content += cleaned_text
            
            return content
            
        except Exception as e:
            logging.error(f"페이지 {page_num} 내용 추출 중 오류: {e}")
            return ""
    
    def _smart_chunk_pdf_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """PDF 구조를 고려한 스마트 청킹"""
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []
        
        chunks = []
        
        # 페이지별로 먼저 분할
        page_pattern = r'\[페이지 (\d+)\]'
        pages = re.split(page_pattern, text)
        
        current_chunk = ""
        page_numbers = re.findall(page_pattern, text)
        
        for i, page_content in enumerate(pages):
            if not page_content.strip():
                continue
            
            # 페이지 헤더 재구성 (첫 번째 요소는 헤더 없음)
            if i > 0 and (i-1) < len(page_numbers):
                page_header = f"[페이지 {page_numbers[i-1]}]\n"
                page_content = page_header + page_content
            
            # 청크 크기 확인
            if len(current_chunk) + len(page_content) > max_chunk_size:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 페이지 자체가 큰 경우 추가 분할 (페이지 헤더 유지)
                if len(page_content) > max_chunk_size:
                    page_chunks = self._chunk_with_page_info(page_content, max_chunk_size, overlap)
                    chunks.extend(page_chunks)
                    current_chunk = ""
                else:
                    current_chunk = page_content
            else:
                current_chunk += "\n\n" + page_content if current_chunk else page_content
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _chunk_with_page_info(self, text: str, max_chunk_size: int, overlap: int) -> List[str]:
        """페이지 정보를 유지하면서 텍스트 청킹"""
        # 페이지 헤더 추출
        page_header_match = re.match(r'(\[페이지 \d+\]\n)', text)
        page_header = page_header_match.group(1) if page_header_match else ""
        
        # 헤더를 제외한 본문
        content = text[len(page_header):] if page_header else text
        
        # 본문을 청킹 (기본 크기 기반 분할)
        content_chunks = self._basic_text_chunk(content, max_chunk_size - len(page_header), overlap)

        # 각 청크에 페이지 헤더 추가
        return [page_header + chunk for chunk in content_chunks if chunk.strip()]

    def _basic_text_chunk(self, text: str, max_chunk_size: int, overlap: int) -> List[str]:
        """기본 크기 기반 텍스트 청킹 (문자열 반환)"""
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chunk_size, len(text))

            # 문장이나 단락 경계에서 끊기
            if end < len(text):
                # 단락 경계 찾기
                paragraph_boundary = text.rfind('\n\n', start, end)

                # 문장 경계 찾기
                sentence_boundary = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )

                if paragraph_boundary > start + max_chunk_size // 2:
                    end = paragraph_boundary + 2
                elif sentence_boundary > start + max_chunk_size // 2:
                    end = sentence_boundary + 2

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            # 다음 시작 위치 (오버랩 고려)
            start = end - overlap if end < len(text) else len(text)

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        추출된 텍스트 정리
        
        Args:
            text (str): 정리할 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 개행 정리
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # 다중 개행 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()