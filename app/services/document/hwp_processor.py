from app.processors.base_processor import BaseDocumentProcessor
from typing import List, Dict, Any, Optional
import olefile
import re
import logging
import struct
from datetime import datetime

class HwpProcessor(BaseDocumentProcessor):
    """HWP 파일 처리기"""
    
    def process(self, file_path: str) -> List[str]:
        """
        HWP 파일에서 향상된 텍스트 추출
        
        Args:
            file_path (str): HWP 파일 경로
            
        Returns:
            List[str]: 추출된 텍스트 청크 목록
        """
        try:
            if not olefile.isOleFile(file_path):
                logging.warning(f"HWP 파일이 아니거나 읽을 수 없는 형식입니다: {file_path}")
                return ["HWP 파일이 아니거나 읽을 수 없는 형식입니다."]
            
            ole = olefile.OleFile(file_path)
            chunks = []
            
            try:
                # HWP 파일 구조 분석
                structure_info = self._analyze_hwp_structure(ole)
                if structure_info:
                    chunks.append(f"[HWP 파일 구조]\n{structure_info}")
                
                # 문서 정보 추출
                doc_info = self._extract_hwp_document_info(ole)
                if doc_info:
                    chunks.append(f"[문서 정보]\n{doc_info}")
                
                # 텍스트 추출 (다양한 방법 시도)
                text_chunks = self._extract_hwp_text_advanced(ole)
                
                if text_chunks:
                    chunks.extend(text_chunks)
                else:
                    logging.warning(f"HWP 파일에서 텍스트를 추출할 수 없습니다: {file_path}")
                    chunks.append("HWP 파일에서 텍스트를 추출할 수 없습니다.")
                    
            finally:
                ole.close()
                
            if not chunks:
                return ["HWP 파일 처리 결과가 없습니다."]
                
            logging.info(f"HWP 처리 완료: {file_path}, {len(chunks)}개 청크 생성")
            return chunks
                
        except Exception as e:
            logging.error(f"HWP 처리 중 오류 발생: {e}")
            return [f"HWP 파일 처리 중 오류가 발생했습니다: {str(e)}"]
    
    def _analyze_hwp_structure(self, ole) -> str:
        """HWP 파일 구조 분석"""
        structure_parts = []
        
        try:
            # OLE 스트림 목록
            streams = ole.listdir()
            structure_parts.append(f"총 스트림 수: {len(streams)}")
            
            # 주요 스트림 확인
            main_streams = []
            for stream_path in streams:
                stream_name = '/'.join(stream_path) if isinstance(stream_path, list) else str(stream_path)
                main_streams.append(stream_name)
            
            if main_streams:
                structure_parts.append(f"주요 스트림: {', '.join(main_streams[:10])}")
                if len(main_streams) > 10:
                    structure_parts.append(f"... 외 {len(main_streams) - 10}개")
            
            # HWP 버전 추정
            if ole.exists('FileHeader'):
                try:
                    header_data = ole.openstream('FileHeader').read()
                    if len(header_data) >= 4:
                        version_info = struct.unpack('<I', header_data[:4])[0]
                        structure_parts.append(f"HWP 버전 정보: {version_info}")
                except Exception as e:
                    logging.warning(f"버전 정보 추출 실패: {e}")
            
        except Exception as e:
            logging.warning(f"HWP 구조 분석 중 오류: {e}")
        
        return "\n".join(structure_parts) if structure_parts else ""
    
    def _extract_hwp_document_info(self, ole) -> str:
        """HWP 문서 정보 추출"""
        doc_info_parts = []
        
        try:
            # DocInfo 스트림에서 문서 정보 추출
            if ole.exists('DocInfo'):
                try:
                    docinfo_data = ole.openstream('DocInfo').read()
                    # 간단한 메타데이터 추출 시도
                    doc_info_parts.append(f"문서 정보 크기: {len(docinfo_data)} 바이트")
                    
                    # 텍스트가 포함된 부분 찾기 (UTF-16LE 디코딩 시도)
                    try:
                        decoded_info = docinfo_data.decode('utf-16le', errors='ignore')
                        # 의미 있는 텍스트 추출
                        meaningful_text = re.findall(r'[\u0020-\u007E\uAC00-\uD7AF]{3,}', decoded_info)
                        if meaningful_text:
                            doc_info_parts.append(f"추출된 정보: {', '.join(meaningful_text[:5])}")
                    except:
                        pass
                    
                except Exception as e:
                    logging.warning(f"DocInfo 스트림 처리 실패: {e}")
            
            # PrvText 기본 정보
            if ole.exists('PrvText'):
                try:
                    prvtext_data = ole.openstream('PrvText').read()
                    doc_info_parts.append(f"미리보기 텍스트 크기: {len(prvtext_data)} 바이트")
                except Exception as e:
                    logging.warning(f"PrvText 정보 추출 실패: {e}")
            
            # SummaryInformation 스트림 (표준 OLE 문서 속성)
            if ole.exists('\x05SummaryInformation'):
                try:
                    summary_data = ole.openstream('\x05SummaryInformation').read()
                    # 간단한 속성 정보 추출
                    doc_info_parts.append(f"요약 정보 크기: {len(summary_data)} 바이트")
                except Exception as e:
                    logging.warning(f"SummaryInformation 처리 실패: {e}")
            
        except Exception as e:
            logging.warning(f"HWP 문서 정보 추출 중 오류: {e}")
        
        return "\n".join(doc_info_parts) if doc_info_parts else ""
    
    def _extract_hwp_text_advanced(self, ole) -> List[str]:
        """향상된 HWP 텍스트 추출"""
        text_chunks = []
        
        try:
            # 방법 1: PrvText 스트림 (미리보기 텍스트)
            prvtext_content = self._extract_from_prvtext(ole)
            if prvtext_content:
                text_chunks.extend(prvtext_content)
            
            # 방법 2: BodyText 스트림들 시도
            bodytext_content = self._extract_from_bodytext_streams(ole)
            if bodytext_content:
                text_chunks.extend(bodytext_content)
            
            # 방법 3: 기타 텍스트 포함 스트림들
            other_content = self._extract_from_other_streams(ole)
            if other_content:
                text_chunks.extend(other_content)
            
        except Exception as e:
            logging.error(f"HWP 텍스트 추출 중 오류: {e}")
        
        return text_chunks
    
    def _extract_from_prvtext(self, ole) -> List[str]:
        """PrvText 스트림에서 텍스트 추출"""
        chunks = []
        
        try:
            if ole.exists('PrvText'):
                text_data = ole.openstream('PrvText').read()
                
                # UTF-16LE 디코딩
                text = text_data.decode('utf-16-le', errors='ignore')
                
                # 텍스트 정리
                clean_text = self._clean_hwp_text(text)
                
                if clean_text and clean_text.strip():
                    # 청크로 분할
                    text_chunks = self._smart_chunk_hwp_text(clean_text)
                    
                    # 각 청크에 출처 표시
                    for i, chunk in enumerate(text_chunks):
                        if chunk.strip():
                            chunks.append(f"[HWP 본문 {i+1}]\n{chunk}")
                            
        except Exception as e:
            logging.warning(f"PrvText 추출 중 오류: {e}")
        
        return chunks
    
    def _extract_from_bodytext_streams(self, ole) -> List[str]:
        """BodyText 관련 스트림에서 텍스트 추출"""
        chunks = []
        
        try:
            # BodyText 디렉토리나 관련 스트림 찾기
            streams = ole.listdir()
            bodytext_streams = []
            
            for stream_path in streams:
                stream_name = '/'.join(stream_path) if isinstance(stream_path, list) else str(stream_path)
                if 'bodytext' in stream_name.lower() or 'section' in stream_name.lower():
                    bodytext_streams.append(stream_name)
            
            for stream_name in bodytext_streams[:5]:  # 처음 5개만 시도
                try:
                    stream_data = ole.openstream(stream_name).read()
                    
                    # UTF-16LE 디코딩 시도
                    try:
                        text = stream_data.decode('utf-16le', errors='ignore')
                        clean_text = self._clean_hwp_text(text)
                        
                        if clean_text and len(clean_text.strip()) > 50:  # 의미있는 텍스트만
                            chunks.append(f"[{stream_name}]\n{clean_text[:1000]}")  # 처음 1000자만
                    except:
                        # 다른 인코딩 시도
                        try:
                            text = stream_data.decode('euc-kr', errors='ignore')
                            clean_text = self._clean_hwp_text(text)
                            
                            if clean_text and len(clean_text.strip()) > 50:
                                chunks.append(f"[{stream_name}]\n{clean_text[:1000]}")
                        except:
                            pass
                            
                except Exception as e:
                    logging.warning(f"BodyText 스트림 '{stream_name}' 처리 실패: {e}")
                    
        except Exception as e:
            logging.warning(f"BodyText 스트림 추출 중 오류: {e}")
        
        return chunks
    
    def _extract_from_other_streams(self, ole) -> List[str]:
        """기타 스트림에서 텍스트 추출"""
        chunks = []
        
        try:
            # 텍스트가 있을 만한 다른 스트림들 시도
            candidate_streams = ['DocOptions', 'Scripts', 'BinData', 'DocInfo']
            
            for stream_name in candidate_streams:
                if ole.exists(stream_name):
                    try:
                        stream_data = ole.openstream(stream_name).read()
                        
                        # UTF-16LE로 텍스트 추출 시도
                        try:
                            text = stream_data.decode('utf-16le', errors='ignore')
                            # 한글이나 영문이 포함된 의미있는 텍스트만 추출
                            meaningful_parts = re.findall(r'[\u0020-\u007E\uAC00-\uD7AF]{10,}', text)
                            
                            if meaningful_parts:
                                combined_text = ' '.join(meaningful_parts[:10])  # 처음 10개 부분만
                                if len(combined_text.strip()) > 20:
                                    chunks.append(f"[{stream_name} 추출 텍스트]\n{combined_text[:500]}")
                        except:
                            pass
                            
                    except Exception as e:
                        logging.warning(f"기타 스트림 '{stream_name}' 처리 실패: {e}")
                        
        except Exception as e:
            logging.warning(f"기타 스트림 추출 중 오류: {e}")
        
        return chunks
    
    def _smart_chunk_hwp_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """HWP 텍스트의 스마트 청킹"""
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []
        
        chunks = []
        
        # HWP 특성상 단락 구분이 명확하므로 단락 단위로 먼저 분할
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 현재 청크에 단락을 추가했을 때 크기 확인
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 단락이 너무 크면 문장 단위로 분할
                if len(paragraph) > max_chunk_size:
                    sentences = re.split(r'[.!?]\s+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 2 > max_chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            temp_chunk += ". " + sentence if temp_chunk else sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _clean_hwp_text(self, text: str) -> str:
        """
        HWP 텍스트 정리
        
        Args:
            text (str): 정리할 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        # 불필요한 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # Windows 스타일 개행문자 통일
        text = text.replace('\r\n', '\n')
        
        # 불필요한 개행 정리
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # 다중 개행 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()