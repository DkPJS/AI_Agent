from app.services.document.base_processor import BaseDocumentProcessor
from typing import List
import logging
import os


class TextProcessor(BaseDocumentProcessor):
    """텍스트 파일 처리기"""

    def process(self, file_path: str) -> List[str]:
        """
        텍스트 파일에서 내용 추출 및 청킹

        Args:
            file_path (str): 텍스트 파일 경로

        Returns:
            List[str]: 추출된 텍스트 청크 목록
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                logging.error(f"텍스트 파일이 존재하지 않습니다: {file_path}")
                return [f"텍스트 파일이 존재하지 않습니다: {os.path.basename(file_path)}"]

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.error(f"텍스트 파일이 비어 있습니다: {file_path}")
                return [f"텍스트 파일이 비어 있습니다: {os.path.basename(file_path)}"]

            logging.info(f"텍스트 파일 처리 시작: {file_path}, 크기: {file_size} 바이트")

            # 텍스트 파일 읽기 (다양한 인코딩 시도)
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
            content = ""

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logging.info(f"파일을 {encoding} 인코딩으로 성공적으로 읽었습니다")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logging.warning(f"인코딩 {encoding}으로 읽기 실패: {e}")
                    continue

            if not content:
                logging.error(f"텍스트 파일 읽기 실패: {file_path}")
                return ["텍스트 파일을 읽을 수 없습니다. 인코딩 문제일 수 있습니다."]

            # 의미 기반 청킹 적용
            chunks = self.chunk_text(
                content,
                max_chunk_size=1500,
                use_semantic=True,
                document_type="general"
            )

            if not chunks:
                logging.error(f"텍스트 파일 청킹 실패: {file_path}")
                return ["텍스트 파일에서 내용을 추출했으나 청킹에 실패했습니다."]

            # 청크가 딕셔너리 형태라면 content만 추출
            if chunks and isinstance(chunks[0], dict):
                text_chunks = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
            else:
                text_chunks = chunks

            logging.info(f"텍스트 파일 처리 완료: {file_path}, {len(text_chunks)}개 청크 생성")
            return text_chunks

        except Exception as e:
            logging.error(f"텍스트 파일 처리 중 예외 발생: {e}")
            return [f"텍스트 파일 처리 중 예외가 발생했습니다: {str(e)}"]