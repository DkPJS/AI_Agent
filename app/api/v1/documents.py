from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from typing import List, Optional
from sqlalchemy.orm import Session
import os
import uuid
import shutil
import logging
import json

from app.infrastructure.database.sql_client import get_db
from app.infrastructure.database.neo4j_client import Neo4jClient
from app.models.document import Document, DocumentChunk
from app.services.document.base_processor import BaseDocumentProcessor, DocumentStructureExtractor
from app.rag.retrieval.embedder import DocumentEmbedder
from app.services.nlp.entity_extractor import EntityExtractor
from app.services.nlp.synonym_mapper import SynonymMapper
from app.core import config

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(""),
    auto_detect_synonyms: str = Form("false"),  # 불리언 대신 문자열로 받기
    analyze_structure: str = Form("true"),  # 문서 구조 분석 활성화/비활성화
    db: Session = Depends(get_db)
):
    """문서 업로드, 처리 및 그래프 저장"""
    try:
        # 디버깅 로그 추가
        logging.info(f"업로드 시작: 파일명={file.filename}, content_type={file.content_type}")
        logging.info(f"파라미터: description={description}, auto_detect_synonyms={auto_detect_synonyms}, analyze_structure={analyze_structure}")
        
        # 파일이 비어있는지 확인
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파일이 제공되지 않았습니다."
            )
            
        # 파일 크기 확인
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="파일이 비어있습니다."
            )
        
        # 포인터 초기화 - 읽은 후 반드시 초기화
        await file.seek(0)

        # 문자열을 불리언으로 변환
        auto_detect = auto_detect_synonyms.lower() in ("true", "1", "t", "yes")
        do_analyze_structure = analyze_structure.lower() in ("true", "1", "t", "yes")
        
        # 1. 파일 검증 및 저장
        content_type = file.content_type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # 지원하는 확장자에 '.txt' 추가
        supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.hwp', '.txt']
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
            )
        
        document_id = str(uuid.uuid4())
        upload_dir = config.settings.UPLOAD_DIR
        file_path = os.path.join(upload_dir, f"{document_id}{file_extension}")
        
        # 업로드 디렉토리 확인 및 생성
        os.makedirs(upload_dir, exist_ok=True)
        
        # 이미 읽어둔 파일 내용을 사용하여 파일 저장
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        logging.info(f"파일 저장 완료: {file_path}")
        
        # 2. DB 문서 생성
        document = Document(
            id=document_id,
            filename=file.filename,
            file_path=file_path,
            content_type=content_type,
            size=len(file_content),
            description=description
        )
        
        db.add(document)
        db.commit()
        logging.info(f"문서 DB 저장 완료: id={document_id}")
        
        # 3. 문서 처리 (텍스트 추출 및 스마트 청킹)
        try:
            processor = BaseDocumentProcessor.get_processor(file_path)
            raw_chunks = processor.process(file_path)

            # 개선된 의미 기반 청킹 적용
            from app.services.document.semantic_chunker import SemanticChunker
            semantic_chunker = SemanticChunker()

            # 원본 청크들을 합쳐서 전체 텍스트 구성
            full_text = "\n\n".join(raw_chunks)

            # 문서 유형에 따른 스마트 청킹 (구조 분석 완료 후 실행)
            chunks = []
            
            # 처리 결과 검증 (빈 리스트거나 오류 메시지만 있는 경우)
            if not raw_chunks:
                # 청크가 없으면
                logging.error(f"문서에서 텍스트를 추출할 수 없습니다: {file_path}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="문서에서 텍스트를 추출할 수 없습니다."
                )

            # 에러 메시지가 있는지 확인 (첫 번째 청크가 오류 메시지인 경우)
            if len(raw_chunks) == 1 and any(raw_chunks[0].startswith(prefix) for prefix in ["PDF 파일", "DOCX 파일", "HWP 파일", "Excel 파일"]):
                logging.error(f"문서 처리 오류: {raw_chunks[0]}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=raw_chunks[0]
                )

            logging.info(f"문서 텍스트 추출 완료: {len(raw_chunks)}개 원본 청크")
            
            # 3.1 문서 구조 분석 및 분류
            document_structure = {}
            document_type = "general"  # 기본값을 영어로 변경 (SemanticChunker 호환)

            if do_analyze_structure:
                try:
                    # 문서 구조 추출
                    structure_extractor = DocumentStructureExtractor()
                    document_structure = structure_extractor.extract_structure(full_text)

                    # 문서 유형 분류
                    document_type = processor.classify_document_type(full_text, document_structure)

                    logging.info(f"문서 구조 분석 완료: {len(document_structure.get('headers', []))}개 헤더, " +
                                f"{len(document_structure.get('sections', []))}개 섹션 발견")
                    logging.info(f"문서 유형 분류 결과: {document_type}")

                    # 메타데이터 업데이트
                    if 'metadata' in document_structure and document_structure['metadata']:
                        # 문서 설명이 비어있을 경우에만 자동 생성된 제목 사용
                        if not document.description and 'title' in document_structure['metadata']:
                            document.description = document_structure['metadata']['title']
                            db.add(document)
                            db.commit()
                            logging.info(f"문서 메타데이터 업데이트: title={document_structure['metadata'].get('title')}")
                except Exception as structure_error:
                    logging.error(f"문서 구조 분석 중 오류: {str(structure_error)}")
                    # 구조 분석 실패해도 계속 진행

            # 3.2 스마트 청킹 실행
            try:
                # 문서 특성에 맞는 최적 청킹 수행
                chunked_results = semantic_chunker.chunk_document(full_text, document_type)

                # 청크 내용만 추출
                chunks = [chunk_data["content"] for chunk_data in chunked_results]

                logging.info(f"스마트 청킹 완료: {len(raw_chunks)}개 원본 → {len(chunks)}개 최적화된 청크")

                # 청킹 품질 정보 로깅
                if chunked_results:
                    avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
                    logging.info(f"청킹 품질: 평균 크기 {avg_chunk_size:.0f}자, 방법: 적응형 의미 기반")

            except Exception as chunking_error:
                logging.error(f"스마트 청킹 실패, 원본 청킹 사용: {str(chunking_error)}")
                # 스마트 청킹 실패 시 원본 청킹 사용
                chunks = raw_chunks
                    
        except Exception as e:
            logging.error(f"문서 처리 중 오류: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}"
            )
        
        # 4. 청크 저장
        entity_extractor = EntityExtractor()
        embedder = DocumentEmbedder()
        synonym_mapper = SynonymMapper() if auto_detect else None
        
        db_chunks = []
        
        # Neo4j 클라이언트 처리 - try/finally 구문으로 수정하여 항상 닫히도록 함
        neo4j_client = None
        try:
            # Neo4j 클라이언트 초기화
            neo4j_client = Neo4jClient()
            logging.info("Neo4j 클라이언트 초기화 완료")
            
            # 문서 노드 생성 (Neo4j)
            document_data = {
                "id": document.id,
                "filename": document.filename,
                "content_type": document.content_type,
                "size": document.size,
                "description": document.description,
                "upload_date": document.upload_date,
                "document_type": document_type  # 문서 유형 추가
            }
            
            # 구조 정보가 있으면 추가
            if document_structure:
                # JSON으로 직렬화하여 Neo4j에 저장 가능한 형태로 변환
                if 'metadata' in document_structure:
                    document_data['metadata'] = json.dumps(document_structure['metadata'], ensure_ascii=False)
                
                # 헤더 수와 섹션 수 정보 추가
                if 'headers' in document_structure:
                    document_data['header_count'] = len(document_structure['headers'])
                
                if 'sections' in document_structure:
                    document_data['section_count'] = len(document_structure['sections'])
            
            try:
                neo4j_client.create_document(document_data)
                logging.info(f"Neo4j 문서 노드 생성 완료: id={document_id}, type={document_type}")
            except Exception as e:
                logging.error(f"Neo4j 문서 노드 생성 오류: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Neo4j 문서 노드 생성 오류: {str(e)}"
                )
            
            # 동의어 탐지용 전체 텍스트 (옵션 활성화된 경우)
            full_text = "\n\n".join(chunks) if auto_detect else ""
            
            # 5. 각 청크 처리
            for idx, chunk_text in enumerate(chunks):
                try:
                    # 최소 길이 확인
                    if len(chunk_text.strip()) < 10:  # 너무 짧은 청크 건너뛰기
                        continue
                        
                    # 청크 ID 생성
                    chunk_id = str(uuid.uuid4())
                    
                    # A. SQL DB에 청크 저장
                    chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        content=chunk_text,
                        chunk_index=idx
                    )
                    db.add(chunk)
                    db_chunks.append(chunk)
                    
                    # 청크가 어떤 섹션에 속하는지 확인 (구조 분석이 된 경우)
                    section_info = None
                    if do_analyze_structure and 'sections' in document_structure:
                        # 청크가 어느 섹션에 속하는지 결정
                        for section in document_structure['sections']:
                            if chunk_text in section['content']:
                                section_info = {
                                    'title': section['title'],
                                    'level': section['level'],
                                    'header_id': section.get('header_id', '')
                                }
                                break
                    
                    # B. 임베딩 생성 및 벡터 DB 저장
                    metadata = {
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "chunk_index": idx,
                        "document_type": document_type
                    }
                    
                    # 섹션 정보가 있으면 메타데이터에 추가
                    if section_info:
                        metadata.update({
                            "section_title": section_info['title'],
                            "section_level": section_info['level']
                        })
                    
                    weaviate_id = embedder.store_chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=chunk_text,
                        metadata=metadata
                    )
                    chunk.embedding_id = weaviate_id
                    
                    # C. Neo4j에 청크 노드 생성
                    chunk_data = {
                        "id": chunk_id,
                        "content": chunk_text,
                        "chunk_index": idx,
                        "embedding_id": weaviate_id
                    }
                    
                    # 섹션 정보 추가
                    if section_info:
                        chunk_data["section_title"] = section_info['title']
                        chunk_data["section_level"] = section_info['level']
                    
                    neo4j_client.create_chunk(chunk_data, document_id)
                    
                    # D. 엔티티 추출 및 그래프에 추가
                    entities = entity_extractor.extract_entities(chunk_text)
                    neo4j_client.extract_and_link_entities(chunk_id, entities)
                    
                except Exception as chunk_error:
                    logging.error(f"청크 {idx} 처리 중 오류: {str(chunk_error)}")
                    # 개별 청크 오류는 건너뛰고 계속 진행
            
            logging.info(f"청크 처리 완료: {len(db_chunks)}개 청크 저장됨")
            
            # 섹션 간 관계 생성 (구조 분석이 된 경우)
            if do_analyze_structure and 'sections' in document_structure and len(document_structure['sections']) > 1:
                try:
                    # 섹션 계층 구조를 Neo4j에 저장
                    for i, section in enumerate(document_structure['sections']):
                        # 현재 섹션 레벨
                        current_level = section['level']
                        
                        # 현재 섹션보다 상위 레벨의 가장 가까운 섹션 찾기 (부모-자식 관계)
                        if i > 0:
                            for j in range(i-1, -1, -1):
                                parent_section = document_structure['sections'][j]
                                if parent_section['level'] < current_level:
                                    # 부모-자식 관계 생성
                                    neo4j_client.create_section_relationship(
                                        document_id,
                                        parent_section['title'],
                                        section['title'],
                                        "CONTAINS"
                                    )
                                    break
                    
                    logging.info("섹션 계층 구조 관계 생성 완료")
                except Exception as section_error:
                    logging.error(f"섹션 관계 생성 중 오류: {str(section_error)}")
                    # 섹션 관계 오류는 무시하고 계속 진행
            
            # 7. 동의어 탐지 및 저장 (옵션 활성화된 경우)
            if auto_detect and synonym_mapper:
                try:
                    detected_synonyms = synonym_mapper.detect_synonyms_from_document(full_text)
                    
                    # 탐지된 동의어 저장
                    for term, synonyms in detected_synonyms.items():
                        for synonym in synonyms:
                            synonym_mapper.add_synonym_pair(term, synonym)
                            
                            # Neo4j에도 동의어 관계 저장
                            neo4j_client.add_synonym_relation(term, synonym)
                    
                    logging.info(f"동의어 탐지 완료: {len(detected_synonyms)}개 용어")
                except Exception as synonym_error:
                    logging.error(f"동의어 탐지 중 오류: {str(synonym_error)}")
                    # 동의어 오류는 무시하고 계속 진행
            
        except Exception as neo4j_error:
            logging.error(f"Neo4j 작업 중 오류: {str(neo4j_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Neo4j 작업 중 오류가 발생했습니다: {str(neo4j_error)}"
            )
        finally:
            # Neo4j 클라이언트 명시적으로 닫기
            if neo4j_client:
                try:
                    neo4j_client.close()
                    logging.info("Neo4j 클라이언트 연결 닫힘")
                except Exception as close_error:
                    logging.error(f"Neo4j 클라이언트 닫기 오류: {str(close_error)}")
        
        # 6. 변경사항 저장
        db.commit()
        logging.info("SQL DB 변경사항 커밋 완료")
        
        # 8. 응답 반환
        response_data = {
            "status": "success",
            "document_id": document_id,
            "filename": document.filename,
            "size": document.size,
            "content_type": document.content_type,
            "chunk_count": len(db_chunks),
            "message": "문서가 성공적으로 업로드되고 처리되었습니다."
        }
        
        # 구조 분석 결과가 있으면 응답에 추가
        if do_analyze_structure:
            response_data.update({
                "document_type": document_type,
                "structure_info": {
                    "header_count": len(document_structure.get('headers', [])),
                    "section_count": len(document_structure.get('sections', [])),
                    "metadata": document_structure.get('metadata', {})
                }
            })
        
        return response_data
        
    except HTTPException as e:
        logging.error(f"문서 업로드 중 HTTP 오류: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"문서 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/upload/fast")
async def upload_document_fast(
    file: UploadFile = File(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    """빠른 문서 업로드 (기능 최소화)"""
    try:
        logging.info(f"빠른 업로드 시작: 파일명={file.filename}")

        # 기본 설정 (속도 우선)
        auto_detect = False  # 동의어 탐지 비활성화
        do_analyze_structure = False  # 구조 분석 비활성화

        # 1. 파일 검증 및 저장
        content_type = file.content_type
        file_extension = os.path.splitext(file.filename)[1].lower()

        supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.hwp', '.txt']
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
            )

        document_id = str(uuid.uuid4())
        upload_dir = config.settings.UPLOAD_DIR
        file_path = os.path.join(upload_dir, f"{document_id}{file_extension}")

        os.makedirs(upload_dir, exist_ok=True)

        # 파일 저장
        file_content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        logging.info(f"파일 저장 완료: {file_path}")

        # 2. SQL DB에 문서 정보 저장
        document = Document(
            id=document_id,
            filename=file.filename,
            content_type=content_type,
            size=len(file_content),
            description=description or f"{file.filename} (빠른 업로드)"
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        logging.info(f"문서 정보 저장 완료: {document_id}")

        # 3. 문서 처리 (기본 청킹만)
        processor = BaseDocumentProcessor.get_processor(file_path)
        chunks = processor.process(file_path)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="문서에서 텍스트를 추출할 수 없습니다."
            )

        logging.info(f"텍스트 추출 완료: {len(chunks)}개 청크")

        # 4. 청크 저장 (벡터 DB만, Neo4j 건너뛰기)
        embedder = DocumentEmbedder()
        db_chunks = []

        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 10:
                continue

            chunk_id = str(uuid.uuid4())

            # SQL DB에 청크 저장
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                chunk_index=idx
            )
            db.add(chunk)
            db_chunks.append(chunk)

            # 벡터 DB 저장 (메타데이터 최소화)
            metadata = {
                "filename": document.filename,
                "chunk_index": idx,
                "document_type": "일반문서"  # 고정값으로 빠른 처리
            }

            try:
                success = embedder.embed_and_store(
                    text=chunk_text,
                    metadata=metadata,
                    chunk_id=chunk_id
                )

                if not success:
                    logging.warning(f"청크 {idx} 임베딩 실패, 계속 진행")

            except Exception as embed_error:
                logging.error(f"청크 {idx} 임베딩 중 오류: {embed_error}")
                continue

        db.commit()

        response_data = {
            "message": "문서가 성공적으로 업로드되었습니다 (빠른 모드)",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_count": len(db_chunks),
            "document_type": "일반문서",
            "mode": "fast"
        }

        logging.info(f"빠른 업로드 완료: {document_id}, {len(db_chunks)}개 청크")
        return response_data

    except HTTPException as e:
        logging.error(f"빠른 업로드 중 HTTP 오류: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"빠른 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"빠른 업로드 중 오류가 발생했습니다: {str(e)}"
        )