#!/bin/bash

echo "RAG 시스템 전체 데이터 초기화 시작..."
echo

echo "1. Docker 서비스 중지 중..."
docker-compose down -v

echo
echo "2. SQLite 데이터베이스 삭제 중..."
if [ -f "rag_chatbot.db" ]; then
    rm rag_chatbot.db
    echo "SQLite DB 삭제 완료"
else
    echo "SQLite DB 파일이 없습니다."
fi

echo
echo "3. 업로드 폴더 초기화 중..."
if [ -d "uploads" ]; then
    rm -rf uploads/*
    echo "업로드 폴더 초기화 완료"
else
    mkdir uploads
    echo "업로드 폴더 생성 완료"
fi

echo
echo "4. Docker 볼륨 삭제 중..."
docker volume rm rag-document-chatbot_neo4j_data 2>/dev/null || true
docker volume rm rag-document-chatbot_weaviate_data 2>/dev/null || true
docker volume rm rag-document-chatbot_ollama_data 2>/dev/null || true
echo "Docker 볼륨 삭제 완료"

echo
echo "5. Docker 서비스 재시작 중..."
docker-compose up -d

echo
echo "✅ 모든 데이터 초기화 완료!"
echo "시스템이 재시작되었습니다."
echo