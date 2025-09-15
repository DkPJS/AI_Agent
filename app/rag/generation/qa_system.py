from typing import Dict, Any
import httpx
import logging
from app.rag.retrieval.question_analyzer import QuestionAnalyzer
from app.rag.retrieval.search_strategy import SearchStrategy
from app.rag.generation.prompt_templates import QAPromptTemplate
from app.core.config import settings
import re

class AdvancedQASystem:
    """그래프 기반 고급 질의응답 시스템"""
    
    def __init__(self):
        self.question_analyzer = QuestionAnalyzer()
        self.search_strategy = SearchStrategy()
        self.prompt_template = QAPromptTemplate()
        self.llm_api_url = settings.LLM_API_URL
        
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """사용자 질문에 답변"""
        # 1. 질문 분석
        question_type, focus_entities = self.question_analyzer.analyze_question(question)
        
        # 2. 검색 전략 선택 및 실행
        context, sources = await self.search_strategy.execute_search(question, question_type, focus_entities)
        
        if not context:
            return {
                "answer": "죄송합니다. 질문에 관련된 정보를 찾을 수 없습니다.",
                "sources": [],
                "question_type": question_type
            }
        
        # 3. 프롬프트 생성 및 응답 생성
        tagged_context = self._tag_context_sources(context)
        prompt = self.prompt_template.get_prompt(question, tagged_context, question_type)
        answer = await self._generate_answer(prompt)
        
        # 4. 결과 반환
        return {
            "answer": answer,
            "sources": sources,
            "question_type": question_type
        }
    
    def _tag_context_sources(self, context: str) -> str:
        """컨텍스트에 소스 태그 추가"""
        chunks = context.split("\n\n")
        tagged_context = ""
        
        filename_pattern = r'\[파일:\s*([^\]]+)\]'
        
        for chunk in chunks:
            if chunk.strip():
                filename = "문서"
                filename_match = re.search(filename_pattern, chunk)
                if filename_match:
                    filename = filename_match.group(1)
                
                page_info = ""
                page_match = re.search(r'\[페이지\s*(\d+)\]', chunk)
                if page_match:
                    page_info = f", 페이지 {page_match.group(1)}"
                
                source_tag = f"[출처: {filename}{page_info}]"
                tagged_context += f"{source_tag}\n{chunk}\n\n"
        
        return tagged_context
    
    
    async def _generate_answer(self, prompt: str) -> str:
        """LLM을 사용하여 응답 생성"""
        try:
            import time
            import random
            
            # 답변 다양성을 위한 동적 파라미터 설정
            current_time = int(time.time())
            random.seed(current_time)
            
            payload = {
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "temperature": 0.7,  # 다양성을 위해 온도를 높임
                "top_p": 0.9,       # nucleus sampling 추가
                "top_k": 40,        # top-k sampling 추가
                "repeat_penalty": 1.2,  # 반복 방지
                "max_tokens": 1024,
                "stream": False,
                "system": f"너는 한국어로만 대답하는 AI 비서입니다. 답변은 항상 한국어로만 작성하세요. 매번 다양하고 창의적인 답변을 제공하세요. 현재 시각: {current_time}"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.llm_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"LLM API 오류: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"LLM 호출 중 오류 발생: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"