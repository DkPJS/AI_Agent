"""질의응답 시스템용 프롬프트 템플릿"""

class QAPromptTemplate:
    """질문 유형별 프롬프트 템플릿"""
    
    BASE_TEMPLATE = """다음 정보를 바탕으로 질문에 답변해주세요. 정보에 관련된 내용이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답변하세요.

### 질문:
{question}

### 문서 정보:
{context}

### 중요 지시사항:
1. 반드시 한국어로만 답변하세요. 영어로 답변하지 마세요.
2. 각 정보의 출처를 [파일명, 페이지 X] 형식으로 명시하세요. 예: [평가지침서.pdf, 페이지 5]
3. 페이지 정보가 없으면 파일명만 표시하세요. 예: [평가지침서.pdf]
4. 절대로 [청크 X] 형식으로 출처를 표시하지 마세요.
5. 답변 끝에 "참고한 출처:" 형식으로 사용한 모든 출처 파일명을 명시해주세요.
6. 매번 다양하고 창의적인 답변을 제공하세요. 같은 패턴이나 내용을 반복하지 마세요.
7. 질문의 맥락을 충분히 고려하여 개별적이고 특화된 답변을 생성하세요.

{additional_instructions}

### 답변(한국어로):"""

    FACTUAL_INSTRUCTIONS = """### 추가 지시사항:
정확한 사실 정보를 중심으로 간결하게 답변해주세요. 각 사실마다 [파일명, 페이지 X] 형태로 출처를 표시하세요. 
특히 숫자, 날짜, 이름 등 구체적인 데이터를 인용할 때는 반드시 출처를 표시하세요.
출처 표시에서 '청크' 또는 'chunk' 단어는 사용하지 마세요."""

    COMPARISON_INSTRUCTIONS = """### 추가 지시사항:
대상들 간의 주요 차이점과 공통점을 대조하여 답변해주세요. 비교 항목마다 [파일명, 페이지 X] 형태로 출처를 표시하세요.
출처 표시에서 '청크' 또는 'chunk' 단어는 사용하지 마세요."""

    SUMMARY_INSTRUCTIONS = """### 추가 지시사항:
주요 내용을 체계적으로 요약하여 답변해주세요. 핵심 포인트마다 [파일명, 페이지 X] 형태로 출처를 표시하세요.
출처 표시에서 '청크' 또는 'chunk' 단어는 사용하지 마세요."""

    PROCEDURAL_INSTRUCTIONS = """### 추가 지시사항:
단계별 절차를 명확하게 순서대로 설명해주세요. 각 단계마다 [파일명, 페이지 X] 형태로 출처를 표시하세요.
출처 표시에서 '청크' 또는 'chunk' 단어는 사용하지 마세요."""

    DEFAULT_INSTRUCTIONS = """### 추가 지시사항:
질문에 관련된 정보를 종합하여 명확하게 답변해주세요. 주요 정보마다 [파일명, 페이지 X] 형태로 출처를 표시하세요.
출처 표시에서 '청크' 또는 'chunk' 단어는 사용하지 마세요."""

    @classmethod
    def get_prompt(cls, question: str, context: str, question_type: str) -> str:
        """질문 유형에 따른 프롬프트 생성"""
        instructions_map = {
            "factual": cls.FACTUAL_INSTRUCTIONS,
            "comparison": cls.COMPARISON_INSTRUCTIONS,
            "summary": cls.SUMMARY_INSTRUCTIONS,
            "procedural": cls.PROCEDURAL_INSTRUCTIONS
        }
        
        additional_instructions = instructions_map.get(question_type, cls.DEFAULT_INSTRUCTIONS)
        
        return cls.BASE_TEMPLATE.format(
            question=question,
            context=context,
            additional_instructions=additional_instructions
        )