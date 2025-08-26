# === Step 1: 질의응답 기본 개념 ===
# 수강생 과제: TODO 부분을 완성하여 Extractive QA의 기본 원리를 이해하세요.

import logging
import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # === 1. 모델과 토크나이저 로드 ===
    
    # TODO: 사전학습된 한국어 QA 모델을 지정하세요
    # 힌트: "monologg/koelectra-base-v3-finetuned-korquad" 또는 로컬 체크포인트 경로
    pretrained = # TODO: 완성하세요

    print(f"Loading model from {pretrained}")
    
    # TODO: AutoTokenizer를 로드하세요
    # 힌트: AutoTokenizer.from_pretrained() 사용
    tokenizer = # TODO: 완성하세요
    
    # TODO: AutoModelForQuestionAnswering을 로드하세요
    # 힌트: AutoModelForQuestionAnswering.from_pretrained() 사용
    model = # TODO: 완성하세요
    
    # TODO: 모델을 평가 모드로 설정하세요
    # 힌트: model.eval() 사용
    # TODO: 완성하세요

    # === 2. 예제 데이터 준비 ===
    
    # TODO: 질의응답에 사용할 컨텍스트(지문)를 작성하세요
    # 힌트: 대한민국에 대한 정보가 포함된 문단 작성
    context = """# TODO: 대한민국에 대한 정보를 포함한 컨텍스트를 작성하세요
    예시: 대한민국은 동아시아의 한반도 남부에 위치한 나라이다...
    """

    # TODO: 컨텍스트를 바탕으로 답할 수 있는 질문들을 작성하세요
    # 힌트: 컨텍스트에서 명확히 답을 찾을 수 있는 질문들
    questions = [
        # TODO: 질문 리스트를 완성하세요
        # 예시: "대한민국의 수도는?", "대한민국의 국화는?"
    ]

    # === 3. 질의응답 함수 구현 ===
    
    def answer_question(question: str, context: str) -> str:
        """
        주어진 질문과 컨텍스트를 바탕으로 답변을 생성하는 함수
        
        Args:
            question: 질문 문자열
            context: 컨텍스트 문자열
            
        Returns:
            str: 예측된 답변
        """
        # TODO: 질문과 컨텍스트를 토크나이즈하세요
        # 힌트: tokenizer.encode_plus() 사용, return_tensors="pt", truncation=True, padding=True
        inputs = # TODO: 완성하세요
        
        # TODO: 모델 추론을 수행하세요 (gradient 계산 없이)
        # 힌트: torch.no_grad() 컨텍스트 내에서 model(**inputs) 호출
        with torch.no_grad():
            outputs = # TODO: 완성하세요

        # TODO: 시작과 끝 위치의 logits를 추출하세요
        # 힌트: outputs.start_logits, outputs.end_logits
        start_logits = # TODO: 완성하세요
        end_logits = # TODO: 완성하세요

        # TODO: 가장 높은 확률을 가진 시작과 끝 인덱스를 찾으세요
        # 힌트: torch.argmax() 사용
        start_index = # TODO: 완성하세요
        end_index = # TODO: 완성하세요

        # TODO: 예측된 답변 토큰들을 추출하세요
        # 힌트: inputs["input_ids"][0, start_index:end_index + 1] 슬라이싱
        predict_answer_tokens = # TODO: 완성하세요
        
        # TODO: 토큰들을 텍스트로 디코딩하세요
        # 힌트: tokenizer.decode() 사용
        answer = # TODO: 완성하세요

        return answer

    # === 4. 질문별 답변 생성 및 출력 ===
    
    print("=== 질의응답 결과 ===")
    for question in questions:
        # TODO: 각 질문에 대해 답변을 생성하고 출력하세요
        answer = # TODO: answer_question 함수 호출
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 50)

"""
학습 목표:
1. Extractive QA의 기본 원리 이해
2. BERT 기반 QA 모델의 구조 파악
3. Start/End Logits의 개념 학습
4. 토큰 위치를 실제 답변으로 변환하는 과정 체험

핵심 개념:

1. Extractive QA:
   - 주어진 컨텍스트에서 답변을 추출하는 방식
   - 새로운 텍스트를 생성하지 않고 기존 텍스트의 일부를 선택
   - Reading Comprehension의 대표적 형태

2. AutoModelForQuestionAnswering:
   - BERT 등의 인코더 모델 위에 QA 헤드 추가
   - 각 토큰 위치에 대해 시작/끝 확률 계산
   - [CLS] 토큰을 통해 "답변 없음" 표현 가능

3. Start/End Logits:
   - start_logits: 각 토큰이 답변 시작 위치일 확률
   - end_logits: 각 토큰이 답변 끝 위치일 확률
   - 가장 높은 확률의 시작/끝 조합으로 답변 결정

4. 토크나이제이션과 디코딩:
   - 질문+컨텍스트를 하나의 시퀀스로 결합
   - [CLS] 질문 [SEP] 컨텍스트 [SEP] 형태
   - 토큰 인덱스를 통해 원본 텍스트의 답변 추출

예상 결과:
- 컨텍스트에서 정확한 답변 추출
- 토큰 단위의 정밀한 위치 예측
- 자연스러운 한국어 답변 생성

주의사항:
- 컨텍스트에 답변이 반드시 포함되어야 함
- 긴 컨텍스트는 토큰 길이 제한으로 잘릴 수 있음
- 서브워드 토크나이제이션으로 인한 불완전한 단어 생성 가능
"""
