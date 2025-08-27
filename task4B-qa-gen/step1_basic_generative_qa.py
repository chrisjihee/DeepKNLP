# === Step 1: Generative QA 기본 개념 ===
# 수강생 과제: TODO 부분을 완성하여 Generative QA의 기본 원리를 이해하세요.
# Extractive QA와 다른 점: 컨텍스트에서 답변을 추출하는 것이 아니라 새로운 텍스트를 생성!

import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # === 1. Seq2Seq QA 모델과 토크나이저 로드 ===
    
    # TODO: 사전학습된 한국어 T5 기반 QA 모델을 지정하세요
    # 힌트: "paust/pko-t5-base-finetuned-korquad" 또는 로컬 체크포인트 경로
    pretrained = # TODO: 완성하세요

    print(f"Loading model from {pretrained}")
    
    # TODO: AutoTokenizer를 로드하세요
    # 힌트: AutoTokenizer.from_pretrained() 사용
    tokenizer = # TODO: 완성하세요
    
    # TODO: AutoModelForSeq2SeqLM을 로드하세요 (BERT가 아닌 T5!)
    # 힌트: AutoModelForSeq2SeqLM.from_pretrained() 사용
    model = # TODO: 완성하세요
    
    # TODO: 모델을 평가 모드로 설정하세요
    # 힌트: model.eval() 사용
    # TODO: 완성하세요

    # === 2. 예제 데이터 준비 ===
    
    # TODO: 질의응답에 사용할 컨텍스트(지문)를 작성하세요
    # 힌트: Extractive QA와 동일한 컨텍스트 사용 가능
    context = """# TODO: 대한민국에 대한 정보를 포함한 컨텍스트를 작성하세요
    예시: 대한민국은 동아시아의 한반도 남부에 위치한 나라이다...
    """

    # TODO: 컨텍스트를 바탕으로 답할 수 있는 질문들을 작성하세요
    # 힌트: Extractive QA와 동일하지만, 더 자유로운 형태의 답변 가능
    questions = [
        # TODO: 질문 리스트를 완성하세요
        # 예시: "대한민국의 수도는?", "대한민국에 대해 설명해주세요"
    ]

    # === 3. Generative QA 함수 구현 ===
    
    def answer_question_generative(question: str, context: str, max_length: int = 50, num_beams: int = 5) -> str:
        """
        T5 모델을 사용하여 질문에 대한 답변을 생성하는 함수
        
        Args:
            question: 질문 문자열
            context: 컨텍스트 문자열
            max_length: 생성할 최대 답변 길이
            num_beams: Beam Search에서 사용할 beam 수
            
        Returns:
            str: 생성된 답변
        """
        # TODO: T5 형식의 입력 텍스트 구성
        # 힌트: "question: {질문} context: {컨텍스트}" 형태
        input_text = # TODO: 완성하세요
        
        # TODO: 입력 텍스트를 토크나이즈하세요
        # 힌트: tokenizer() 사용, return_tensors="pt", truncation=True, padding=True
        inputs = # TODO: 완성하세요
        
        # TODO: 텍스트 생성 (gradient 계산 없이)
        with torch.no_grad():
            # TODO: model.generate() 메소드로 답변 생성
            # 힌트: input_ids, attention_mask, max_length, num_beams 사용
            output_ids = model.generate(
                # TODO: 필요한 인수들을 완성하세요
            )

        # TODO: 생성된 토큰 ID들을 텍스트로 디코딩
        # 힌트: tokenizer.decode() 사용, skip_special_tokens=True
        answer = # TODO: 완성하세요

        return answer

    # === 4. Extractive vs Generative 비교 ===
    
    print("=== Generative QA vs Extractive QA 비교 ===")
    print("Extractive QA: 컨텍스트에서 기존 텍스트 '추출'")
    print("Generative QA: 새로운 텍스트를 '생성'하여 답변")
    print()

    # === 5. 질문별 답변 생성 및 출력 ===
    
    print("=== Generative QA 결과 ===")
    for question in questions:
        # TODO: 각 질문에 대해 답변을 생성하고 출력하세요
        answer = # TODO: answer_question_generative 함수 호출
        
        print(f"Question: {question}")
        print(f"Generated Answer: {answer}")
        print("-" * 70)

    # === 6. 다양한 생성 파라미터 실험 ===
    
    print("\n=== 생성 파라미터 실험 ===")
    test_question = questions[0] if questions else "대한민국의 수도는?"
    
    # TODO: Beam Search 크기 변경 실험
    print("1. Beam Search 크기 비교:")
    for num_beams in [1, 3, 5]:
        # TODO: 각기 다른 beam 수로 답변 생성
        answer = # TODO: answer_question_generative 호출 (num_beams 변경)
        print(f"   Beams={num_beams}: {answer}")

    # TODO: 최대 길이 변경 실험
    print("\n2. 최대 답변 길이 비교:")
    for max_len in [20, 50, 100]:
        # TODO: 각기 다른 최대 길이로 답변 생성
        answer = # TODO: answer_question_generative 호출 (max_length 변경)
        print(f"   Max_len={max_len}: {answer}")

    # === 7. 창의적 질문 실험 ===
    
    print("\n=== 창의적 질문 (Generative QA의 장점) ===")
    
    # TODO: Extractive QA로는 답하기 어려운 창의적 질문들
    creative_questions = [
        # TODO: 요약, 설명, 비교 등의 질문 추가
        # 예시: "대한민국에 대해 간략히 설명해주세요"
        # "대한민국의 주요 특징은 무엇인가요?"
    ]
    
    for question in creative_questions:
        # TODO: 창의적 질문에 대한 답변 생성
        answer = # TODO: answer_question_generative 호출
        print(f"Creative Q: {question}")
        print(f"Generated A: {answer}")
        print("-" * 70)

"""
학습 목표:
1. Generative QA의 기본 원리와 Extractive QA와의 차이점 이해
2. T5 모델의 Seq2Seq 구조와 텍스트 생성 방식 학습
3. Beam Search와 생성 파라미터의 효과 체험
4. Generative QA의 장점과 한계 파악

핵심 개념:

1. Extractive vs Generative QA:
   - Extractive: 컨텍스트에서 기존 텍스트 스팬 추출
   - Generative: 새로운 텍스트를 생성하여 답변
   - Generative는 요약, 설명, 추론 등 더 복잡한 답변 가능

2. T5 (Text-to-Text Transfer Transformer):
   - 모든 NLP 태스크를 텍스트 생성으로 통일
   - Encoder-Decoder 구조 (BERT는 Encoder만)
   - "question: Q context: C" → 답변 텍스트 생성

3. Seq2Seq QA 입력 형식:
   - 표준화된 프롬프트 형식 사용
   - "question: 질문 context: 컨텍스트"
   - 모델이 이 형식에 맞춰 fine-tuning됨

4. 텍스트 생성 파라미터:
   - max_length: 생성할 최대 토큰 수
   - num_beams: Beam Search 폭 (높을수록 품질↑, 속도↓)
   - temperature: 창의성 조절 (일반적으로 1.0)
   - top_k, top_p: 토큰 선택 범위 제한

5. Beam Search:
   - 여러 경로를 동시에 탐색하여 최적 시퀀스 찾기
   - num_beams=1: Greedy Search
   - num_beams>1: 더 나은 품질, 더 많은 계산

6. 장점과 단점:
   장점:
   - 요약, 설명, 추론 등 복잡한 답변 가능
   - 컨텍스트에 없는 정보도 생성 가능 (상식 활용)
   - 자연스럽고 유창한 답변

   단점:
   - 사실과 다른 정보 생성 가능 (Hallucination)
   - 계산량이 많음 (생성 과정 필요)
   - 정확성 보장이 어려움

실무 적용 시나리오:
- 교육: 설명이 필요한 질문 답변
- 고객지원: 복합적 문제에 대한 안내
- 연구: 논문 요약, 핵심 내용 설명
- 창작: 스토리텔링, 창의적 답변

주의사항:
- 생성된 답변의 사실성 검증 필요
- 컨텍스트 범위를 벗어난 답변 가능
- 편향된 정보 생성 위험성
- 실시간 서비스에서 응답 시간 고려
"""
