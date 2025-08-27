# === Step 2: Seq2Seq QA 데이터 전처리 ===
# 수강생 과제: TODO 부분을 완성하여 Generative QA의 전처리 과정을 이해하세요.
# Extractive QA와의 큰 차이점: 복잡한 위치 매핑 대신 간단한 텍스트 변환!

import logging
import torch
from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # === 1. 토크나이저 로드 ===
    
    # TODO: 사전학습된 T5 토크나이저를 로드하세요
    # 힌트: "paust/pko-t5-base-finetuned-korquad" 사용
    pretrained = # TODO: 완성하세요
    tokenizer = # TODO: 완성하세요

    # === 2. 예제 QA 데이터 준비 (KorQuAD 형식) ===
    
    # TODO: Seq2Seq QA 학습 데이터 형태로 예제를 구성하세요
    # 힌트: Extractive QA와 동일한 구조이지만 사용 방식이 다름
    raw_data = [
        {
            "id": "qa_001",
            "question": # TODO: 질문 작성,
            "context": # TODO: 컨텍스트 작성,
            "answers": {
                "answer_start": [# TODO: 시작 위치 (Generative에서는 실제로 사용 안함)],
                "text": [# TODO: 정답 텍스트 (이것만 중요!)]
            }
        },
        # TODO: 더 많은 예제 추가 (최소 3개)
        # 힌트: 요약형 답변, 설명형 답변도 포함해보세요
    ]

    # 데이터셋 생성
    dataset = Dataset.from_list(raw_data)

    # === 3. Seq2Seq QA 전처리 설정 ===
    
    # TODO: Seq2Seq QA 전처리에 필요한 파라미터들을 설정하세요
    max_seq_length = # TODO: 입력 최대 길이 (예: 512)
    max_answer_length = # TODO: 답변 최대 길이 (예: 50)
    pad_to_max_length = # TODO: 최대 길이로 패딩 여부 (True/False)

    # === 4. 입력 형식 변환 함수 구현 ===
    
    def preprocess_squad_batch(examples, question_column: str = "question", context_column: str = "context", answer_column: str = "answers"):
        """
        QA 데이터를 T5 입력 형식으로 변환하는 함수
        
        Args:
            examples: 원본 QA 데이터 배치
            question_column: 질문 컬럼명
            context_column: 컨텍스트 컬럼명  
            answer_column: 답변 컬럼명
            
        Returns:
            tuple: (입력 텍스트 리스트, 타겟 텍스트 리스트)
        """
        # TODO: 각 컬럼에서 데이터 추출
        questions = # TODO: examples[question_column]
        contexts = # TODO: examples[context_column] 
        answers = # TODO: examples[answer_column]

        def generate_input(question: str, context: str) -> str:
            """T5 형식의 입력 텍스트 생성"""
            # TODO: "question: 질문내용 context: 컨텍스트내용" 형식으로 구성
            # 힌트: question.lstrip(), context.lstrip()로 앞의 공백 제거
            return # TODO: 완성하세요

        # TODO: 모든 질문-컨텍스트 쌍을 T5 입력 형식으로 변환
        inputs = # TODO: [generate_input() for question, context in zip()]
        
        # TODO: 답변 텍스트 추출 (첫 번째 답변만 사용)
        # 힌트: answer["text"][0] if len(answer["text"]) > 0 else ""
        targets = # TODO: 완성하세요

        return inputs, targets

    # === 5. 학습용 전처리 함수 구현 ===
    
    def preprocess_function(examples):
        """
        학습용 Seq2Seq QA 데이터 전처리 함수
        
        Args:
            examples: 원본 QA 데이터 배치
            
        Returns:
            dict: 토크나이즈된 입력과 라벨
        """
        # TODO: T5 입력 형식으로 변환
        inputs, targets = # TODO: preprocess_squad_batch() 호출

        # TODO: 입력 텍스트들을 토크나이즈
        # 힌트: tokenizer() 사용, max_length, padding, truncation 설정
        model_inputs = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
            # inputs, max_length, padding, truncation
        )
        
        # TODO: 타겟(답변) 텍스트들을 토크나이즈
        # 힌트: tokenizer(text_target=targets, ...) 사용
        labels = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
            # text_target=targets, max_length, padding, truncation
        )

        # TODO: 패딩 토큰을 -100으로 변경 (loss 계산에서 무시하기 위함)
        if pad_to_max_length:
            # 힌트: tokenizer.pad_token_id를 -100으로 변경
            labels["input_ids"] = [
                # TODO: 완성하세요
                # [(l if l != tokenizer.pad_token_id else -100) for l in label]
            ]

        # TODO: 모델 입력에 라벨 추가
        model_inputs["labels"] = # TODO: labels["input_ids"]

        return model_inputs

    # === 6. 추론용 전처리 함수 구현 ===
    
    def preprocess_validation_function(examples):
        """
        추론용 Seq2Seq QA 데이터 전처리 함수
        
        Args:
            examples: 원본 QA 데이터 배치
            
        Returns:
            dict: 토크나이즈된 입력과 라벨 (후처리를 위한 정보 포함)
        """
        # TODO: T5 입력 형식으로 변환
        inputs, targets = # TODO: preprocess_squad_batch() 호출

        # TODO: 입력 텍스트들을 토크나이즈 (Extractive QA와 달리 간단!)
        # 힌트: return_overflowing_tokens, return_offsets_mapping 불필요
        model_inputs = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
        )
        
        # TODO: 타겟 텍스트들을 토크나이즈
        labels = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
        )

        # TODO: 패딩 토큰을 -100으로 변경
        if pad_to_max_length:
            labels["input_ids"] = [
                # TODO: 완성하세요
            ]

        # TODO: 후처리를 위한 example_id 저장
        model_inputs["example_id"] = []
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            # TODO: example_id 추가
            model_inputs["example_id"].append(examples["id"][i])
            labels_out.append(labels["input_ids"][i])

        model_inputs["labels"] = labels_out
        return model_inputs

    # === 7. 전처리 실행 및 결과 확인 ===
    
    print("=== Seq2Seq QA vs Extractive QA 전처리 비교 ===")
    print("Extractive QA: 복잡한 offset mapping, 위치 라벨링")
    print("Seq2Seq QA: 단순한 텍스트 변환, 입력-출력 매핑")
    print()
    
    print("=== 학습용 전처리 결과 ===")
    # TODO: 학습용 전처리 함수 실행
    train_features = # TODO: preprocess_function(dataset)
    
    print(f"원본 샘플 수: {len(dataset)}")
    print(f"전처리 후 features 수: {len(train_features['input_ids'])}")
    
    # 첫 번째 feature 상세 정보 출력
    for i in range(min(2, len(train_features['input_ids']))):
        print(f"\n--- Feature {i} ---")
        
        # TODO: 입력 텍스트 디코딩
        input_text = # TODO: tokenizer.decode() 사용
        print(f"Input text: {input_text}")
        
        # TODO: 라벨(타겟) 텍스트 디코딩
        # 힌트: -100을 pad_token_id로 변경 후 디코딩
        label_ids = train_features['labels'][i]
        label_ids_clean = [l if l != -100 else tokenizer.pad_token_id for l in label_ids]
        target_text = # TODO: tokenizer.decode() 사용
        print(f"Target text: {target_text}")
        
        print(f"Input length: {len(train_features['input_ids'][i])}")
        print(f"Target length: {len([l for l in label_ids if l != -100])}")

    print("\n=== 추론용 전처리 결과 ===")
    # TODO: 추론용 전처리 함수 실행
    val_features = # TODO: preprocess_validation_function(dataset)
    
    print(f"추론용 features 수: {len(val_features['input_ids'])}")
    print(f"Example IDs: {val_features['example_id']}")

    # === 8. T5 입력 형식 실험 ===
    
    print("\n=== T5 입력 형식 실험 ===")
    
    # TODO: 다양한 입력 형식 테스트
    test_question = "대한민국의 수도는?"
    test_context = "대한민국의 수도는 서울특별시이다."
    
    # 표준 형식
    standard_input = # TODO: "question: {test_question} context: {test_context}"
    print(f"Standard format: {standard_input}")
    
    # TODO: 다른 형식들도 실험해보세요
    alternative_formats = [
        # TODO: 다양한 프롬프트 형식 실험
        # 예시: f"질문: {test_question} 지문: {test_context}"
        # f"Q: {test_question} C: {test_context}"
    ]
    
    for alt_format in alternative_formats:
        print(f"Alternative format: {alt_format}")
    
    # === 9. 토큰 길이 분석 ===
    
    print("\n=== 토큰 길이 분석 ===")
    
    # TODO: 입력과 출력의 토큰 길이 분포 분석
    input_lengths = []
    target_lengths = []
    
    for i in range(len(train_features['input_ids'])):
        input_len = len([t for t in train_features['input_ids'][i] if t != tokenizer.pad_token_id])
        target_len = len([t for t in train_features['labels'][i] if t != -100])
        
        input_lengths.append(input_len)
        target_lengths.append(target_len)
    
    print(f"평균 입력 길이: {sum(input_lengths) / len(input_lengths):.1f}")
    print(f"평균 타겟 길이: {sum(target_lengths) / len(target_lengths):.1f}")
    print(f"최대 입력 길이: {max(input_lengths)}")
    print(f"최대 타겟 길이: {max(target_lengths)}")

"""
학습 목표:
1. Seq2Seq QA의 간단하고 직관적인 전처리 과정 이해
2. T5 입력 형식과 텍스트-to-텍스트 변환 방식 학습
3. Extractive QA와의 전처리 복잡도 차이 체감
4. 입력-출력 매핑의 단순함과 유연성 경험

핵심 개념:

1. Seq2Seq QA 전처리의 단순함:
   - Extractive QA: offset mapping, 위치 라벨링, 다중 청크
   - Seq2Seq QA: 단순 텍스트 변환, 입력-출력 매핑
   - 복잡성이 10분의 1 수준으로 감소!

2. T5 입력 형식:
   - "question: 질문 context: 컨텍스트" → 답변
   - 표준화된 프롬프트 형식
   - 모델이 이 형식에 맞춰 학습됨

3. 텍스트-to-텍스트 변환:
   - 모든 입력을 텍스트로, 모든 출력도 텍스트로
   - NLP 태스크를 통일된 생성 문제로 변환
   - 유연하고 확장 가능한 접근법

4. 라벨 처리:
   - text_target 파라미터로 타겟 텍스트 토크나이즈
   - 패딩 토큰(-100)은 loss 계산에서 제외
   - 순수한 생성 과제로 학습

5. 전처리 단계별 비교:

   **Extractive QA 전처리**:
   1. 복잡한 토크나이제이션 (overflow, offset mapping)
   2. 문자-토큰 위치 매핑
   3. 다중 청크 처리
   4. 시작/끝 위치 라벨링
   5. 유효성 검사
   
   **Seq2Seq QA 전처리**:
   1. 간단한 텍스트 결합
   2. 표준 토크나이제이션
   3. 타겟 텍스트 토크나이제이션
   4. 패딩 처리
   5. 완료!

6. 장점과 특징:
   - 구현이 간단하고 직관적
   - 다양한 형태의 답변 생성 가능
   - 요약, 설명, 추론 등 복잡한 답변 지원
   - 새로운 정보 생성 가능

7. 실무 고려사항:
   - 입력 형식 표준화 중요
   - 답변 길이 제한 설정
   - 생성 품질 vs 속도 트레이드오프
   - 다양한 프롬프트 형식 실험

메모리 효율성:
- Extractive QA: 복잡한 메타데이터 저장 필요
- Seq2Seq QA: 단순한 입력-출력 쌍만 저장
- 전처리 속도 대폭 향상

확장성:
- 새로운 질문 유형 쉽게 추가
- 다양한 언어로 확장 용이
- 멀티태스크 학습 가능 (번역, 요약 등 동시 학습)
"""
