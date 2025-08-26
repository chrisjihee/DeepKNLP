# === Step 2: QA 데이터 전처리 ===
# 수강생 과제: TODO 부분을 완성하여 QA 데이터의 복잡한 전처리 과정을 이해하세요.

import logging
import torch
from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # === 1. 토크나이저 로드 ===
    
    # TODO: 사전학습된 토크나이저를 로드하세요
    # 힌트: "monologg/koelectra-base-v3-finetuned-korquad" 사용
    pretrained = # TODO: 완성하세요
    tokenizer = # TODO: 완성하세요

    # === 2. 예제 QA 데이터 준비 ===
    
    # TODO: QA 학습 데이터 형태로 예제를 구성하세요
    # 힌트: KorQuAD 형식 - id, question, context, answers (start, text)
    raw_data = [
        {
            "id": "qa_001",
            "question": # TODO: 질문 작성,
            "context": # TODO: 컨텍스트 작성,
            "answers": {
                "answer_start": [# TODO: 답변 시작 위치 (문자 단위)],
                "text": [# TODO: 정답 텍스트]
            }
        },
        # TODO: 더 많은 예제 추가 (최소 3개)
    ]

    # 데이터셋 생성
    dataset = Dataset.from_list(raw_data)

    # === 3. QA 전처리 설정 ===
    
    # TODO: QA 전처리에 필요한 파라미터들을 설정하세요
    max_seq_length = # TODO: 최대 시퀀스 길이 (예: 384)
    doc_stride = # TODO: 긴 문서 처리를 위한 stride (예: 128)
    pad_to_max_length = # TODO: 최대 길이로 패딩 여부 (True/False)

    # === 4. 학습용 전처리 함수 구현 ===
    
    def prepare_train_features(examples):
        """
        학습용 QA 데이터 전처리 함수
        긴 컨텍스트를 여러 청크로 나누고, 각 청크에 대한 답변 위치를 계산
        
        Args:
            examples: 원본 QA 데이터 배치
            
        Returns:
            dict: 토크나이즈된 features
        """
        # TODO: 질문에서 좌측 공백 제거
        # 힌트: [q.lstrip() for q in examples["question"]]
        questions = # TODO: 완성하세요

        # TODO: 토크나이제이션 수행
        # 힌트: tokenizer() 호출, truncation="only_second", return_overflowing_tokens=True, return_offsets_mapping=True
        tokenized_examples = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
            # questions, contexts, truncation, max_length, stride, 
            # return_overflowing_tokens, return_offsets_mapping, padding
        )

        # TODO: 샘플 매핑과 오프셋 매핑 추출
        # 힌트: tokenized_examples.pop() 사용
        sample_mapping = # TODO: 완성하세요
        offset_mapping = # TODO: 완성하세요

        # 답변 위치 라벨링을 위한 리스트 초기화
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # TODO: CLS 토큰 인덱스 찾기 (불가능한 답변의 기본 위치)
            # 힌트: tokenizer.cls_token_id 또는 tokenizer.bos_token_id 사용
            input_ids = tokenized_examples["input_ids"][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = # TODO: 완성하세요
            elif tokenizer.bos_token_id in input_ids:
                cls_index = # TODO: 완성하세요
            else:
                cls_index = 0

            # TODO: 시퀀스 ID 가져오기 (질문과 컨텍스트 구분)
            # 힌트: tokenized_examples.sequence_ids(i) 사용
            sequence_ids = # TODO: 완성하세요

            # TODO: 현재 샘플 인덱스와 답변 정보 가져오기
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # 답변이 없는 경우 처리
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # TODO: 답변의 문자 단위 시작/끝 위치 계산
                start_char = # TODO: 완성하세요
                end_char = # TODO: 완성하세요

                # TODO: 컨텍스트 부분의 토큰 시작/끝 인덱스 찾기
                # 힌트: sequence_ids를 이용하여 컨텍스트 부분(1 또는 0) 찾기
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:  # 컨텍스트는 보통 1
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # TODO: 답변이 현재 청크 범위 내에 있는지 확인
                # 힌트: offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char
                if not (# TODO: 조건 완성):
                    # 답변이 범위 밖에 있으면 CLS 인덱스 사용
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # TODO: 정확한 토큰 위치 찾기
                    # 시작 위치: 답변 시작 문자를 포함하는 토큰
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    # 끝 위치: 답변 끝 문자를 포함하는 토큰
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # === 5. 추론용 전처리 함수 구현 ===
    
    def prepare_validation_features(examples):
        """
        추론용 QA 데이터 전처리 함수
        답변 위치는 계산하지 않고, 후처리를 위한 정보만 보존
        
        Args:
            examples: 원본 QA 데이터 배치
            
        Returns:
            dict: 토크나이즈된 features (with example_id, offset_mapping)
        """
        # TODO: 질문에서 좌측 공백 제거
        questions = # TODO: 완성하세요

        # TODO: 토크나이제이션 수행 (학습용과 동일하지만 답변 라벨링 없음)
        tokenized_examples = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
        )

        # TODO: 샘플 매핑 추출
        sample_mapping = # TODO: 완성하세요

        # 후처리를 위한 example_id 저장
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # TODO: 시퀀스 ID와 컨텍스트 인덱스 설정
            sequence_ids = # TODO: 완성하세요
            context_index = 1  # 일반적으로 컨텍스트는 1

            # TODO: 샘플 인덱스와 example_id 저장
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # TODO: 컨텍스트가 아닌 부분의 offset_mapping을 None으로 설정
            # 힌트: 후처리에서 컨텍스트 부분만 고려하기 위함
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # === 6. 전처리 실행 및 결과 확인 ===
    
    print("=== 학습용 전처리 결과 ===")
    # TODO: 학습용 전처리 함수 실행
    train_features = # TODO: 완성하세요
    
    print(f"원본 샘플 수: {len(dataset)}")
    print(f"전처리 후 features 수: {len(train_features['input_ids'])}")
    
    # 첫 번째 feature 상세 정보 출력
    for i in range(min(2, len(train_features['input_ids']))):
        print(f"\n--- Feature {i} ---")
        print(f"Input IDs length: {len(train_features['input_ids'][i])}")
        print(f"Start position: {train_features['start_positions'][i]}")
        print(f"End position: {train_features['end_positions'][i]}")
        
        # TODO: 토큰들을 텍스트로 디코딩하여 확인
        tokens = # TODO: tokenizer.convert_ids_to_tokens() 사용
        print(f"Decoded tokens (first 50): {tokens[:50]}")
        
        # 답변 부분 확인
        start_pos = train_features['start_positions'][i]
        end_pos = train_features['end_positions'][i]
        if start_pos < len(tokens) and end_pos < len(tokens):
            # TODO: 예측된 답변 토큰들 추출 및 디코딩
            answer_tokens = # TODO: 완성하세요
            predicted_answer = # TODO: tokenizer.decode() 사용
            print(f"Predicted answer: '{predicted_answer}'")

    print("\n=== 추론용 전처리 결과 ===")
    # TODO: 추론용 전처리 함수 실행
    val_features = # TODO: 완성하세요
    
    print(f"추론용 features 수: {len(val_features['input_ids'])}")
    print(f"Example IDs: {val_features['example_id']}")

"""
학습 목표:
1. QA 데이터의 복잡한 전처리 과정 이해
2. 긴 컨텍스트를 여러 청크로 나누는 방법 학습
3. 문자 위치를 토큰 위치로 변환하는 과정 체험
4. 학습용과 추론용 전처리의 차이점 파악

핵심 개념:

1. 토크나이제이션 복잡성:
   - 질문 + 컨텍스트를 하나의 시퀀스로 결합
   - [CLS] 질문 [SEP] 컨텍스트 [SEP] 형태
   - 서브워드 토크나이제이션으로 인한 위치 매핑 필요

2. Doc Stride:
   - 긴 컨텍스트를 겹치는 청크로 분할
   - stride만큼 이동하면서 여러 feature 생성
   - 답변이 청크 경계에 걸치는 경우 대응

3. Offset Mapping:
   - 각 토큰의 원본 문자 위치 정보
   - (start_char, end_char) 튜플 형태
   - 문자 기반 답변을 토큰 기반으로 변환

4. 위치 라벨링:
   - 문자 단위 답변 위치를 토큰 단위로 변환
   - 시작/끝 토큰 인덱스 정확히 계산
   - 답변이 청크 범위 밖이면 CLS 토큰 사용

5. 학습 vs 추론:
   - 학습: start_positions, end_positions 계산
   - 추론: example_id, offset_mapping 보존
   - 후처리에서 원본 답변 복원을 위한 정보 유지

복잡성 요인:
- 서브워드 토크나이제이션
- 긴 컨텍스트 처리
- 정확한 문자-토큰 위치 매핑
- 다중 청크 처리

실무 고려사항:
- 메모리 효율성 (긴 문서 처리)
- 답변 경계 정확성
- 다양한 언어의 토크나이제이션 특성
- 실시간 처리 성능
"""
