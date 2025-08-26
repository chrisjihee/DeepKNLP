# === Step 2 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 2 TODO 해답:

1. 사전학습된 토크나이저 로드:
   pretrained = "monologg/koelectra-base-v3-finetuned-korquad"
   tokenizer = AutoTokenizer.from_pretrained(pretrained)

2. QA 학습 데이터 구성:
   raw_data = [
       {
           "id": "qa_001",
           "question": "대한민국의 수도는?",
           "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다.",
           "answers": {
               "answer_start": [25],
               "text": ["서울특별시"]
           }
       },
       {
           "id": "qa_002", 
           "question": "대한민국의 국화는?",
           "context": "대한민국의 국기는 태극기이며, 국가는 애국가, 국화는 무궁화이다.",
           "answers": {
               "answer_start": [29],
               "text": ["무궁화"]
           }
       },
       {
           "id": "qa_003",
           "question": "한반도의 위치는?", 
           "context": "한반도는 동아시아 지역에 위치하며, 중국과 러시아, 일본 사이에 있다.",
           "answers": {
               "answer_start": [5],
               "text": ["동아시아 지역"]
           }
       }
   ]

3. QA 전처리 파라미터 설정:
   max_seq_length = 384
   doc_stride = 128
   pad_to_max_length = True

4. 질문에서 좌측 공백 제거:
   questions = [q.lstrip() for q in examples["question"]]

5. 토크나이제이션 수행:
   tokenized_examples = tokenizer(
       questions,
       examples["context"],
       truncation="only_second",
       max_length=max_seq_length,
       stride=doc_stride,
       return_overflowing_tokens=True,
       return_offsets_mapping=True,
       padding="max_length" if pad_to_max_length else False,
   )

6. 샘플 매핑과 오프셋 매핑 추출:
   sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
   offset_mapping = tokenized_examples.pop("offset_mapping")

7. CLS 토큰 인덱스 찾기:
   if tokenizer.cls_token_id in input_ids:
       cls_index = input_ids.index(tokenizer.cls_token_id)
   elif tokenizer.bos_token_id in input_ids:
       cls_index = input_ids.index(tokenizer.bos_token_id)
   else:
       cls_index = 0

8. 시퀀스 ID 가져오기:
   sequence_ids = tokenized_examples.sequence_ids(i)

9. 현재 샘플 인덱스와 답변 정보 가져오기:
   sample_index = sample_mapping[i]
   answers = examples["answers"][sample_index]

10. 답변의 문자 단위 시작/끝 위치 계산:
    start_char = answers["answer_start"][0]
    end_char = start_char + len(answers["text"][0])

11. 답변이 현재 청크 범위 내에 있는지 확인:
    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):

12. 추론용 전처리에서 질문 공백 제거:
    questions = [q.lstrip() for q in examples["question"]]

13. 추론용 토크나이제이션:
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

14. 샘플 매핑 추출:
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

15. 시퀀스 ID와 컨텍스트 인덱스 설정:
    sequence_ids = tokenized_examples.sequence_ids(i)
    context_index = 1

16. 학습용 전처리 함수 실행:
    train_features = prepare_train_features(dataset)

17. 토큰들을 텍스트로 디코딩:
    tokens = tokenizer.convert_ids_to_tokens(train_features['input_ids'][i])

18. 예측된 답변 토큰들 추출 및 디코딩:
    answer_tokens = tokens[start_pos:end_pos+1]
    predicted_answer = tokenizer.decode(train_features['input_ids'][i][start_pos:end_pos+1])

19. 추론용 전처리 함수 실행:
    val_features = prepare_validation_features(dataset)

핵심 개념:

1. 복잡한 전처리 과정:
   - 단순한 토크나이제이션을 넘어선 위치 매핑
   - 긴 컨텍스트를 여러 청크로 분할
   - 정확한 문자-토큰 위치 변환

2. Doc Stride:
   - 긴 문서를 겹치는 청크로 분할
   - stride=128: 128 토큰씩 이동하며 청크 생성
   - 답변이 청크 경계에 걸치는 경우 대응

3. Offset Mapping:
   - (start_char, end_char) 형태의 문자 위치 정보
   - 토큰별로 원본 텍스트의 문자 위치 매핑
   - 문자 기반 답변을 토큰 기반으로 정확히 변환

4. 위치 라벨링 복잡성:
   - 문자 단위 정답 → 토큰 단위 라벨 변환
   - 서브워드 토크나이제이션 고려
   - 답변이 청크 범위 밖이면 CLS 토큰 사용

5. 학습 vs 추론 차이:
   - 학습: start_positions, end_positions 계산 필요
   - 추론: example_id, offset_mapping 보존 필요
   - 후처리에서 원본 답변 복원을 위한 정보 유지

6. 다중 청크 처리:
   - 하나의 예제가 여러 feature로 분할 가능
   - sample_mapping으로 feature-example 연결
   - 각 청크에서 독립적으로 답변 위치 계산

실무 중요성:
- QA 성능의 대부분이 전처리 품질에 의존
- 정확한 위치 매핑이 핵심
- 언어별 토크나이제이션 특성 고려 필요
- 메모리 효율성과 정확성의 균형

디버깅 팁:
- 원본 텍스트와 토큰 위치 매핑 확인
- 예상 답변과 실제 라벨 위치 비교
- offset_mapping의 None 값 패턴 점검
- 긴 컨텍스트에서 청크 분할 확인
"""
