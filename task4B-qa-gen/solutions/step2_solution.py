# === Step 2 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 2 TODO 해답:

1. 사전학습된 T5 토크나이저 로드:
   pretrained = "paust/pko-t5-base-finetuned-korquad"
   tokenizer = AutoTokenizer.from_pretrained(pretrained)

2. Seq2Seq QA 학습 데이터 구성:
   raw_data = [
       {
           "id": "qa_001",
           "question": "대한민국의 수도는?",
           "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다.",
           "answers": {
               "answer_start": [25],  # Generative에서는 실제로 사용 안함
               "text": ["서울특별시"]  # 이것만 중요!
           }
       },
       {
           "id": "qa_002",
           "question": "대한민국에 대해 설명해주세요",  # 요약형 질문
           "context": "대한민국은 동아시아의 한반도에 위치한 나라이다. 수도는 서울이며, 인구는 약 5천만명이다.",
           "answers": {
               "answer_start": [0],
               "text": ["대한민국은 동아시아의 한반도에 위치한 나라로, 수도는 서울이며 인구는 약 5천만명이다."]
           }
       },
       {
           "id": "qa_003", 
           "question": "한국의 주요 특징은?",  # 설명형 질문
           "context": "한국은 동아시아에 위치하며, 발달된 기술과 문화로 유명하다. K-pop과 한류가 세계적으로 인기있다.",
           "answers": {
               "answer_start": [10],
               "text": ["발달된 기술과 문화, K-pop과 한류로 유명한 나라"]
           }
       }
   ]

3. Seq2Seq QA 전처리 파라미터 설정:
   max_seq_length = 512
   max_answer_length = 50
   pad_to_max_length = True

4. 각 컬럼에서 데이터 추출:
   questions = examples[question_column]
   contexts = examples[context_column]
   answers = examples[answer_column]

5. T5 형식의 입력 텍스트 생성:
   return " ".join(["question:", question.lstrip(), "context:", context.lstrip()])

6. T5 입력 형식으로 변환:
   inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]

7. 답변 텍스트 추출:
   targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]

8. T5 입력 형식으로 변환:
   inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

9. 입력 텍스트들을 토크나이즈:
   model_inputs = tokenizer(
       inputs,
       max_length=max_seq_length,
       padding="max_length" if pad_to_max_length else False,
       truncation=True
   )

10. 타겟 텍스트들을 토크나이즈:
    labels = tokenizer(
        text_target=targets,
        max_length=max_answer_length, 
        padding="max_length" if pad_to_max_length else False,
        truncation=True
    )

11. 패딩 토큰을 -100으로 변경:
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

12. 모델 입력에 라벨 추가:
    model_inputs["labels"] = labels["input_ids"]

13. T5 입력 형식으로 변환:
    inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

14. 입력 텍스트들을 토크나이즈:
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding="max_length" if pad_to_max_length else False,
        truncation=True
    )

15. 타겟 텍스트들을 토크나이즈:
    labels = tokenizer(
        text_target=targets,
        max_length=max_answer_length,
        padding="max_length" if pad_to_max_length else False, 
        truncation=True
    )

16. 패딩 토큰을 -100으로 변경:
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

17. 학습용 전처리 함수 실행:
    train_features = preprocess_function(dataset)

18. 입력 텍스트 디코딩:
    input_text = tokenizer.decode(train_features['input_ids'][i], skip_special_tokens=True)

19. 라벨 텍스트 디코딩:
    target_text = tokenizer.decode(label_ids_clean, skip_special_tokens=True)

20. 추론용 전처리 함수 실행:
    val_features = preprocess_validation_function(dataset)

21. 표준 형식:
    standard_input = f"question: {test_question} context: {test_context}"

22. 다른 형식들:
    alternative_formats = [
        f"질문: {test_question} 지문: {test_context}",
        f"Q: {test_question} C: {test_context}",
        f"[질문] {test_question} [맥락] {test_context}"
    ]

핵심 개념:

1. Seq2Seq QA 전처리의 단순함:
   - Extractive QA: 복잡한 offset mapping, 위치 라벨링
   - Seq2Seq QA: 단순한 텍스트 변환, 입력-출력 매핑
   - 복잡성이 1/10 수준으로 감소!

2. T5 입력 형식의 표준화:
   - "question: ... context: ..." 형태
   - 모델이 이 형식에 특화되어 학습됨
   - 일관된 프롬프트 형식 중요

3. 텍스트-to-텍스트 변환:
   - 모든 입력을 텍스트로, 모든 출력도 텍스트로
   - NLP 태스크를 통일된 생성 문제로 변환
   - 유연하고 확장 가능한 접근법

4. text_target 파라미터:
   - T5의 타겟 텍스트 토크나이제이션
   - 디코더 입력 자동 생성
   - 라벨 시퀀스 구성

5. 패딩 토큰 처리:
   - 패딩 토큰(-100)은 loss 계산에서 제외
   - ignore_index=-100 활용
   - 실제 타겟 토큰만 학습에 사용

6. 전처리 단계별 비교:

   **Extractive QA**:
   ⭐⭐⭐⭐⭐ 1. 복잡한 토크나이제이션 (overflow, offset)
   ⭐⭐⭐⭐⭐ 2. 문자-토큰 위치 매핑
   ⭐⭐⭐⭐⭐ 3. 다중 청크 처리
   ⭐⭐⭐⭐⭐ 4. 시작/끝 위치 라벨링
   ⭐⭐⭐⭐   5. 유효성 검사

   **Seq2Seq QA**:
   ⭐        1. 텍스트 결합
   ⭐        2. 표준 토크나이제이션
   ⭐        3. 타겟 토크나이제이션
   ⭐        4. 패딩 처리
               5. 완료!

7. 답변 형태의 유연성:
   - Extractive: 컨텍스트의 정확한 스팬만
   - Seq2Seq: 요약, 설명, 추론 등 자유로운 형태
   - 교육적 가치가 높은 답변 생성 가능

8. 데이터 확장성:
   - 새로운 질문 유형 쉽게 추가
   - 다양한 답변 스타일 지원
   - 멀티태스크 학습 가능

실무 장점:
- 구현 속도 10배 향상
- 디버깅 용이성
- 유지보수 간편
- 확장성 우수

메모리 효율성:
- 복잡한 메타데이터 불필요
- 단순한 입력-출력 쌍
- 캐싱 전략 단순화
- 처리 속도 향상

교육적 가치:
- 직관적 이해 가능
- 단계별 학습 용이
- 실습 시간 단축
- 핵심 개념 집중
"""
