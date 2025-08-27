# === Step 3 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 3 TODO 해답:

1. T5 입력 형식 구성:
   input_text = f"question: {example['question']} context: {example['context']}"

2. 입력 토크나이제이션:
   inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

3. 점수와 함께 생성:
   outputs = model.generate(
       input_ids=inputs["input_ids"],
       attention_mask=inputs["attention_mask"],
       max_length=max_length,
       num_beams=num_beams,
       return_dict_in_generate=True,
       output_scores=True
   )

4. 생성된 시퀀스에서 답변 추출:
   generated_sequence = outputs.sequences[0]

5. 답변 텍스트 디코딩:
   answer = tokenizer.decode(generated_sequence, skip_special_tokens=True)

6. 소프트맥스로 확률 변환:
   token_prob = F.softmax(outputs.scores[i-1], dim=-1)[0, token_id].item()

7. 전체 점수 계산:
   overall_score = torch.prod(torch.tensor(token_scores)).item() if token_scores else 0.0

8. 점수 없이 간단한 생성:
   outputs = model.generate(
       input_ids=inputs["input_ids"], 
       attention_mask=inputs["attention_mask"],
       max_length=max_length,
       num_beams=num_beams
   )

9. 답변 추출 및 디코딩:
   answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

10. 후처리된 결과 리스트 초기화:
    processed_results = []

11. 간단한 후처리:
    cleaned_prediction = predicted_answer.strip()
    cleaned_expected = expected_answer.strip()

12. 정확도 계산:
    exact_match = cleaned_prediction.lower() == cleaned_expected.lower()

13. 단순 F1 점수 계산:
    pred_tokens = cleaned_prediction.split()
    expected_tokens = cleaned_expected.split()

14. 공통 토큰 계산:
    common_tokens = set(pred_tokens) & set(expected_tokens)

15. Precision, Recall, F1 계산:
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(expected_tokens)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

16. 사전학습된 T5 QA 모델 로드:
    pretrained = "paust/pko-t5-base-finetuned-korquad"

17. 토크나이저와 모델 로드:
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)

18. 테스트용 QA 데이터 구성:
    test_examples = [
        {
            "id": "test_001",
            "question": "대한민국의 수도는?",
            "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다.",
            "answers": {"text": ["서울특별시"]}
        },
        {
            "id": "test_002", 
            "question": "대한민국에 대해 설명해주세요",
            "context": "대한민국은 한반도 남부에 위치한 나라로, 수도는 서울이며 인구는 약 5천만명이다.",
            "answers": {"text": ["대한민국은 한반도 남부에 위치한 나라로, 수도는 서울이며 인구는 약 5천만명이다."]}
        }
    ]

19. Greedy Search:
    greedy_results = generate_answers_with_scores(model, tokenizer, test_examples, num_beams=1)

20. Beam Search:
    beam_results = generate_answers_with_scores(model, tokenizer, test_examples, num_beams=5)

21. 다양한 최대 길이로 실험:
    for max_len in [10, 30, 50, 100]:
        length_results = generate_answers_with_scores(model, tokenizer, [test_question], max_length=max_len)

22. 후처리 함수 실행:
    processed_results = simple_postprocess_seq2seq_predictions(test_examples, beam_results["predictions"])

23. 창의적 질문들:
    creative_examples = [
        {
            "id": "creative_001",
            "question": "대한민국에 대해 간략히 설명해주세요",
            "context": "대한민국은 동아시아의 한반도 남부에 위치한 나라이다. 수도는 서울특별시이며...",
            "answers": {"text": ["대한민국은 동아시아 한반도 남부의 나라로, 수도는 서울이다."]}
        }
    ]

24. 창의적 질문들에 대한 답변 생성:
    creative_results = generate_answers_with_scores(model, tokenizer, creative_examples, max_length=100)

핵심 개념:

1. Generative QA 후처리의 단순함:
   - Extractive QA: 10+ 단계의 복잡한 후처리
   - Generative QA: 3단계 간단한 후처리
   1) 텍스트 디코딩
   2) 텍스트 정제
   3) 평가 메트릭 계산

2. 텍스트 생성 과정:
   - model.generate(): 핵심 생성 메소드
   - return_dict_in_generate=True: 상세 정보 반환
   - output_scores=True: 토큰별 점수 정보 포함
   - skip_special_tokens=True: [PAD], [EOS] 등 제거

3. 생성 전략 비교:
   - Greedy (num_beams=1): 빠름, 단조로움
   - Beam Search (num_beams>1): 느림, 고품질
   - 실무에서는 3-5가 적절한 균형점

4. 점수 계산 방식:
   - 토큰별 확률: F.softmax(logits, dim=-1)
   - 전체 점수: 모든 토큰 확률의 곱
   - 긴 답변일수록 점수 낮아짐 (확률 곱셈 특성)
   - 점수 해석 시 길이 고려 필요

5. 평가 메트릭:
   - Exact Match: 완전 일치 여부 (대소문자 무시)
   - F1 Score: 토큰 단위 겹침 정도
   - BLEU, ROUGE: 더 정교한 생성 평가 (추가 고려)

6. 후처리 복잡도 비교:

   **Extractive QA 후처리** (⭐⭐⭐⭐⭐):
   1. N-best 후보 생성
   2. 모든 start×end 조합 검토
   3. 복잡한 점수 계산
   4. 위치 유효성 검사
   5. SQuAD v2.0 null 답변 처리
   6. 문자-토큰 변환
   7. 컨텍스트 범위 검증
   8. 답변 길이 제한
   9. 최종 후보 선택
   10. 텍스트 복원

   **Generative QA 후처리** (⭐):
   1. 직접 텍스트 디코딩
   2. 간단한 텍스트 정제
   3. 평가 메트릭 계산

7. 생성 길이의 영향:
   - 짧은 길이: 불완전한 답변, 높은 점수
   - 긴 길이: 완전한 답변, 낮은 점수
   - 도메인에 따른 최적 길이 설정 필요

8. 창의적 질문의 장점:
   - 요약형 답변 생성
   - 설명형 답변 제공
   - 추론 기반 답변
   - 컨텍스트 재구성

9. 실무 최적화 포인트:
   - Beam 크기: 품질 vs 속도 균형
   - 최대 길이: 도메인별 최적값
   - 배치 처리: 처리량 향상
   - 캐싱: 반복 질문 대응

디버깅 전략:
- 생성 과정 스텝별 확인
- 토큰별 점수 분포 분석
- 다양한 생성 파라미터 실험
- 예상 답변과 실제 생성 비교

성능 분석:
- 점수 분포와 품질 상관관계
- 생성 길이와 완성도 관계
- Beam 크기와 다양성 효과
- 창의적 파라미터 영향도

실무 활용 팁:
- 도메인별 최적 파라미터 튜닝
- 사용자 피드백 기반 점수 보정
- 답변 품질 필터링 시스템
- A/B 테스트를 통한 파라미터 최적화
"""
