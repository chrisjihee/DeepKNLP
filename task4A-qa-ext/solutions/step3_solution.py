# === Step 3 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 3 TODO 해답:

1. start_logits과 end_logits 분리:
   start_logits, end_logits = predictions

2. example_id를 키로 하는 feature 매핑 생성:
   example_id = feature["example_id"]

3. 현재 예제에 해당하는 features 가져오기:
   features = all_features[example_id]

4. 현재 feature의 start/end logits 가져오기:
   start_logit = start_logits[feature_index]
   end_logit = end_logits[feature_index]

5. offset_mapping 가져오기:
   offset_mapping = feature["offset_mapping"]

6. 가능한 시작 위치들 찾기:
   start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
   end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()

7. 추가 유효성 검사 조건들:
   if (start_index <= end_index and 
       end_index - start_index < max_answer_length):

8. 문자 단위 시작/끝 위치 계산:
   start_char = offset_mapping[start_index][0]
   end_char = offset_mapping[end_index][1]

9. 컨텍스트에서 답변 텍스트 추출:
   answer_text = context[start_char:end_char]

10. 답변 점수 계산:
    score = start_logit[start_index] + end_logit[end_index]

11. 답변 후보들을 점수 순으로 정렬:
    valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)

12. CLS 토큰 점수 계산:
    cls_score = start_logits[0][0] + end_logits[0][0]

13. 최고 점수 답변 선택:
    best_answer = valid_answers[0]["text"]

14. "답변 없음" 반환:
    best_answer = ""

15. SQuAD v1.0의 경우 최고 점수 답변 선택:
    if valid_answers:
        best_answer = valid_answers[0]["text"]

16. 최종 결과에 추가:
    all_predictions[example_id] = best_answer

간단한 후처리 함수:

17. start/end logits 분리:
    start_logits, end_logits = predictions

18. 가장 높은 확률의 시작/끝 위치 찾기:
    start_index = np.argmax(start_logits[i])
    end_index = np.argmax(end_logits[i])

19. offset_mapping 가져오기:
    offset_mapping = feature["offset_mapping"]

20. 문자 위치 계산 및 답변 추출:
    start_char = offset_mapping[start_index][0]
    end_char = offset_mapping[end_index][1]
    answer = context[start_char:end_char]

가상 데이터 구성:

21. 각 토큰 위치별 시작 확률 로그 값:
    start_logits = np.array([
        [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1,
         0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ])

22. 각 토큰 위치별 끝 확률 로그 값:
    end_logits = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]
    ])

23. 간단한 후처리 함수 실행:
    simple_results = simple_postprocess_qa_predictions(examples, features, predictions)

24. 전체 후처리 함수 실행:
    full_results = postprocess_qa_predictions(
        examples, features, predictions,
        n_best_size=20, max_answer_length=30
    )

핵심 개념:

1. 후처리의 핵심 역할:
   - QA에서 가장 복잡하고 중요한 단계
   - 원시 logits을 의미 있는 답변으로 변환
   - 단순 argmax로는 최적 답변 보장 불가

2. N-best 후보 생성 이유:
   - 시작 위치 최고점 + 끝 위치 최고점 ≠ 최고 조합
   - 모든 가능한 시작-끝 조합 고려 필요
   - 유효성 검사를 통한 불가능한 조합 제거

3. 유효성 검사 항목:
   - start_index <= end_index: 논리적 순서
   - 답변 길이 <= max_answer_length: 과도한 길이 방지
   - offset_mapping ≠ None: 컨텍스트 부분만 고려
   - 실제 문자 위치의 유효성

4. 점수 계산 방식:
   - Log 확률의 덧셈 = 확률의 곱셈
   - start_logit + end_logit = log(P_start × P_end)
   - 높은 점수 = 높은 신뢰도

5. SQuAD v2.0 특별 처리:
   - CLS 토큰 = "답변 없음" 점수
   - 일반 답변 점수 vs CLS 점수 비교
   - null_score_diff_threshold로 민감도 조정

6. 문자-토큰 변환의 정밀성:
   - offset_mapping의 정확한 활용
   - 서브워드 경계와 문자 경계 고려
   - 토큰 인덱스 → 문자 위치 → 실제 답변

실무 최적화 포인트:

1. 성능 향상:
   - n_best_size 조정: 더 많은 후보 vs 계산 비용
   - max_answer_length: 도메인별 최적값 설정
   - 임계값 튜닝: null_score_diff_threshold

2. 정확성 개선:
   - 언어별 후처리 로직 최적화
   - 도메인 특화 유효성 검사 추가
   - 앙상블 기법 적용

3. 효율성 개선:
   - 벡터화된 연산 활용
   - 불필요한 계산 생략
   - 메모리 사용량 최적화

디버깅 전략:
- 예상 답변과 실제 예측 위치 비교
- offset_mapping과 원본 텍스트 매칭 확인
- 점수 분포와 순위 분석
- 다양한 길이와 위치의 답변 테스트
"""
