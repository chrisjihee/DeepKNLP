# === Step 3 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 3 TODO 해답:

1. 기본 설정 (반복 발생 가능):
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
   )

2. no_repeat_ngram_size=3 적용:
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
       no_repeat_ngram_size=3,
   )

3. 각 repetition_penalty 값으로 생성:
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
       repetition_penalty=penalty,
   )

4. 각 temperature 값으로 생성:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_k=50,
       temperature=temp,
   )

5. 각 top_k 값으로 생성:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_k=k,
   )

6. 각 top_p 값으로 생성:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_p=p,
   )

7. 보수적 파라미터 조합:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       temperature=0.8,
       top_k=40,
       top_p=0.9,
       repetition_penalty=1.1,
   )

8. 창의적 파라미터 조합:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       temperature=1.2,
       top_k=100,
       top_p=0.95,
       repetition_penalty=1.3,
   )

9. 극단적 창의 파라미터 조합:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       temperature=2.0,
       top_k=200,
       repetition_penalty=1.5,
   )

10. 통합 최적화 설정:
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=10,
        max_length=50,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
    )

핵심 파라미터 상세 설명:

1. no_repeat_ngram_size:
   - 연속된 n-gram의 반복을 방지
   - 3: 3개 토큰 연속 반복 금지
   - 값이 클수록 더 긴 패턴의 반복 방지
   - 자연스러운 대화체 생성에 필수

2. repetition_penalty:
   - 이미 등장한 토큰의 재출현 확률 감소
   - 1.0: 패널티 없음 (기본값)
   - >1.0: 반복 억제 (1.1-1.5 권장)
   - 과도하면 부자연스러운 문장 생성

3. temperature:
   - 확률 분포의 날카로움 조절
   - <1.0: 더 확신 있는 선택 (보수적)
   - =1.0: 원래 모델 분포 유지
   - >1.0: 더 다양한 선택 (창의적)
   - 0에 가까우면 Greedy와 유사
   - 너무 크면 무작위에 가까워짐

4. top_k 조절:
   - 후보 토큰 수 직접 제한
   - 작은 값: 안정적이지만 단조로움
   - 큰 값: 다양하지만 품질 저하 위험
   - 40-100 범위가 일반적

5. top_p (Nucleus) 조절:
   - 누적 확률 기준 동적 후보 선택
   - 0.9-0.95: 실무에서 가장 많이 사용
   - 0.01: 매우 보수적 (Greedy와 유사)
   - 1.0: 전체 어휘 고려 (품질 저하)

파라미터 조합 전략:

1. 보수적 조합 (안정성 중시):
   - temperature=0.8, top_k=40, top_p=0.9
   - repetition_penalty=1.1
   - 뉴스, 공식 문서 등에 적합

2. 창의적 조합 (다양성 중시):
   - temperature=1.2, top_k=100, top_p=0.95
   - repetition_penalty=1.3
   - 소설, 시, 창작물에 적합

3. 균형 조합 (품질과 다양성):
   - temperature=0.9, top_k=50, top_p=0.92
   - repetition_penalty=1.2, no_repeat_ngram_size=3
   - 챗봇, 대화 시스템에 적합

실무 적용 팁:
- 도메인별 최적 파라미터는 실험을 통해 발견
- 사용자 피드백을 바탕으로 지속적 조정
- A/B 테스트를 통한 파라미터 최적화
- 실시간 서비스에서는 품질보다 속도 우선 고려
"""
