# === Step 2 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 2 TODO 해답:

1. Greedy Search:
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
   )

2. Beam Search (num_beams=3):
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
       num_beams=3,
   )

3. Beam Search (num_beams=1) - Greedy와 동일:
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
       num_beams=1,
   )

4. Top-k Sampling (k=50):
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_k=50,
   )

5. Top-k Sampling (k=1) - Greedy와 유사:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_k=1,
   )

6. Top-p Sampling (p=0.92):
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_p=0.92,
   )

7. Top-p Sampling (p=0.01) - Greedy와 유사:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_p=0.01,
   )

8. Top-k Sampling 3회 반복:
   generated_ids = model.generate(
       input_ids,
       do_sample=True,
       min_length=10,
       max_length=50,
       top_k=50,
   )

핵심 개념:

1. do_sample 파라미터:
   - False: 확정적 생성 (Greedy/Beam Search)
   - True: 확률적 생성 (Sampling 기법들)

2. Beam Search:
   - 여러 경로를 동시 탐색하여 전체적으로 더 좋은 문장 생성
   - num_beams: 탐색할 경로 수 (계산량과 품질의 트레이드오프)
   - num_beams=1: Greedy Search와 완전히 동일

3. Top-k Sampling:
   - 상위 k개 확률 토큰만 고려하여 샘플링
   - k가 클수록 다양성 증가, 작을수록 안정성 증가
   - k=1: 사실상 Greedy Search

4. Top-p (Nucleus) Sampling:
   - 누적 확률이 p에 도달할 때까지의 토큰들만 고려
   - 문맥에 따라 동적으로 후보 토큰 수 조절
   - p=0.01: 매우 보수적, p=1.0: 전체 어휘 고려

5. 생성 전략 비교:
   - Greedy: 빠르고 안정적, 단조로움
   - Beam: 품질 향상, 계산량 증가
   - Top-k: 다양성과 품질의 균형
   - Top-p: 문맥 적응적 다양성 조절

실무 선택 가이드:
- 안정적 서비스: Greedy Search
- 품질 중시: Beam Search (beams=3~5)
- 창의적 생성: Top-k (k=50) + Top-p (p=0.9)
- 실시간 서비스: Greedy 또는 작은 k값
"""
