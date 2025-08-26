# === Step 1 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 1 TODO 해답:

1. 사전학습된 한국어 GPT2 모델명 지정:
   pretrained = "skt/kogpt2-base-v2"

2. GPT2LMHeadModel 로드 및 평가 모드 설정:
   model = GPT2LMHeadModel.from_pretrained(pretrained)
   model.eval()

3. PreTrainedTokenizerFast 로드:
   tokenizer = PreTrainedTokenizerFast.from_pretrained(
       pretrained,
       eos_token="</s>",
   )

4. 생성 시작 프롬프트 지정:
   input_sentence = "안녕하세요"
   # 또는 input_sentence = "대한민국의 수도는"

5. 입력 텍스트 토큰화:
   input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

6. 기본 텍스트 생성 (Greedy Search):
   generated_ids = model.generate(
       input_ids,
       do_sample=False,
       min_length=10,
       max_length=50,
   )

7. 생성된 토큰 ID들을 텍스트로 디코딩:
   generated_text = tokenizer.decode([el.item() for el in generated_ids[0]])

핵심 개념:
- GPT2LMHeadModel: Causal Language Model (다음 토큰 예측)
- AutoRegressive Generation: 이전 토큰들을 바탕으로 다음 토큰 예측
- do_sample=False: Greedy Search (항상 최고 확률 토큰 선택)
- eos_token: 문장 종료를 나타내는 특수 토큰
- return_tensors="pt": PyTorch 텐서 형태로 반환
- model.eval(): 드롭아웃 등 비활성화, 추론 모드 설정
- 한국어 GPT2: SKT에서 공개한 한국어 특화 언어 모델

Greedy Search 특징:
- 확정적 (Deterministic): 동일 입력 → 동일 출력
- 빠른 추론 속도
- 안전하지만 단조로운 결과
- 창의성 부족, 반복 패턴 발생 가능
"""
