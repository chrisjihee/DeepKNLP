from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# 모델이 저장된 폴더 경로(체크포인트 경로)
model_dir = "output/korquad/train=KPF-BERT=dgx-a100/checkpoint-15723"

# 1. Tokenizer와 Model 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

# 2. 파이프라인 생성
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 3. 임의의 question / context 예시
context = """오늘은 날씨가 맑고, 기온은 20도까지 올라가며
             오후에는 가벼운 비가 내릴 것으로 예측됩니다."""
question = "오늘 오후에 날씨는 어떨 것 같나요?"

# 4. 추론
result = qa_pipeline(question=question, context=context)
print(result)
