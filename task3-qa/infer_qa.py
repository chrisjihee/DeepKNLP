from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# 모델이 저장된 폴더 경로(체크포인트 경로)
model_dir = "output/korquad/train=KPF-BERT=dgx-a100/checkpoint-15723"

# 1. Tokenizer와 Model 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

# 2. 파이프라인 생성
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 3. 임의의 question / context 예시
context = """대한민국은 동아시아의 한반도 군사 분계선 남부에 위치한 나라이다. 약칭으로 한국(한국 한자: 韓國)과 남한(한국 한자: 南韓)으로 부르며 현정체제는 대한민국 제6공화국이다. 대한민국의 국기는 대한민국 국기법에 따라 태극기이며, 국가는 관습상 애국가, 국화는 관습상 무궁화이다. 공용어는 한국어와 한국 수어이다. 수도는 서울특별시이다. 인구는 2024년 2월 기준으로 5,130만명이고, 이 중 절반이 넘는(50.74%) 2,603만명이 수도권에 산다."""
questions = [
    "대한민국의 수도는?",
    "대한민국의 국화는?",
    "대한민국의 국가는?",
    "대한민국의 위치는?",
    "대한민국의 약칭은?",
    "대한민국의 약칭 2가지는?",
    "대한민국의 인구는?",
    "대한민국의 공용어는?",
    "대한민국의 공용어 2가지는?",
    "대한민국의 헌정체제는?",
    "대한민국 대부분의 인구는 어디에 사는가?",
    "한반도는 지구상 어디에 있는가?",
]

# 4. 추론
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result}")
    print()
