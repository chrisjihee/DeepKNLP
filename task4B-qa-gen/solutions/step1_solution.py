# === Step 1 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 1 TODO 해답:

1. 사전학습된 한국어 T5 기반 QA 모델 지정:
   pretrained = "paust/pko-t5-base-finetuned-korquad"

2. AutoTokenizer 로드:
   tokenizer = AutoTokenizer.from_pretrained(pretrained)

3. AutoModelForSeq2SeqLM 로드:
   model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)

4. 모델을 평가 모드로 설정:
   model.eval()

5. 컨텍스트 작성:
   context = '''대한민국은 동아시아의 한반도 군사 분계선 남부에 위치한 나라이다. 
   약칭으로 한국(한국 한자: 韓國)과 남한(한국 한자: 南韓)으로 부르며 현정체제는 대한민국 제6공화국이다. 
   대한민국의 국기는 대한민국 국기법에 따라 태극기이며, 국가는 관습상 애국가, 국화는 관습상 무궁화이다. 
   공용어는 한국어와 한국 수어이다. 수도는 서울특별시이다. 
   인구는 2024년 2월 기준으로 5,130만명이고, 이 중 절반이 넘는(50.74%) 2,603만명이 수도권에 산다.'''

6. 질문 리스트 작성:
   questions = [
       "대한민국의 수도는?",
       "대한민국의 국화는?", 
       "대한민국에 대해 설명해주세요",  # 생성형 QA의 장점!
       "대한민국의 주요 특징은?",
       "대한민국과 한국의 차이는?",
   ]

7. T5 형식의 입력 텍스트 구성:
   input_text = f"question: {question} context: {context}"

8. 입력 텍스트 토크나이즈:
   inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

9. 텍스트 생성:
   with torch.no_grad():
       output_ids = model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"], 
           max_length=max_length,
           num_beams=num_beams
       )

10. 생성된 토큰들을 텍스트로 디코딩:
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

11. 각 질문에 대해 답변 생성:
    answer = answer_question_generative(question, context)

12. Beam Search 크기 변경 실험:
    for num_beams in [1, 3, 5]:
        answer = answer_question_generative(question, context, num_beams=num_beams)

13. 최대 길이 변경 실험:
    for max_len in [20, 50, 100]:
        answer = answer_question_generative(question, context, max_length=max_len)

14. 창의적 질문들:
    creative_questions = [
        "대한민국에 대해 간략히 설명해주세요",
        "대한민국의 주요 특징은 무엇인가요?",
        "대한민국과 북한의 차이점을 설명해주세요",
        "대한민국의 역사를 요약해주세요"
    ]

15. 창의적 질문에 대한 답변 생성:
    answer = answer_question_generative(question, context, max_length=100)

핵심 개념:

1. Generative vs Extractive QA:
   - Extractive: 컨텍스트에서 기존 텍스트 스팬 추출
   - Generative: 새로운 텍스트를 생성하여 답변
   - Generative는 요약, 설명, 추론 등 더 복잡한 답변 가능

2. T5 (Text-to-Text Transfer Transformer):
   - 모든 NLP 태스크를 텍스트 생성으로 통일
   - Encoder-Decoder 구조 (BERT는 Encoder만)
   - "question: Q context: C" → 답변 텍스트 생성

3. 입력 형식의 중요성:
   - T5는 프롬프트 형식에 민감
   - "question: ... context: ..." 표준 형식 사용
   - 모델이 이 형식에 맞춰 fine-tuning됨

4. 생성 파라미터:
   - max_length: 생성할 최대 토큰 수
   - num_beams: Beam Search 폭 (품질 vs 속도)
   - do_sample: 확률적 생성 여부
   - temperature: 창의성 조절

5. Beam Search vs Greedy:
   - num_beams=1: Greedy Search (빠름, 단조로움)
   - num_beams>1: Beam Search (느림, 고품질)
   - 실무에서는 3-5가 적절

6. Generative QA 장점:
   - 요약형 답변 가능
   - 설명형 답변 생성
   - 창의적 추론 지원
   - 컨텍스트 외 정보 활용

7. 주의사항:
   - Hallucination 가능성
   - 사실성 검증 필요
   - 생성 시간 고려
   - 품질 변동성

실무 활용:
- 교육: 개념 설명, 요약
- 고객지원: 복합 문제 해결
- 연구: 논문 해석, 요약
- 창작: 스토리텔링, 아이디어 생성

Extractive QA와의 차이점:
- 구현 복잡도: Generative < Extractive (전후처리 측면)
- 답변 유연성: Generative > Extractive
- 정확성 보장: Generative < Extractive
- 처리 속도: Generative < Extractive
- 창의성: Generative > Extractive (압도적)
"""
