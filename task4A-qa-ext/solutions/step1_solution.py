# === Step 1 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 1 TODO 해답:

1. 사전학습된 한국어 QA 모델 지정:
   pretrained = "monologg/koelectra-base-v3-finetuned-korquad"

2. AutoTokenizer 로드:
   tokenizer = AutoTokenizer.from_pretrained(pretrained)

3. AutoModelForQuestionAnswering 로드:
   model = AutoModelForQuestionAnswering.from_pretrained(pretrained)

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
       "대한민국의 국가는?",
       "대한민국의 위치는?",
       "대한민국의 약칭은?",
       "대한민국의 인구는?",
       "대한민국의 공용어는?",
   ]

7. 질문과 컨텍스트 토크나이즈:
   inputs = tokenizer.encode_plus(
       question, context, return_tensors="pt", truncation=True, padding=True
   )

8. 모델 추론 수행:
   with torch.no_grad():
       outputs = model(**inputs)

9. start/end logits 추출:
   start_logits = outputs.start_logits
   end_logits = outputs.end_logits

10. 가장 높은 확률의 시작/끝 인덱스 찾기:
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

11. 예측된 답변 토큰들 추출:
    predict_answer_tokens = inputs["input_ids"][0, start_index:end_index + 1]

12. 토큰들을 텍스트로 디코딩:
    answer = tokenizer.decode(predict_answer_tokens)

13. 각 질문에 대해 답변 생성:
    answer = answer_question(question, context)

핵심 개념:

1. Extractive QA 원리:
   - 컨텍스트에서 답변을 직접 추출
   - 새로운 텍스트 생성하지 않음
   - Reading Comprehension의 핵심 방식

2. AutoModelForQuestionAnswering:
   - BERT + QA Head 구조
   - 각 토큰 위치에 대해 start/end 확률 계산
   - [CLS] 토큰으로 "답변 없음" 표현

3. Start/End Logits:
   - start_logits[i]: i번째 토큰이 답변 시작일 확률
   - end_logits[i]: i번째 토큰이 답변 끝일 확률
   - 가장 높은 확률 조합으로 답변 결정

4. 토크나이제이션:
   - [CLS] 질문 [SEP] 컨텍스트 [SEP] 형태
   - 질문과 컨텍스트를 하나의 시퀀스로 결합
   - 최대 길이 제한으로 긴 컨텍스트 잘림 가능

5. 한국어 QA 특성:
   - 서브워드 토크나이제이션
   - 조사, 어미 처리 복잡성
   - 문맥 의존적 의미 해석

예상 결과:
- "대한민국의 수도는?" → "서울특별시"
- "대한민국의 국화는?" → "무궁화"
- "대한민국의 공용어는?" → "한국어와 한국 수어"

주의사항:
- 컨텍스트에 답변이 반드시 포함되어야 함
- 긴 컨텍스트는 토큰 길이 제한으로 잘릴 수 있음
- 서브워드로 인한 불완전한 답변 생성 가능
"""
