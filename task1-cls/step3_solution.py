# === Step 3 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 3 TODO 해답:

1. training_step - 모델 순전파:
   outputs: SequenceClassifierOutput = self.lang_model(**inputs)

2. training_step - 라벨과 예측값 추출:
   labels: torch.Tensor = inputs["labels"]
   preds: torch.Tensor = outputs.logits.argmax(dim=-1)

3. training_step - 정확도 계산:
   acc: torch.Tensor = accuracy(preds=preds, labels=labels)

4. validation_step - 모델 순전파:
   outputs: SequenceClassifierOutput = self.lang_model(**inputs)

5. validation_step - 라벨과 예측값을 리스트로 변환:
   labels: List[int] = inputs["labels"].tolist()
   preds: List[int] = outputs.logits.argmax(dim=-1).tolist()

6. test_step - validation_step 재사용:
   return self.validation_step(inputs, batch_idx)

7. infer_one - 텍스트 토크나이즈:
   inputs = self.lm_tokenizer(
       text,
       max_length=self.args.model.seq_len,
       padding="max_length",
       truncation=True,
       return_tensors="pt",
   )

8. infer_one - 모델 추론:
   outputs: SequenceClassifierOutput = self.lang_model(**inputs)

9. infer_one - 확률로 변환:
   prob = outputs.logits.softmax(dim=1)

핵심 개념:
- training_step: loss와 accuracy 반환, gradient 계산 허용
- validation_step: 예측과 라벨 수집, @torch.no_grad() 데코레이터
- test_step: validation_step과 동일한 로직
- infer_one: 단일 텍스트 추론, 웹 서비스용 결과 포맷
- outputs.logits: 원시 로짓값, softmax로 확률 변환
- argmax: 가장 높은 확률의 클래스 선택
"""
