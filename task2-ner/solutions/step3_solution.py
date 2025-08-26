# === Step 3 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 3 TODO 해답:

1. training_step - example_ids 제거:
   inputs.pop("example_ids")

2. training_step - 토큰 분류 모델 순전파:
   outputs: TokenClassifierOutput = self.lang_model(**inputs)

3. training_step - 라벨과 예측값 추출:
   labels: torch.Tensor = inputs["labels"]
   preds: torch.Tensor = outputs.logits.argmax(dim=-1)

4. training_step - 정확도 계산:
   acc: torch.Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

5. validation_step - 예제 ID 추출:
   example_ids: List[int] = inputs.pop("example_ids").tolist()

6. validation_step - 토큰 분류 모델 순전파:
   outputs: TokenClassifierOutput = self.lang_model(**inputs)

7. validation_step - 예측값 추출:
   preds: torch.Tensor = outputs.logits.argmax(dim=-1)

8. test_step - validation_step과 동일한 처리:
   return self.validation_step(inputs, batch_idx)

9. infer_one - 텍스트 토크나이즈:
   inputs = self.lm_tokenizer(
       tupled(text),
       max_length=self.args.model.seq_len,
       padding="max_length",
       truncation=True,
       return_tensors="pt",
   )

10. infer_one - 토큰 분류 모델 추론:
    outputs: TokenClassifierOutput = self.lang_model(**inputs)

11. infer_one - 라벨 확률 계산:
    all_probs: Tensor = outputs.logits[0].softmax(dim=1)
    top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)

핵심 개념:
- training_step: 단순한 토큰 레벨 분류 (분류 모델과 유사)
- validation_step: NER의 핵심 복잡성 (토큰-문자 매핑)
- example_ids: 원본 데이터와의 매핑을 위한 식별자
- ignore_index=0: 패딩 토큰 제외 정확도 계산
- 간소화된 validation_step: 실제로는 매우 복잡한 BIO 태깅 후처리
- TokenClassifierOutput: 각 토큰 위치별 라벨 분류 결과
- infer_one: 실시간 개체명 인식 서비스용 단일 문장 처리
- tupled(): 단일 문장을 배치 차원으로 확장하는 유틸리티 함수
"""
