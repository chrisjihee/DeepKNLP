# === Step 2 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 2 TODO 해답:

1. AdamW 옵티마이저 반환:
   return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

2. 학습 데이터셋 생성:
   train_dataset = NERDataset("train", data=self.data, tokenizer=self.lm_tokenizer)

3. 학습용 데이터로더 생성:
   train_dataloader = DataLoader(
       train_dataset,
       sampler=RandomSampler(train_dataset, replacement=False),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.train_batch,
       collate_fn=self.data.encoded_examples_to_batch,
       drop_last=False,
   )

4. 검증 데이터셋 생성:
   val_dataset = NERDataset("valid", data=self.data, tokenizer=self.lm_tokenizer)

5. 검증용 데이터로더 생성:
   val_dataloader = DataLoader(
       val_dataset,
       sampler=SequentialSampler(val_dataset),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.infer_batch,
       collate_fn=self.data.encoded_examples_to_batch,
       drop_last=False,
   )

6. validation_step에서 토큰-문자 매핑을 위해 데이터셋 저장:
   self._infer_dataset = val_dataset

7. 테스트 데이터셋 생성:
   test_dataset = NERDataset("test", data=self.data, tokenizer=self.lm_tokenizer)

8. 테스트용 데이터로더 생성:
   test_dataloader = DataLoader(
       test_dataset,
       sampler=SequentialSampler(test_dataset),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.infer_batch,
       collate_fn=self.data.encoded_examples_to_batch,
       drop_last=False,
   )

9. test_step에서 토큰-문자 매핑을 위해 데이터셋 저장:
   self._infer_dataset = test_dataset

핵심 개념:
- NERDataset: NER 특화 데이터셋 클래스 (토큰 레벨 라벨링)
- encoded_examples_to_batch: NER 전용 collate 함수 (example_ids 포함)
- _infer_dataset: validation_step에서 토큰-문자 매핑을 위해 필요
- example_ids: 배치 내 각 예제의 원본 인덱스 (복잡한 후처리에 필요)
- 분류 vs NER collate_fn: data_collator vs encoded_examples_to_batch
- NER 특화 요소: 토큰-문자 오프셋 매핑, BIO 태깅 후처리, 다층 평가
"""
