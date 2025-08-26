# === Step 2 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 2 TODO 해답:

1. AdamW 옵티마이저 반환:
   return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

2. 학습 데이터셋 생성:
   train_dataset = ClassificationDataset(
       "train", data=self.data, tokenizer=self.lm_tokenizer
   )

3. 학습용 데이터로더 생성:
   train_dataloader = DataLoader(
       train_dataset,
       sampler=RandomSampler(train_dataset, replacement=False),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.train_batch,
       collate_fn=data_collator,
       drop_last=False,
   )

4. 검증 데이터셋 생성:
   val_dataset = ClassificationDataset(
       "valid", data=self.data, tokenizer=self.lm_tokenizer
   )

5. 검증용 데이터로더 생성:
   val_dataloader = DataLoader(
       val_dataset,
       sampler=SequentialSampler(val_dataset),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.infer_batch,
       collate_fn=data_collator,
       drop_last=False,
   )

6. 테스트 데이터셋 생성:
   test_dataset = ClassificationDataset(
       "test", data=self.data, tokenizer=self.lm_tokenizer
   )

7. 테스트용 데이터로더 생성:
   test_dataloader = DataLoader(
       test_dataset,
       sampler=SequentialSampler(test_dataset),
       num_workers=self.args.hardware.cpu_workers,
       batch_size=self.args.hardware.infer_batch,
       collate_fn=data_collator,
       drop_last=False,
   )

핵심 개념:
- ClassificationDataset: NSMC 데이터를 PyTorch 형식으로 변환
- RandomSampler vs SequentialSampler: 학습 시 랜덤, 평가 시 순차
- batch_size: 학습용은 train_batch, 추론용은 infer_batch
- data_collator: 배치 내 샘플들의 길이를 맞춰주는 함수
- num_workers: 데이터 로딩 병렬 처리 수
"""
