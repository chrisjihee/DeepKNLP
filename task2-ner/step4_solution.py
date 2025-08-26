# === Step 4 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 4 TODO 해답:

train_loop:

1. 에포크 수 가져오기:
   for epoch in range(model.args.learning.num_epochs):

2. 모델을 학습 모드로 설정:
   model.train()

3. 역전파 및 가중치 업데이트:
   optimizer.zero_grad()
   outputs = model.training_step(batch, i)
   fabric.backward(outputs["loss"])
   optimizer.step()

4. 분산 환경에서 메트릭 수집:
   "loss": fabric.all_gather(outputs["loss"]).mean().item(),
   "acc": fabric.all_gather(outputs["acc"]).mean().item(),

5. 주기적 검증 및 체크포인트 저장:
   if model.args.prog.global_step % check_interval < 1:
       val_loop(model, val_dataloader, checkpoint_saver)

val_loop:

6. 검증 단계 실행:
   outputs = model.validation_step(batch, i)

7. 결과 수집:
   preds.extend(outputs["preds"])
   labels.extend(outputs["labels"])
   losses.append(outputs["loss"])

8. 분산 환경에서 모든 예측 결과 수집:
   all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
   all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()

9. NER 메트릭들 계산:
   "val_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
   "val_acc": accuracy(all_preds, all_labels, ignore_index=0).item(),
   "val_F1c": NER_Char_MacroF1.all_in_one(all_preds, all_labels, label_info=model.labels),
   "val_F1e": NER_Entity_MacroF1.all_in_one(all_preds, all_labels, label_info=model.labels),

핵심 개념:
- train_loop: 전체 에포크와 배치에 걸친 학습 관리
- val_loop: 검증 데이터 전체에 대한 성능 평가
- fabric.all_gather(): 분산 학습 환경에서 모든 프로세스 결과 수집
- check_interval: 학습 중 주기적 검증 수행 간격
- NER 특화 메트릭: 
  * val_F1c: 문자 레벨 Macro F1 (Character-level)
  * val_F1e: 개체 레벨 Macro F1 (Entity-level)
- 토큰 vs 문자 vs 개체: 3가지 레벨의 평가 관점
- checkpoint_saver: 성능 기반 모델 저장 (보통 val_F1c 기준)
- fabric.barrier(): 분산 환경 프로세스 동기화
- 분류 vs NER 차이: 단순 accuracy vs 다층 F1 메트릭
"""
