# === Step 4 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 4 TODO 해답:

1. 에포크 수 가져오기:
   for epoch in range(model.args.learning.num_epochs):

2. 모델을 학습 모드로 설정:
   model.train()

3. 역전파 및 가중치 업데이트 구현:
   optimizer.zero_grad()  # 기울기 초기화
   outputs = model.training_step(batch, i)  # Forward pass
   fabric.backward(outputs["loss"])  # Backward pass
   optimizer.step()  # 가중치 업데이트

4. 분산 환경에서 메트릭 수집 및 평균화:
   "loss": fabric.all_gather(outputs["loss"]).mean().item(),
   "acc": fabric.all_gather(outputs["acc"]).mean().item(),

5. 주기적 검증 조건:
   if model.args.prog.global_step % check_interval < 1:
       val_loop(model, val_dataloader, checkpoint_saver)

6. val_loop - 검증 단계 실행:
   outputs = model.validation_step(batch, i)

7. val_loop - 결과 수집:
   preds.extend(outputs["preds"])
   labels.extend(outputs["labels"])
   losses.append(outputs["loss"])

8. val_loop - 분산 환경에서 예측 결과 수집:
   all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
   all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()

9. val_loop - 메트릭 계산:
   "val_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
   "val_acc": accuracy(all_preds, all_labels).item(),

핵심 개념:
- 에포크: 전체 데이터셋을 한 번 다 본 것
- 배치: 한 번에 처리하는 데이터의 묶음
- optimizer.zero_grad(): 이전 배치의 gradient 정보 제거
- fabric.backward(): 분산 환경에서 안전한 역전파
- fabric.all_gather(): 여러 GPU/프로세스의 값들을 모두 수집
- check_interval: 정기적으로 검증하는 주기
- validation은 gradient 계산 없이 성능만 측정
"""
