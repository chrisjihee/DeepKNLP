### 목적과 전체 구조
- **역할**: `task1-cls/run_cls.py`는 NSMC 감성분석(Task1-Classification)을 위한 학습/평가/서빙을 한 파일에서 제공하는 CLI 엔트리 포인트입니다.  
- **구성**: Typer 기반 CLI(`train`, `test`, `serve`) → Lightning Fabric로 분산/혼합정밀 제어 → Hugging Face Transformers로 `AutoModelForSequenceClassification` fine-tuning → CSV/TensorBoard 로깅 → Checkpoint 저장/로드 → Flask 웹서빙.  

---

### 주요 의존성/유틸
- **Transformers**: `AutoConfig`, `AutoTokenizer`, `AutoModelForSequenceClassification`
- **Lightning Fabric**: 분산 실행, logger(TensorBoard/CSV), seed, barrier, all_gather
- **Dataset/Helper**: `ClassificationDataset`, `NsmcCorpus`, `data_collator`, `accuracy`, `CheckpointSaver`
- **Flask/Flask-Classful**: 간단한 inference Web API/페이지
- **Typer**: CLI 옵션 파싱과 서브커맨드

---

### `NSMCModel` 클래스
- 초기화
  - `NsmcCorpus(args)`로 데이터 메타 준비, `AutoConfig/Tokenizer/Model` 로드, label 수를 config에 반영.
- 체크포인트
  - `to_checkpoint`/`from_checkpoint`: 모델 state와 실행 progress 저장/복원.
  - 파일 로더: `load_checkpoint_file`, `load_last_checkpoint_file("**/*.ckpt")`.
- 최적화/데이터
  - `configure_optimizers`: `AdamW(lr=args.learning.learning_rate)`
  - `train_dataloader`/`val_dataloader`/`test_dataloader`: `ClassificationDataset` + Sampler + `data_collator`
- 학습/평가 스텝
  - `training_step`: forward → loss/acc 산출
  - `validation_step`/`test_step`: loss, preds, labels 반환
- 단일 문장 추론
  - `infer_one(text)`: tokenizer → model → softmax 확률 → 긍정/부정 및 시각화용 포맷 반환
- 웹서버
  - `run_server`: 내부 `WebAPI` 등록
  - `WebAPI.index`: 템플릿 렌더
  - `WebAPI.api`: JSON body(text) 받아 `infer_one` 실행 결과 반환

---

### 루프 함수들
- `train_loop`
  - Epoch/step 진행, `fabric.backward`로 역전파, 주기적 metric 집계(all_gather 평균), 로그, `val_loop` 트리거, barrier 동기화.
- `val_loop`
  - 전체 preds/labels를 all_gather 후 `accuracy`, `val_loss` 계산/로그.
  - `CheckpointSaver.save_checkpoint`로 best 기준(`saving_mode`) 관리.
- `test_loop`
  - 선택적 checkpoint 로드 → 전체 set 평가(`test_acc`, `test_loss`) 로깅.

---

### CLI 서브커맨드

#### 1) `train`
- 옵션 그룹
  - **env**: 프로젝트/잡 이름, 버전, 로깅 파일/형식
  - **data**: `data_home/name`, 파일 경로, `num_check`
  - **model**: `pretrained`, `finetuning` 출력 경로, `seq_len`
  - **hardware**: `accelerator(cuda/cpu/mps)`, `precision(16-mixed 등)`, `strategy(ddp 등)`, `devices([0,1,...])`, 배치/워커
  - **printing**: step/ratio 기반 출력 간격, 로그 tag 포맷
  - **learning**: `learning_rate`, `random_seed`, `saving_mode(max val_acc 등)`, `num_saving`, `num_epochs`, 검증 주기, ckpt 이름 포맷
- 실행 흐름
  - 출력 디렉토리와 Logger 준비(CSV/TensorBoard)
  - Fabric 초기화/launch, 전 rank 버전 동기화, barrier
  - `NSMCModel`/`optimizer`를 Fabric에 setup
  - dataloader들을 Fabric setup
  - `CheckpointSaver` 준비 후 `train_loop` 실행
  - test 파일 지정 시 best checkpoint로 `test_loop` 실행

#### 2) `test`
- 학습 없이 모델만 Fabric setup.
- `finetuning_home / model_name` 하위의 모든 `*.ckpt`에 대해 반복 평가 후 로그 기록.

#### 3) `serve`
- CPU Fabric으로 간단 세팅
- 마지막 체크포인트 자동 로드(`**/*.ckpt` 중 최신)
- Flask 서버 구동: `GET /` 템플릿, `POST /api`로 JSON(text) 입력 받아 추론 결과 반환.

---

### 분산/정밀도/로깅 포인트
- **분산**: `strategy="ddp"` 등, `devices=[...]`로 GPU 리스트 지정, `barrier`와 `all_gather`로 안전한 동기화/집계.
- **정밀도**: `precision="16-mixed" | "bf16-mixed" | "32-true"` 등 선택.
- **로깅**: `fabric.log_dict`로 step 기준 metric 기록, CSV/TensorBoard 동시 기록.

---

### 실행 예시
```bash
# 학습
python task1-cls/run_cls.py train --pretrained beomi/KcELECTRA-base --data_home data --data_name nsmc --num_epochs 3 --strategy ddp --device [0,1]

# 테스트(여러 체크포인트 일괄 평가)
python task1-cls/run_cls.py test --model_name "train=*"

# 서빙
python task1-cls/run_cls.py serve --server_port 9164 --server_page serve_cls.html
```

---

### 커스터마이징 힌트
- **모델/토크나이저**: `--pretrained`로 교체 가능(HF Hub).
- **시퀀스 길이**: `--seq_len`으로 trade-off 조정.
- **배치/워커**: 하드웨어 상황에 맞게 `--train_batch`, `--infer_batch`, `--cpu_workers`.
- **학습 스케줄**: 현재 고정 lr(AdamW). 필요 시 스케줄러 추가 가능.
- **저장 정책**: `--saving_mode "max val_acc"` 등 변경, `--num_saving`으로 k-best 유지.

---

- 이 파일은 Typer 기반 CLI로 학습/평가/서빙 전체 수명주기를 관리하고, Lightning Fabric과 Transformers를 통해 분산/정밀도/로깅까지 일관되게 묶어 둔 구조입니다.