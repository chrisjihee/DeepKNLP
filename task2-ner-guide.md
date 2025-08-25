### 목적과 전체 구조
- **역할**: `task2-ner/run_ner.py`는 한국어 NER(Task2: Named Entity Recognition)을 위한 학습, 평가, 서빙을 하나의 Typer 기반 CLI로 제공합니다.
- **구성**: Typer CLI(`train`, `test`, `serve`) → Lightning Fabric으로 분산/정밀도/로깅 제어 → Hugging Face Transformers `AutoModelForTokenClassification` fine-tuning → CSV/TensorBoard 로깅 → Checkpoint 저장/로드 → Flask 웹서빙.

### 주요 의존성
- **Transformers**: `AutoConfig`, `AutoTokenizer(PreTrainedTokenizerFast)`, `AutoModelForTokenClassification`, `CharSpan`
- **Lightning Fabric**: 분산 실행, precision, barrier/all_gather, CSV/TensorBoard logger
- **데이터/헬퍼**: `NERCorpus`, `NERDataset`, `NEREncodedExample`, `encoded_examples_to_batch`, 지표(`accuracy`, `NER_Char_MacroF1`, `NER_Entity_MacroF1`), `CheckpointSaver`
- **웹**: Flask + Flask-Classful로 간단 REST API/템플릿 렌더
- **Typer**: 하위 커맨드와 옵션 파싱

### NERModel 클래스 핵심
- 초기화
  - `NERCorpus(args)`로 라벨/코퍼스 메타 로드, `labels`와 `label<->id` 매핑 생성.
  - `AutoConfig.from_pretrained(..., num_labels=...)`
  - `AutoTokenizer.from_pretrained(..., use_fast=True)`를 강제하며 `PreTrainedTokenizerFast`만 지원(assert).
  - `AutoModelForTokenClassification.from_pretrained(...)`.
- 체크포인트 I/O
  - `to_checkpoint`/`from_checkpoint`: 언어모델 state와 진행상태(`args.prog`) 저장/복원.
  - `load_checkpoint_file`, `load_last_checkpoint_file("**/*.ckpt")`.
- 데이터로더
  - `train/val/test_dataloader`: `NERDataset` + `Random/SequentialSampler` + `encoded_examples_to_batch`를 사용.
  - 검증/테스트 시 `_infer_dataset` 보관하여 step 단계에서 역참조.
- 학습/평가 스텝
  - `training_step`: `TokenClassifierOutput`에서 `loss`, `logits.argmax`로 `acc` 계산(`ignore_index=0`).
  - `validation_step`:
    - batch 내 `example_ids`로 원문 단위로 복원.
    - token-level 예측을 `CharSpan(token_to_chars)`로 문자 단위 span으로 펼침.
    - BIO 체계 적용을 위한 `label_to_char_labels`로 토큰 라벨을 문자 라벨 시퀀스로 변환.
    - 유효 문자 위치만 골라 `char_level`의 `labels/preds`를 구성해 반환.
  - `test_step`: `validation_step`을 재사용.
- 단일 문장 추론
  - `infer_one(text)`: tokenizer → model → 각 토큰의 top-1 label/확률을 반환(특수 토큰 제외).
- 웹서버
  - `run_server`: 내부 `WebAPI` 등록.
  - `WebAPI.index`: 템플릿 렌더(페이지).
  - `WebAPI.api`: `POST /api` JSON(text)을 받아 `infer_one` 결과 반환.

### 루프 함수들
- `train_loop`
  - step/epoch 관리, `fabric.backward`로 역전파, 주기적 로그(`fabric.log_dict`), 비율/스텝 기반 출력, 주기적으로 `val_loop` 호출, barrier 동기화.
- `val_loop`
  - 각 step의 char-level `preds/labels/loss` 수집 → `all_gather`로 전 rank 집계.
  - 지표 계산: `val_loss`, `val_acc(ignore_index=0)`, `val_F1c`(Char Macro-F1), `val_F1e`(Entity Macro-F1).
  - 체크포인트 저장 기준은 `saving_mode`(예: "max val_F1c").
- `test_loop`
  - 선택적 checkpoint 로드 → 전체 test set에 대해 `test_loss`, `test_acc`, `test_F1c`, `test_F1e` 로깅.

### CLI 서브커맨드

#### 1) train
- 옵션
  - **env**: 프로젝트/잡/버전/로깅 설정
  - **data**: `data_home/name`와 파일들(`train/valid/test`), `num_check`
  - **model**: `pretrained`, `finetuning` 출력 경로, `model_name`, `seq_len`
  - **hardware**: `accelerator(cuda|cpu|mps)`, `precision(16-mixed 등)`, `strategy(ddp 등)`, `devices`, 배치/워커
  - **printing**: 출력 빈도(비율/스텝), 로그 tag 포맷
  - **learning**: `learning_rate`, `random_seed`, `saving_mode`, `num_saving`, `num_epochs`, `check_rate_on_training`, checkpoint 이름 포맷
- 흐름
  - output 디렉토리 및 CSV/TensorBoard logger 구성 → Fabric 초기화/launch → rank 간 job_version 합의
  - `NERModel`과 `optimizer(AdamW)`를 Fabric에 setup
  - dataloader setup → `CheckpointSaver` 준비 → `train_loop`
  - test 파일이 있으면 best ckpt로 `test_loop`

#### 2) test
- 학습 없이 모델만 setup, `finetuning_home / model_name` 하위 `**/*.ckpt` 전부에 대해 반복 평가 후 로그 기록.

#### 3) serve
- CPU Fabric으로 간단 세팅, 최신 ckpt 자동 로드(`**/*.ckpt` 중 마지막).
- Flask 서버 실행: `GET /` 템플릿, `POST /api` 텍스트 입력 → 토큰 단위 label과 확률 반환.

### 분산/정밀도/로깅
- **분산**: `strategy="ddp"`와 `devices=[...]` 설정, barrier와 all_gather로 안전한 집계.
- **정밀도**: `precision="16-mixed" | "bf16-mixed" | "32-true"` 등.
- **로깅**: `fabric.log_dict`로 step 기준 CSV/TensorBoard 동시 기록.

### 실행 예시
```bash
# 학습
python task2-ner/run_ner.py train --pretrained klue/roberta-base --data_home data --data_name klue-ner --num_epochs 3 --strategy ddp --device [0,1]

# 테스트(여러 ckpt 일괄 평가)
python task2-ner/run_ner.py test --model_name "train=*"

# 서빙
python task2-ner/run_ner.py serve --server_port 9164 --server_page serve_ner.html
```

### 실전 포인트
- 토크나이저는 반드시 `PreTrainedTokenizerFast`여야 합니다(assert 존재).
- 문자 단위 평가는 `token_to_chars` 기반의 span 전개와 BIO 규칙 적용으로 구현됩니다.
- 체크포인트 선택은 기본 `max val_F1c`로 설정되어 NER 특성에 맞게 최적화되어 있습니다.

- 핵심: 토큰 분류 출력을 문자/엔티티 수준으로 재매핑해 평가하는 구조, Fabric으로 분산/정밀도/로깅을 일관되게 처리, Typer/Flask로 실험 전주기(학습-평가-서빙)를 한 파일에서 관리합니다.