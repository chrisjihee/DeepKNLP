지금 `task3-qa` 폴더의 핵심 파일들을 빠르게 읽어서 구조와 흐름을 정확히 요약하겠습니다. 잠시만요.

[5 tools called]

방금 `task3-qa`의 핵심 파이썬 파일들을 열어 구조를 파악하고 있습니다. 이어서 서빙/추론 스크립트들도 동시에 확인하겠습니다.

[5 tools called]

### 목적과 전체 구조
- **역할**: `task3-qa`는 KorQuAD 등 QA를 위한 end-to-end 워크플로우(학습/평가/추론/서빙)를 제공합니다. Extractive QA(span 예측)와 Seq2Seq QA(생성형)를 모두 지원합니다.
- **구성**: 학습 스크립트(`train_qa.py`, `train_qa_seq2seq.py`) + Trainer 서브클래스(`trainer_qa.py`, `trainer_seq2seq_qa.py`) + 후처리(`utils_qa.py`) + 추론(`infer_qa*.py`) + 서빙(`serve_qa*.py`) + KorQuAD 전용 평가(`evaluate-KorQuAD-v1.py`) + 쉘 래퍼 스크립트들.
- 영어 용어는 그대로 사용하겠습니다 [[memory:7168608]].

### 핵심 파일별 역할
- **`train_qa.py` (extractive)**: 
  - HF `AutoModelForQuestionAnswering` + `Trainer`로 미세튜닝.
  - 데이터 로딩(로컬 `csv/json/jsonl` 또는 Hub), 전처리(overflow/stride), 평가용 후처리(`utils_qa.postprocess_qa_predictions`), metric(`squad`/`squad_v2`).
  - `QuestionAnsweringTrainer`로 evaluate/predict 시 후처리-지표 연결.
- **`train_qa_seq2seq.py` (seq2seq)**:
  - HF `AutoModelForSeq2SeqLM` + `Seq2SeqTrainer`.
  - 입력을 “question: Q context: C” 포맷으로 구성해 인코딩, 타깃은 `answers.text[0]`로 라벨링.
  - 평가 후처리에서 `batch_decode`로 텍스트 복원, `eval_predictions.json` 저장.
  - metric(`squad`/`squad_v2`), generation 설정(max_length/num_beams) 반영.
- **`trainer_qa.py` / `trainer_seq2seq_qa.py`**:
  - HF `Trainer`/`Seq2SeqTrainer`를 상속, `evaluate/predict`에서 임시로 metric 비활성화 → 예측 산출 → `post_process_function` 실행 → metric 계산 → 로깅 형식 일원화.
- **`utils_qa.py`**:
  - Extractive QA 후처리: start/end logits에서 상위 후보를 offset 매핑으로 원문 substring으로 변환, n-best와 확률 산출, `version_2_with_negative` 지원.
  - `postprocess_qa_predictions_with_beam_search`: beam search 기반 모델용(예: XLNet 스타일) 변형 로직도 포함.
- **`serve_qa.py` (extractive Flask)**:
  - `AutoTokenizer` + `AutoModelForQuestionAnswering` 로드, start/end argmax로 답 스팬 디코딩.
  - 선택적으로 softmax 기반 score 계산(normalized).
  - `chrisbase.io.paths`로 `checkpoint-*` 패턴 중 최신 경로 자동 선택, `POST /api`로 질의/문맥 받음.
- **`serve_qa_seq2seq.py` (seq2seq Flask)**:
  - `generate`(num_beams, max_length)로 답 생성, `output_scores`로 토큰 확률 곱을 score로 계산.
- **`infer_qa.py` / `infer_qa_seq2seq.py`**:
  - 로컬 예시 컨텍스트와 여러 질문에 대해 간단 추론 데모(최신 체크포인트 자동 선택).
- **`evaluate-KorQuAD-v1.py`**:
  - KorQuAD v1 공식 형식 평가(EM/F1). 한국어 특수기호/공백 처리 포함한 normalize로 견고한 점수 계산.
- **쉘 스크립트들**: `train_qa-*.sh`, `eval_qa-*.sh`, seq2seq 버전 등은 위 파이썬 스크립트들에 대한 실행 래퍼.

### 데이터/전처리/후처리 포인트
- **데이터 포맷**: 로컬 파일 `csv/json/jsonl` 지원(jsonl 시 `datasets.load_dataset`에서 field=None로 처리).
- **길이 초과 대응**: 긴 context는 `doc_stride`로 overlap을 두고 여러 feature로 분할, evaluation에서 feature→example 매핑으로 후처리.
- **후처리(extractive)**: start/end 상위 인덱스 조합을 필터링(길이/맥스컨텍스트/offset 유효성) 후 원문 substring 복구, softmax로 확률화, n-best/최종 예측 저장.

### 학습/평가 흐름(공통)
- 공통 인자 파싱(`HfArgumentParser`) → 모델/토크나이저 로드 → 데이터셋 로드/전처리 → Trainer 구성(`post_process_function`, `compute_metrics`) → train/eval/predict → 메트릭과 예측 저장.
- 기존 체크포인트 존재 시 `get_last_checkpoint`로 재개(resume) 처리.

### 서빙/추론
- Flask 기반 UI(`templates` HTML) + REST API(`/api` JSON).
- Extractive: logits argmax 스팬 디코딩(+ 선택적 softmax score).
- Seq2Seq: `generate` 결과 디코딩, `output_scores`로 token-level 확률에서 종합 score.

### 실행 예시(요약)
- 학습(extractive): 
  - `python task3-qa/train_qa.py --train_file data/train.json --validation_file data/valid.json --output_dir output/korquad --do_train --do_eval`
- 학습(seq2seq): 
  - `python task3-qa/train_qa_seq2seq.py --train_file data/train.json --validation_file data/valid.json --output_dir output/korquad --do_train --do_eval --predict_with_generate`
- 서빙:
  - `python task3-qa/serve_qa.py serve --pretrained "output/korquad/train_qa-*/checkpoint-*"`
  - `python task3-qa/serve_qa_seq2seq.py serve --pretrained "output/korquad/train_qa_seq2seq-*/checkpoint-*"`

- 핵심 요지
  - 두 파이프라인(extractive/seq2seq)을 동일한 틀(HF Trainer, datasets, evaluate)로 구성.
  - 전처리에서 overflow/stride로 feature 생성, 후처리에서 example 단위로 복원해 metric 계산.
  - 간단한 Flask 서버로 최신 체크포인트를 자동 로드해 즉시 데모 가능.