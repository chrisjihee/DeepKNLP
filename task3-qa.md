### 목적과 전체 구조
- **역할**: `task3-qa`는 KorQuAD 등 한국어 QA를 대상으로 학습(Training), 평가(Evaluation), 추론(Inference), 서빙(Serving)까지 end-to-end 파이프라인을 제공합니다.
- **두 가지 파이프라인**: 
  - **Extractive QA**: 문맥에서 답변 span의 시작/끝 위치를 예측.
  - **Seq2Seq QA**: 질문-문맥을 입력으로 받아 답변 텍스트를 생성.

### 파일 구성 한눈에 보기
- **학습**
  - Extractive: `train_qa.py`
  - Seq2Seq: `train_qa_seq2seq.py`
- **Trainer 서브클래스**
  - Extractive: `trainer_qa.py`
  - Seq2Seq: `trainer_seq2seq_qa.py`
- **후처리/유틸**
  - `utils_qa.py` (Extractive 후처리: n-best, offset 매핑 등)
- **추론(데모)**
  - Extractive: `infer_qa.py`
  - Seq2Seq: `infer_qa_seq2seq.py`
- **서빙**
  - Extractive: `serve_qa.py` (Flask)
  - Seq2Seq: `serve_qa_seq2seq.py` (Flask)
- **평가**
  - KorQuAD v1 전용: `evaluate-KorQuAD-v1.py`
- **쉘 스크립트**
  - `train_qa-*.sh`, `eval_qa-*.sh`, `*_seq2seq-*.sh` 등 실행 래퍼

### 데이터 입출력과 전처리/후처리 공통 포인트
- **데이터 포맷**: `csv/json/jsonl` 지원. `jsonl`은 `datasets.load_dataset` 호출 시 `field=None`로 취급되도록 처리됨.
- **길이 처리**: 긴 문맥을 `doc_stride`로 겹치게 분할하고, feature→example 역매핑으로 평가 시 재조립.
- **메트릭**: `evaluate`의 `squad`/`squad_v2`를 사용해 EM/F1 계산.
- **Extractive 후처리(`utils_qa.py`)**:
  - `postprocess_qa_predictions`: start/end logits에서 상위 인덱스 조합을 필터링(유효 offset/길이/max_context) 후 원문 substring 복구, softmax로 확률화, n-best/최종 예측 저장. `version_2_with_negative`와 `null_score_diff_threshold` 지원.
  - `postprocess_qa_predictions_with_beam_search`: beam search 기반 모델용 변형.
- **Seq2Seq 후처리(`train_qa_seq2seq.py`)**:
  - `generate` 결과 토큰을 디코딩해 예측 텍스트로 변환, example 매핑으로 묶어 `eval_predictions.json` 저장.

### 학습 스크립트
- **Extractive: `train_qa.py`**
  - 모델/토크나이저: `AutoModelForQuestionAnswering`, `AutoTokenizer`.
  - 데이터 전처리: 질문 좌측 공백 제거, `truncation="only_second"`(padding side에 따라 반전), `return_overflowing_tokens/offset_mapping`으로 span 학습/평가 세트 생성.
  - 후처리+지표: `postprocess_qa_predictions` → `squad`/`squad_v2`로 metric 계산.
  - Trainer: `QuestionAnsweringTrainer`를 사용해 evaluate/predict 단계에서 후처리→지표를 자동 연결.
  - 체크포인트 재개: `get_last_checkpoint` 지원.
- **Seq2Seq: `train_qa_seq2seq.py`**
  - 모델/토크나이저: `AutoModelForSeq2SeqLM`, `AutoTokenizer`.
  - 입력 포맷: `"question: {Q} context: {C}"`, 라벨은 `answers.text[0]`(없으면 빈 문자열).
  - 학습/평가: `Seq2SeqTrainer` 서브클래스 `QuestionAnsweringSeq2SeqTrainer` 사용. `predict_with_generate` 시 `max_length/num_beams` 반영.
  - 후처리: 배치 디코딩 후 example 매핑으로 dict 구성, `eval_predictions.json` 저장.

### Trainer 서브클래스
- **`trainer_qa.py` (Extractive)**:
  - `evaluate/predict`에서 임시로 metric 계산을 비활성 → 루프 실행해 로짓 수집 → `post_process_function` 호출 → metric 계산/로깅.
- **`trainer_seq2seq_qa.py` (Seq2Seq)**:
  - `Seq2SeqTrainer` 기반. `evaluate/predict`에서도 동일 패턴으로 후처리-지표 계산을 연결. generation 인자(`max_length`, `num_beams`)를 반영.

### 추론 스크립트(데모)
- **`infer_qa.py` (Extractive)**
  - **체크포인트 자동 선택**: `chrisbase.io.paths`로 `"output/korquad/train_qa-*/checkpoint-*"` 패턴 중 최신 경로를 선택.
  - **로딩**: `AutoTokenizer.from_pretrained`, `AutoModelForQuestionAnswering.from_pretrained`.
  - **예시 입력**: 한국 관련 단락을 `context`로, 다수의 질문 리스트를 제공.
  - **추론 로직**:
    - `tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, padding=True)`
    - 모델 forward → `start_logits`, `end_logits` argmax로 span 인덱스 결정 → 해당 구간 `input_ids`를 `tokenizer.decode(...)`로 디코딩해 답 구성.
  - **출력**: 각 질문에 대한 Answer를 stdout으로 출력.
- **`infer_qa_seq2seq.py` (Seq2Seq)**
  - **체크포인트 자동 선택**: `"output/korquad/train_qa_seq2seq-*/checkpoint-*"` 최신 경로 선택.
  - **로딩**: `AutoTokenizer`, `AutoModelForSeq2SeqLM`.
  - **예시 입력**: 동일한 `context`와 질문 리스트.
  - **추론 로직**:
    - 입력을 `f"question: {question} context: {context}"`로 구성 → 토크나이즈.
    - `model.generate(..., max_length=50, num_beams=5)` → `tokenizer.decode(..., skip_special_tokens=True)`로 답 생성.
  - **출력**: 각 질문에 대한 생성 답변을 stdout으로 출력.

### 서빙(Flask)
- **`serve_qa.py` (Extractive)**
  - **모델**: `AutoModelForQuestionAnswering`. latest 체크포인트 자동 선택.
  - **API**:
    - `GET /`: 템플릿(`serve_qa.html`) 렌더.
    - `POST /api`: `{question, context}` JSON 입력 → 위 extractive 방식으로 답/score 계산 후 반환.
  - **점수**: 옵션에 따라 softmax 확률 기반 score 또는 로짓 합 기반 score.
- **`serve_qa_seq2seq.py` (Seq2Seq)**
  - **모델**: `AutoModelForSeq2SeqLM`. latest 체크포인트 자동 선택.
  - **API**:
    - `GET /`: 템플릿(`serve_qa_seq2seq.html`) 렌더.
    - `POST /api`: `{question, context}` JSON 입력 → `generate`로 답 생성, token-level 확률(`output_scores=True`)을 곱해 score 산출.

### 평가 스크립트
- **`evaluate-KorQuAD-v1.py`**
  - KorQuAD v1 포맷 전용. 한국어 기호/공백 정규화 후 **문자 단위 F1**과 EM 계산.
  - `dataset.json`과 `predictions.json`을 입력받아 `{exact_match, f1}` 출력.

### 실행 예시
- **Extractive 학습/평가**
  - `python task3-qa/train_qa.py --train_file data/train.json --validation_file data/valid.json --output_dir output/korquad --do_train --do_eval`
- **Seq2Seq 학습/평가**
  - `python task3-qa/train_qa_seq2seq.py --train_file data/train.json --validation_file data/valid.json --output_dir output/korquad --do_train --do_eval --predict_with_generate`
- **추론 데모**
  - `python task3-qa/infer_qa.py`
  - `python task3-qa/infer_qa_seq2seq.py`
- **서빙**
  - `python task3-qa/serve_qa.py serve --pretrained "output/korquad/train_qa-*/checkpoint-*"`
  - `python task3-qa/serve_qa_seq2seq.py serve --pretrained "output/korquad/train_qa_seq2seq-*/checkpoint-*"`

### 실전 팁
- **jsonl 지원**: 로컬 데이터가 jsonl이면 별도 `field` 없이 바로 로드되도록 이미 처리되어 있습니다.
- **길이/stride 튜닝**: `max_seq_length`, `doc_stride`로 long context 대응 성능과 속도를 trade-off.
- **Seq2Seq 생성 품질**: `max_length`, `num_beams`로 제어. 학습 시 `predict_with_generate`를 켜면 평가 단계에서 생성 기반 metric을 즉시 산출합니다.
- **Extractive 안정성**: `n_best_size`, `max_answer_length`, `version_2_with_negative`/`null_score_diff_threshold`로 무답 처리 품질을 조절.

- 요약: 학습(Extractive/Seq2Seq)→전처리/후처리→지표→추론 데모(`infer_qa*.py`)→웹 서빙(`serve_qa*.py`)까지 일관된 HF Trainer/Datasets/Evaluate 스택으로 구성되어 있으며, 최신 체크포인트 자동 선택과 한국어 평가 정규화가 실전에 맞게 제공됩니다.