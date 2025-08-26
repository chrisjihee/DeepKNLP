# NER 모델 단계별 학습 가이드

## 📋 개요

이 프로젝트는 **Named Entity Recognition (NER)** 모델을 단계별로 학습할 수 있도록 설계된 교육용 자료입니다. 수강생들은 5단계에 걸쳐 NER의 핵심 개념과 구현 방법을 체계적으로 학습할 수 있습니다.

## 🎯 학습 목표

- **NER vs 분류의 차이점** 이해
- **토큰 레벨 라벨링**과 **BIO 태깅** 스킴 학습
- **복잡한 토큰-문자 매핑** 과정 이해
- **다층 평가 메트릭** (토큰/문자/개체 레벨) 활용
- **실시간 NER 웹 서비스** 구축

## 📚 단계별 구성

### 🔷 Step 1: NER 기본 구조 이해
**파일**: `step1_basic_structure.py`

**학습 내용**:
- NER 특화 라이브러리 import
- `NERModel` 클래스 초기화
- 라벨 매핑 딕셔너리 구성
- Fast 토크나이저의 중요성

**핵심 TODO**:
```python
# NER 데이터 코퍼스 초기화
self.data: NERCorpus = # TODO: 완성하세요

# 라벨 → ID 매핑 딕셔너리
self._label_to_id: Dict[str, int] = # TODO: 완성하세요

# AutoModelForTokenClassification 로드
self.lang_model: PreTrainedModel = # TODO: 완성하세요
```

**학습 포인트**:
- `AutoModelForSequenceClassification` vs `AutoModelForTokenClassification`
- BIO 태깅: `['O', 'B-PER', 'I-PER', 'B-LOC', ...]`
- Fast 토크나이저 필수 이유: `token_to_chars()` 메소드

---

### 🔷 Step 2: NER 데이터 처리
**파일**: `step2_data_processing.py`

**학습 내용**:
- NER 전용 데이터셋과 데이터로더
- `encoded_examples_to_batch` collate 함수
- `example_ids`의 역할과 중요성

**핵심 TODO**:
```python
# NER 학습 데이터셋 생성
train_dataset = # TODO: 완성하세요

# NER 전용 collate 함수 사용
collate_fn=self.data.encoded_examples_to_batch,  # TODO: 확인

# validation_step에서 토큰-문자 매핑을 위해 필요
self._infer_dataset = # TODO: 완성하세요
```

**학습 포인트**:
- 분류 vs NER: `data_collator` vs `encoded_examples_to_batch`
- `example_ids`: 복잡한 후처리를 위한 원본 데이터 매핑
- 추론 시 데이터셋 보존의 필요성

---

### 🔷 Step 3: NER 학습/평가 스텝
**파일**: `step3_training_steps.py`

**학습 내용**:
- 토큰 레벨 분류 학습
- 복잡한 validation_step (간소화 버전)
- 단일 텍스트 NER 추론

**핵심 TODO**:
```python
# 토큰 분류 모델 순전파
outputs: TokenClassifierOutput = # TODO: 완성하세요

# 정확도 계산 (패딩 토큰 제외)
acc: torch.Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

# NER 추론을 위한 토크나이즈
inputs = # TODO: 완성하세요
```

**학습 포인트**:
- `training_step`: 상대적으로 단순 (분류와 유사)
- `validation_step`: NER의 핵심 복잡성 (토큰-문자 매핑)
- `ignore_index=0`: 패딩 토큰 제외 평가
- `infer_one`: 실시간 서비스용 단일 문장 처리

---

### 🔷 Step 4: NER 학습 루프
**파일**: `step4_training_loops.py`

**학습 내용**:
- 전체 학습 과정 관리
- NER 특화 검증 메트릭
- 분산 학습 환경에서의 메트릭 수집

**핵심 TODO**:
```python
# 학습 루프: 역전파 및 가중치 업데이트
optimizer.zero_grad()
outputs = model.training_step(batch, i)
fabric.backward(outputs["loss"])
optimizer.step()

# NER 특화 메트릭 계산
"val_F1c": # TODO: 문자 레벨 Macro F1
"val_F1e": # TODO: 개체 레벨 Macro F1
```

**학습 포인트**:
- 3가지 평가 레벨: 토큰/문자/개체
- `fabric.all_gather()`: 분산 환경 메트릭 수집
- 체크포인트 저장: 보통 `val_F1c` 기준
- NER vs 분류: 단순 accuracy vs 다층 F1 메트릭

---

### 🔷 Step 5: CLI와 웹 서비스
**파일**: `step5_cli_and_serving.py`

**학습 내용**:
- 완전한 CLI 명령어 시스템
- Flask 기반 웹 API 서비스
- 실시간 NER 추론 시스템

**핵심 TODO**:
```python
# Flask WebAPI 등록
NERModel.WebAPI.register(route_base="/", app=server, init_argument=self)

# NER API 엔드포인트
response = self.model.infer_one(text=request.json)
return jsonify(response)

# 전체 학습 파이프라인
args = TrainerArguments(...)
fabric = Fabric(...)
train_loop(model, optimizer, ...)
```

**학습 포인트**:
- `train`, `test`, `serve` 세 가지 CLI 명령어
- Flask-Classful: 클래스 기반 웹 API 구조화
- REST API: `POST /api` 엔드포인트
- 실시간 NER 서비스: 토큰별 라벨과 확률 제공

## 🚀 실행 방법

### 1단계부터 순차적 학습:
```bash
# Step 1: 기본 구조 이해
python step1_basic_structure.py

# Step 2: 데이터 처리 (TODO 완성 후)
python step2_data_processing.py

# Step 3: 학습/평가 스텝 (TODO 완성 후)
python step3_training_steps.py

# Step 4: 학습 루프 (TODO 완성 후)
python step4_training_loops.py

# Step 5: 완전한 시스템 (TODO 완성 후)
python step5_cli_and_serving.py train
python step5_cli_and_serving.py test
python step5_cli_and_serving.py serve
```

## 🎓 교육적 장점

### 1. **점진적 복잡도 증가**
- Step 1: 기본 개념 → Step 5: 완전한 시스템
- 각 단계별 명확한 학습 목표와 TODO

### 2. **NER 특화 개념 강조**
- 분류와의 차이점 명확화
- 토큰-문자 매핑의 복잡성 이해
- 다층 평가 메트릭의 필요성

### 3. **실무 중심 학습**
- Production 레벨의 분산 학습
- 웹 서비스 구축 경험
- CLI 도구 개발 스킬

### 4. **해답 제공**
- 각 단계별 solution 파일
- 상세한 개념 설명과 힌트
- 단계별 핵심 포인트 정리

## 🔧 GitHub 배포 전략

### Repository 구조:
```
ner-stepwise-learning/
├── step1_basic_structure.py      # 학생용 (TODO 포함)
├── step2_data_processing.py      # 학생용 (TODO 포함)
├── step3_training_steps.py       # 학생용 (TODO 포함)
├── step4_training_loops.py       # 학생용 (TODO 포함)
├── step5_cli_and_serving.py      # 학생용 (TODO 포함)
├── solutions/                    # 강사 전용 (Private Repository)
│   ├── step1_solution.py
│   ├── step2_solution.py
│   ├── step3_solution.py
│   ├── step4_solution.py
│   └── step5_solution.py
├── data/                         # 예제 데이터
├── templates/                    # 웹 UI 템플릿
└── README_stepwise_learning.md   # 이 파일
```

### 접근 권한 관리:
- **Public Repository**: 학생용 TODO 파일들
- **Private Repository**: 해답 파일들 (강사만 접근)
- **Release 기반**: 각 단계별 태그로 점진적 공개

### 수업 진행 방식:
1. **Step별 순차 공개**: 이전 단계 완료 후 다음 단계 접근
2. **Pull Request 검토**: 학생 제출물 코드 리뷰
3. **실시간 피드백**: GitHub Issues/Discussions 활용
4. **최종 프로젝트**: 완전한 NER 시스템 구축

## 📖 참고 자료

- **BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers"
- **NER Survey**: "Named Entity Recognition: A Literature Survey"
- **BIO Tagging**: "Introduction to the CoNLL-2003 Shared Task"
- **Korean NER**: "KLUE: Korean Language Understanding Evaluation"

## 🔍 핵심 개념 정리

| 개념 | 분류 모델 | NER 모델 |
|------|----------|----------|
| **입력** | 문장 | 문장 |
| **출력** | 단일 라벨 | 토큰별 라벨 |
| **모델** | SequenceClassification | TokenClassification |
| **평가** | accuracy | accuracy + F1c + F1e |
| **복잡도** | 단순 | 복잡 (토큰-문자 매핑) |
| **활용** | 감정분석, 스팸분류 | 개체명인식, 정보추출 |

이 단계별 학습을 통해 수강생들은 NER의 복잡성을 점진적으로 이해하고, 실무에서 활용할 수 있는 완전한 NER 시스템을 구축할 수 있게 됩니다. 🎯
