# NSMC 감성분석 단계별 학습 가이드

## 📋 개요

이 디렉토리는 `run_cls.py`의 NSMC 감성분석 코드를 수강생들이 단계적으로 완성하면서 배울 수 있도록 구성된 교육용 자료입니다.

## 🎯 학습 목표

- PyTorch와 Hugging Face Transformers를 활용한 분류 모델 구현
- Lightning Fabric을 이용한 분산 학습 이해
- CLI 프레임워크(Typer)와 웹 서비스(Flask) 개발
- 체크포인트 관리와 모델 평가 방법론

## 📁 파일 구조

```
task1-cls/
├── step1_basic_structure.py      # 1단계: 기본 구조와 모델 초기화
├── step2_data_processing.py      # 2단계: 데이터 파이프라인 구현
├── step3_training_steps.py       # 3단계: 학습/평가 스텝 구현
├── step4_training_loops.py       # 4단계: 전체 학습 루프 구현
├── step5_cli_and_serving.py      # 5단계: CLI와 웹 서비스 구현
├── solutions/                    # 단계별 해답
│   ├── step1_solution.py
│   ├── step2_solution.py
│   ├── step3_solution.py
│   ├── step4_solution.py
│   └── step5_solution.py
├── data/                         # 예제 데이터
├── templates/                    # 웹 UI 템플릿
└── README_stepwise_learning.md   # 이 파일
```

## 🚀 단계별 학습 과정

### 1단계: 기본 구조 이해 (`step1_basic_structure.py`)

**학습 목표:**
- 라이브러리 Import의 역할 이해
- LightningModule 클래스 구조 파악
- Hugging Face 모델 컴포넌트 초기화

**완성할 TODO:**
- NSMC 데이터 코퍼스 초기화
- 사전학습 모델 설정 로드
- 토크나이저 로드
- 분류용 사전학습 모델 로드

**핵심 개념:**
- `AutoConfig`, `AutoTokenizer`, `AutoModelForSequenceClassification`
- 사전학습 모델의 fine-tuning을 위한 설정

### 2단계: 데이터 처리 (`step2_data_processing.py`)

**학습 목표:**
- PyTorch 데이터 파이프라인 구현
- 학습/검증/테스트 데이터로더 차이점 이해
- 배치 처리와 샘플링 전략

**완성할 TODO:**
- AdamW 옵티마이저 설정
- 학습/검증/테스트 데이터셋 생성
- 각각에 맞는 데이터로더 구성

**핵심 개념:**
- `RandomSampler` vs `SequentialSampler`
- `batch_size`, `num_workers`, `collate_fn`
- 학습과 추론 시 다른 배치 크기 사용

### 3단계: 학습/평가 스텝 (`step3_training_steps.py`)

**학습 목표:**
- Forward/Backward pass 이해
- 학습과 평가 모드의 차이점
- 단일 텍스트 추론 구현

**완성할 TODO:**
- `training_step`: loss와 accuracy 계산
- `validation_step`: 예측값과 라벨 수집
- `test_step`: 검증과 동일한 로직
- `infer_one`: 웹 서비스용 단일 추론

**핵심 개념:**
- `@torch.no_grad()` 데코레이터
- logits을 확률로 변환 (`softmax`)
- 배치 단위 vs 전체 데이터 평가

### 4단계: 학습 루프 (`step4_training_loops.py`)

**학습 목표:**
- 전체 학습 과정의 흐름 이해
- 분산 학습에서의 메트릭 동기화
- 체크포인트 저장과 검증 주기

**완성할 TODO:**
- 에포크와 배치 루프 구현
- gradient 계산과 가중치 업데이트
- 분산 환경에서 메트릭 수집
- 정기적 검증 실행

**핵심 개념:**
- `fabric.all_gather()`: 분산 메트릭 수집
- `optimizer.zero_grad()`, `fabric.backward()`, `optimizer.step()`
- 학습 중 정기적 검증의 중요성

### 5단계: CLI와 서빙 (`step5_cli_and_serving.py`)

**학습 목표:**
- 명령줄 인터페이스 구현
- 웹 서비스 API 개발
- 설정 관리와 환경 분기

**완성할 TODO:**
- Flask 웹 API 구현
- Typer CLI 명령어 완성
- 설정 인수 매핑
- 환경별 Fabric 설정

**핵심 개념:**
- Flask-Classful을 이용한 REST API
- Typer의 타입 힌트 기반 CLI
- Lightning Fabric의 다양한 실행 환경

## 💡 수업 진행 방법

### 준비 단계
1. 각 단계별 파일을 수강생들에게 제공
2. 필요한 의존성 설치 확인
3. NSMC 데이터셋 다운로드

### 단계별 진행
1. **강의**: 해당 단계의 핵심 개념 설명
2. **실습**: 수강생들이 TODO 부분 완성
3. **토론**: 구현 과정에서의 질문과 답변
4. **해답 공개**: solution 파일을 통한 정답 확인
5. **심화**: 다음 단계로 넘어가기 전 추가 설명

### 평가 방법
- 각 단계별 TODO 완성도
- 코드 실행 가능성
- 핵심 개념 이해도

## 🔧 실행 방법

### 각 단계별 독립 실행 불가
각 단계 파일은 교육 목적으로 일부 기능만 포함하므로 독립적으로 실행할 수 없습니다.

### 완성된 코드 실행
모든 단계를 완성한 후에는 원본 `run_cls.py`와 동일한 방식으로 실행:

```bash
# 학습
python step5_cli_and_serving.py train --num_epochs 3

# 테스트  
python step5_cli_and_serving.py test

# 서빙
python step5_cli_and_serving.py serve --server_port 9164
```

## 📚 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Lightning Fabric](https://lightning.ai/docs/fabric/)
- [Typer 문서](https://typer.tiangolo.com/)
- [Flask 공식 문서](https://flask.palletsprojects.com/)

## ❓ 자주 묻는 질문

**Q: 중간 단계에서 코드를 실행해볼 수 있나요?**
A: 각 단계는 완전한 형태가 아니므로 직접 실행은 어렵습니다. 대신 solution 파일을 참고하여 완성 후 테스트하세요.

**Q: TODO 부분을 다르게 구현해도 되나요?**
A: 네, 핵심 개념을 이해하고 동일한 결과를 얻는다면 다른 방식으로 구현해도 좋습니다.

**Q: 에러가 발생하면 어떻게 하나요?**
A: solution 파일과 비교하여 문제점을 찾거나, 에러 메시지를 통해 디버깅하세요.

## 📈 학습 효과

이 단계별 학습을 통해 다음을 얻을 수 있습니다:

1. **체계적 이해**: 전체 시스템을 단계적으로 구축하며 각 부분의 역할 파악
2. **실전 경험**: 실제 프로덕션 레벨의 코드 작성 경험
3. **문제 해결**: TODO를 완성하며 자연스러운 디버깅 능력 향상
4. **확장 가능성**: 학습한 패턴을 다른 NLP 태스크에 적용

이 가이드가 Deep Learning과 NLP 학습에 도움이 되길 바랍니다! 🚀
