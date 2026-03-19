# Sentence Classification

NSMC 감성분석 실습 태스크입니다.

참고 자료:
- https://ratsgo.github.io/nlpbook/docs/doc_cls

학생 구현 대상:
- task1-cls/run_cls.py

실습 방식:
- 이 태스크는 하나의 파일에서 Step 1, Step 2, Step 3를 누적해서 완성합니다.
- 파일 안의 `TODO Step 1`, `TODO Step 2`, `TODO Step 3` 표시를 따라가면 됩니다.
- 실제로 채워야 하는 핵심 블록은 원래 코드 흐름 안에 in-place 형태로 비워 두었습니다.
- CLI 옵션은 일부러 많이 남겨 두었습니다. 전체 학습 설정의 폭을 체감하는 것도 목표입니다.

Step 1:
- 목표: 모델/토크나이저/데이터 로딩과 전처리 흐름 이해
- 구현 포인트: `NSMCModel.__init__`, `NSMCModel.train_dataloader`
- 실행 예시:
```bash
python task1-cls/run_cls.py train --data_home data --data_name nsmc --pretrained beomi/KcELECTRA-base --num_epochs 0
```
- 기대 결과: 데이터셋/데이터로더/모델 로딩이 성공하고 학습 직전까지 실행됩니다.

Step 2:
- 목표: 학습, 검증, 테스트, 추론 흐름 완성
- 구현 포인트: `NSMCModel.training_step`, `NSMCModel.validation_step`
- 실행 예시:
```bash
python task1-cls/run_cls.py train --data_home data --data_name nsmc --pretrained beomi/KcELECTRA-base --num_epochs 1
python task1-cls/run_cls.py test --data_home data --data_name nsmc --pretrained beomi/KcELECTRA-base
```
- 기대 결과: 체크포인트 저장과 평가 로그를 확인할 수 있습니다.

Step 3:
- 목표: 웹 서빙 완성
- 구현 포인트: `NSMCModel.infer_one`
- 실행 예시:
```bash
python task1-cls/run_cls.py serve --data_home data --data_name nsmc --pretrained beomi/KcELECTRA-base --server_page serve_cls.html
```
- 기대 결과: 브라우저에서 감성분석 데모를 사용할 수 있습니다.

해답 파일:
- 각 `stepX_solution.py`는 전체 정답 파일이 아니라, 해당 단계에서 채워야 할 in-place 블록 스니펫만 담고 있습니다.
- task1-cls/solutions/step1_solution.py
- task1-cls/solutions/step2_solution.py
- task1-cls/solutions/step3_solution.py
- task1-cls/solutions/run_cls_reference.py
