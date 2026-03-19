# Sequence Labelling

한국어 NER 실습 태스크입니다.

참고 자료:
- https://ratsgo.github.io/nlpbook/docs/ner

학생 구현 대상:
- task2-ner/run_ner.py

실습 방식:
- 이 태스크는 하나의 파일에서 Step 1, Step 2, Step 3를 누적해서 완성합니다.
- 파일 안의 `TODO Step 1`, `TODO Step 2`, `TODO Step 3` 표시를 따라가면 됩니다.
- 실제로 채워야 하는 핵심 블록은 helper 함수 형태로 비워 두었습니다.
- CLI 옵션은 일부러 많이 남겨 두었습니다. 모델/데이터/하드웨어 옵션의 폭을 보는 것도 학습 목표입니다.

Step 1:
- 목표: 라벨 맵, fast tokenizer, 데이터 전처리, token-char alignment 이해
- 구현 포인트: `complete_step1_model_setup`, `complete_step1_train_dataloader`
- 실행 예시:
```bash
python task2-ner/run_ner.py train --data_home data --data_name klue-ner --pretrained klue/roberta-base --num_epochs 0
```
- 기대 결과: 데이터셋/데이터로더/모델 로딩이 성공하고 학습 직전까지 실행됩니다.

Step 2:
- 목표: 학습, 문자 단위 평가, 테스트, 추론 흐름 완성
- 구현 포인트: `complete_step2_training_batch`, `complete_step2_validation_batch`
- 실행 예시:
```bash
python task2-ner/run_ner.py train --data_home data --data_name klue-ner --pretrained klue/roberta-base --num_epochs 1
python task2-ner/run_ner.py test --data_home data --data_name klue-ner --pretrained klue/roberta-base
```
- 기대 결과: 체크포인트 저장과 `F1c`, `F1e` 평가 로그를 확인할 수 있습니다.

Step 3:
- 목표: 웹 서빙 완성
- 구현 포인트: `complete_step3_tokenize_text`, `complete_step3_format_inference`
- 실행 예시:
```bash
python task2-ner/run_ner.py serve --data_home data --data_name klue-ner --pretrained klue/roberta-base --server_page serve_ner.html
```
- 기대 결과: 브라우저에서 토큰별 예측 결과를 확인할 수 있습니다.

해답 파일:
- 각 `stepX_solution.py`는 전체 정답 파일이 아니라, 해당 단계에서 채워야 할 helper 답안만 담고 있습니다.
- task2-ner/solutions/step1_solution.py
- task2-ner/solutions/step2_solution.py
- task2-ner/solutions/step3_solution.py
- task2-ner/solutions/run_ner_reference.py
