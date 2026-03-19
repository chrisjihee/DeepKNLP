# Question Answering (Extractive)

KorQuAD 기반 추출형 QA 실습 태스크입니다.

참고 자료:
- https://ratsgo.github.io/nlpbook/docs/qa
- https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

학생 구현 대상:
- task4A-qa-ext/train_qa.py

지원 모듈:
- task4A-qa-ext/trainer_qa.py
- task4A-qa-ext/utils_qa.py
- task4A-qa-ext/serve_qa.py

실습 의도:
- 이 태스크는 Hugging Face 공식 예시를 한국어 QA에 맞게 조정한 구조를 배우는 것이 중요합니다.
- 학생은 `train_qa.py`를 중심으로 구현하고, `trainer_qa.py`와 `utils_qa.py`는 제공된 지원 모듈로 가져다 사용합니다.
- 실제로 채워야 하는 핵심 블록은 helper 함수 형태의 TODO로 비워 두었습니다.

Step 1:
- 목표: HF 예시의 전체 구조, 데이터 로딩, 토크나이저, feature preprocessing 이해
- 구현 포인트: `complete_step1_load_raw_datasets`, `complete_step1_load_model_bundle`
- 실행 예시:
```bash
python task4A-qa-ext/train_qa.py --model_name_or_path klue/bert-base --train_file data/korquad/train-half.jsonl --validation_file data/korquad/validation.jsonl --output_dir output/korquad-lab --do_eval --max_eval_samples 4
```

Step 2:
- 목표: Trainer, post-processing, metric 계산을 연결해서 train/eval/predict 수행
- 지원 모듈: `trainer_qa.py`, `utils_qa.py`
- 구현 포인트: `complete_step2_build_trainer`, `complete_step2_run_train`, `complete_step2_run_eval`, `complete_step2_run_predict`
- 실행 예시:
```bash
python task4A-qa-ext/train_qa.py --model_name_or_path klue/bert-base --train_file data/korquad/train-half.jsonl --validation_file data/korquad/validation.jsonl --output_dir output/korquad-lab --do_train --do_eval
```

Step 3:
- 목표: 학습된 체크포인트를 불러 웹 서빙 수행
- 구현 포인트: `complete_step3_build_serve_hint`로 서빙 명령을 확인한 뒤 `serve_qa.py`를 사용합니다.
- 실행 예시:
```bash
python task4A-qa-ext/serve_qa.py serve --pretrained "output/korquad-lab/checkpoint-*"
```

해답 파일:
- 각 `stepX_solution.py`는 전체 정답 파일이 아니라, 해당 단계에서 채워야 할 helper 답안만 담고 있습니다.
- task4A-qa-ext/solutions/step1_solution.py
- task4A-qa-ext/solutions/step2_solution.py
- task4A-qa-ext/solutions/step3_solution.py
- task4A-qa-ext/solutions/train_qa_reference.py
