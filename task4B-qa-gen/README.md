# Question Answering (Generative)

KorQuAD 기반 생성형 QA 실습 태스크입니다.

참고 자료:
- https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

학생 구현 대상:
- task4B-qa-gen/train_qa_seq2seq.py

지원 모듈:
- task4B-qa-gen/trainer_seq2seq_qa.py
- task4B-qa-gen/serve_qa_seq2seq.py

실습 의도:
- 이 태스크도 Hugging Face 공식 예시를 한국어 생성형 QA에 맞게 조정한 구조를 배우는 것이 중요합니다.
- 학생은 `train_qa_seq2seq.py`를 중심으로 구현하고, trainer 모듈은 제공 인프라로 사용합니다.

Step 1:
- 목표: 데이터 로딩, question-context formatting, tokenizer preprocessing 이해
- 실행 예시:
```bash
python task4B-qa-gen/train_qa_seq2seq.py --model_name_or_path paust/pko-t5-base --train_file data/korquad/train-half.jsonl --validation_file data/korquad/validation.jsonl --output_dir output/korquad-seq2seq-lab --do_eval --max_eval_samples 4 --predict_with_generate
```

Step 2:
- 목표: Seq2SeqTrainer 기반 train/eval/predict 수행
- 실행 예시:
```bash
python task4B-qa-gen/train_qa_seq2seq.py --model_name_or_path paust/pko-t5-base --train_file data/korquad/train-half.jsonl --validation_file data/korquad/validation.jsonl --output_dir output/korquad-seq2seq-lab --do_train --do_eval --predict_with_generate
```

Step 3:
- 목표: 학습된 체크포인트를 불러 웹 서빙 수행
- 실행 예시:
```bash
python task4B-qa-gen/serve_qa_seq2seq.py serve --pretrained "output/korquad-seq2seq-lab/checkpoint-*"
```

해답 파일:
- task4B-qa-gen/solutions/step1_solution.py
- task4B-qa-gen/solutions/step2_solution.py
- task4B-qa-gen/solutions/step3_solution.py
- task4B-qa-gen/solutions/train_qa_seq2seq_reference.py
