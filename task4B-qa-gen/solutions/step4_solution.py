# === Step 4 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 4 TODO 해답:

ModelArguments 설정:

1. 사전학습 T5 모델 경로 설정:
   model_name_or_path: str = field(
       default="paust/pko-t5-base",
       metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
   )

2. 설정 파일 경로:
   config_name: Optional[str] = field(
       default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
   )

3. 토크나이저 경로:
   tokenizer_name: Optional[str] = field(
       default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
   )

4. Fast tokenizer 사용 여부:
   use_fast_tokenizer: bool = field(default=True)

DataTrainingArguments 설정:

5. 데이터셋 관련 설정:
   dataset_name: Optional[str] = field(default=None)

6. 컬럼명 설정:
   context_column: Optional[str] = field(default="context")
   question_column: Optional[str] = field(default="question")
   answer_column: Optional[str] = field(default="answers")

7. 학습/검증 파일 경로:
   train_file: Optional[str] = field(default="data/KorQuAD_v1.0_train.json")
   validation_file: Optional[str] = field(default="data/KorQuAD_v1.0_dev.json")

8. Seq2Seq QA 특화 파라미터들:
   max_seq_length: int = field(default=512)
   max_answer_length: int = field(default=50)
   val_max_answer_length: Optional[int] = field(default=None)

9. 생성 관련 파라미터:
   num_beams: Optional[int] = field(default=None)
   ignore_pad_token_for_loss: bool = field(default=True)

main() 함수 구현:

10. Seq2Seq 인수 파서 생성:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

11. 명령행 인수 파싱:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

12. 재현 가능한 실험을 위한 시드 설정:
    set_seed(training_args.seed)

13. Hugging Face Hub에서 데이터셋 로드:
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )

14. 로컬 파일에서 데이터셋 로드:
    data_files["train"] = data_args.train_file
    data_files["validation"] = data_args.validation_file

15. JSON 형식으로 데이터셋 로드:
    raw_datasets = load_dataset("json", data_files=data_files)

16. T5 모델 설정 로드:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

17. T5 토크나이저 로드:
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

18. T5 Seq2Seq 모델 로드:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

19. 토큰 임베딩 크기 조정:
    model.resize_token_embeddings(len(tokenizer))

20. 최대 시퀀스 길이 조정:
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else False

21. T5 입력 형식으로 변환:
    inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

22. 입력 토크나이제이션:
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding,
        truncation=True
    )

23. 타겟 토크나이제이션:
    labels = tokenizer(
        text_target=targets,
        max_length=max_answer_length,
        padding=padding,
        truncation=True
    )

24. 패딩 토큰을 -100으로 변경:
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

25. 입력 토크나이제이션 (overflow 토큰 포함):
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

26. 학습 데이터셋 전처리:
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset"
    )

27. 검증 데이터셋 전처리:
    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        preprocess_validation_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset"
    )

28. Seq2Seq 전용 데이터 콜레이터 설정:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

29. SQuAD 메트릭 로드:
    metric = evaluate.load("squad", cache_dir=model_args.cache_dir)

30. 메트릭 계산:
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

31. -100을 패딩 토큰으로 변경:
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

32. 배치 디코딩:
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

33. 평가 형식에 맞게 변환:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]

34. QuestionAnsweringSeq2SeqTrainer 초기화:
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        post_process_function=post_processing_function
    )

35. 학습 실행:
    train_result = trainer.train()

36. 모델 저장:
    trainer.save_model()

37. 평가 실행:
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

핵심 개념:

1. Seq2Seq vs Extractive QA 학습 차이:

   **Extractive QA**:
   - QuestionAnsweringTrainer 사용
   - 위치 예측 손실 (CrossEntropyLoss)
   - 복잡한 후처리 기반 평가
   - BERT 인코더 모델

   **Seq2Seq QA**:
   - Seq2SeqTrainer 사용
   - 생성 손실 (Language Modeling Loss)
   - 텍스트 생성 기반 평가
   - T5 인코더-디코더 모델

2. T5 모델 특성:
   - Text-to-Text Transfer Transformer
   - 모든 태스크를 텍스트 생성으로 통일
   - Encoder-Decoder 구조
   - Prefix 기반 태스크 구분

3. Seq2SeqTrainingArguments:
   - TrainingArguments 확장
   - 생성 관련 파라미터 추가:
     - generation_max_length: 생성 최대 길이
     - generation_num_beams: Beam Search 크기
     - predict_with_generate: 생성 기반 평가

4. DataCollatorForSeq2Seq:
   - 인코더와 디코더 입력 모두 처리
   - 동적 패딩 지원
   - 라벨 마스킹 (-100) 처리
   - T5 특화 데이터 처리

5. 생성 기반 평가:
   - model.generate() 사용
   - 실제 텍스트 생성 후 평가
   - BLEU, ROUGE, SQuAD 메트릭 활용
   - 느리지만 실제 성능 반영

6. 학습 파라미터 가이드:
   - learning_rate: 1e-4 ~ 5e-4 (T5는 높게)
   - max_seq_length: 512 (T5는 길게 처리 가능)
   - max_answer_length: 50 (생성 답변 길이)
   - num_beams: 4-8 (평가 시 beam search)
   - batch_size: 8-16 (메모리 제약으로 작게)

7. 메모리 고려사항:
   - 인코더-디코더로 메모리 사용량 높음
   - 생성 과정에서 추가 메모리 필요
   - Gradient Checkpointing 활용 권장
   - 더 작은 배치 사이즈 사용

8. 학습 최적화 전략:
   - Mixed Precision (fp16) 활용
   - Gradient Accumulation 사용
   - DataParallel/DistributedDataParallel
   - Early Stopping으로 과적합 방지

실무 배포 고려사항:

1. 모델 압축:
   - 양자화 (Quantization)
   - 지식 증류 (Knowledge Distillation)
   - 프루닝 (Pruning)

2. 추론 최적화:
   - ONNX 변환
   - TensorRT 최적화
   - 배치 처리 개선

3. 서비스 안정성:
   - 생성 길이 제한
   - 타임아웃 설정
   - 에러 핸들링

활용 전략:
- 교육: 설명형 QA 시스템
- 고객지원: 복합 문제 해결
- 연구: 논문 요약 및 해석
- 창작: 스토리텔링 지원

성능 벤치마크:
- SQuAD 1.1/2.0 평가
- KorQuAD 벤치마크
- 도메인별 평가 데이터
- 사용자 만족도 측정
"""
