# === Step 4 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 4 TODO 해답:

ModelArguments 설정:

1. 사전학습 모델 경로 설정:
   model_name_or_path: str = field(
       default="monologg/koelectra-base-v3",
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

DataTrainingArguments 설정:

4. 데이터셋 관련 설정:
   dataset_name: Optional[str] = field(
       default=None,
       metadata={"help": "The name of the dataset to use (via the datasets library)."}
   )

5. 학습 파일 경로:
   train_file: Optional[str] = field(
       default="data/KorQuAD_v1.0_train.json",
       metadata={"help": "The input training data file (a json file)."}
   )

6. 검증 파일 경로:
   validation_file: Optional[str] = field(
       default="data/KorQuAD_v1.0_dev.json",
       metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a json file)."}
   )

7. QA 특화 파라미터들:
   max_seq_length: int = field(default=384)
   doc_stride: int = field(default=128)
   n_best_size: int = field(default=20)
   max_answer_length: int = field(default=30)
   version_2_with_negative: bool = field(default=False)

main() 함수 구현:

8. HfArgumentParser 생성:
   parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

9. 명령행 인수 파싱:
   model_args, data_args, training_args = parser.parse_args_into_dataclasses()

10. 재현 가능한 실험을 위한 시드 설정:
    set_seed(training_args.seed)

11. Hugging Face Hub에서 데이터셋 로드:
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )

12. 로컬 파일에서 데이터셋 로드:
    data_files["train"] = data_args.train_file
    data_files["validation"] = data_args.validation_file

13. JSON 형식으로 데이터셋 로드:
    raw_datasets = load_dataset("json", data_files=data_files)

14. 모델 설정 로드:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

15. 토크나이저 로드:
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

16. QA 모델 로드:
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

17. 학습 데이터셋 가져오기:
    train_dataset = raw_datasets["train"]

18. 전처리 적용:
    train_dataset = train_dataset.map(
        prepare_train_features_with_args,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )

19. 검증 데이터셋 가져오기 및 전처리:
    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        prepare_validation_features_with_args,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

20. 데이터 콜레이터 선택:
    data_collator = (
        default_data_collator 
        if data_args.pad_to_max_length 
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

21. SQuAD 메트릭 로드:
    metric = evaluate.load(
        "squad_v2" if data_args.version_2_with_negative else "squad",
        cache_dir=model_args.cache_dir
    )

22. 메트릭 계산:
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

23. QuestionAnsweringTrainer 초기화:
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

24. 학습 실행:
    train_result = trainer.train()

25. 모델 저장:
    trainer.save_model()

26. 평가 실행:
    metrics = trainer.evaluate()

핵심 개념:

1. Hugging Face 생태계 활용:
   - 표준화된 인수 파싱 (HfArgumentParser)
   - 자동 모델/토크나이저 로딩 (Auto* 클래스)
   - 통합된 학습 파이프라인 (Trainer)

2. QA 특화 설정:
   - max_seq_length=384: BERT 최적 길이
   - doc_stride=128: 긴 문서 처리를 위한 겹침
   - n_best_size=20: 후처리에서 고려할 후보 수
   - max_answer_length=30: 한국어 답변 평균 길이

3. 데이터 처리 파이프라인:
   - 로컬 파일 vs Hub 데이터셋 유연 지원
   - 전처리 함수의 map() 적용
   - 배치 처리로 효율성 향상

4. 평가 메트릭:
   - SQuAD v1.0: Exact Match, F1 Score
   - SQuAD v2.0: 추가로 HasAns/NoAns 분리 평가
   - 한국어 평가 시 정규화 필요

5. 학습 최적화:
   - Mixed Precision (fp16): 메모리 절약
   - Dynamic Padding: 실제 길이에 맞춰 패딩
   - 분산 학습 지원: 멀티 GPU 활용

하이퍼파라미터 가이드:

1. 학습률 (Learning Rate):
   - BERT-base: 2e-5 ~ 5e-5
   - ELECTRA: 3e-5 (일반적으로 높게)
   - RoBERTa: 1e-5 ~ 3e-5

2. 배치 크기 (Batch Size):
   - GPU 메모리에 따라 8, 16, 32
   - Gradient Accumulation으로 큰 배치 효과

3. 에포크 수:
   - 2-5 에포크 (과적합 주의)
   - Early Stopping 권장

4. Warmup:
   - 전체 스텝의 10% (warmup_ratio=0.1)
   - 안정적 학습 시작

성능 향상 전략:

1. 모델 선택:
   - KoELECTRA: 한국어 특화, 효율적
   - KoBERT: 안정적 성능
   - KoRoBERTa: 높은 성능

2. 데이터 증강:
   - 패러프레이징: 질문 다양화
   - 역번역: 데이터 확장
   - 하드 네거티브: 어려운 예제 추가

3. 앙상블:
   - 여러 모델 결합
   - 다양한 하이퍼파라미터 조합
   - Voting 또는 평균 기법

실무 배포 고려사항:
- 모델 크기 vs 정확성 트레이드오프
- 추론 속도 최적화 (양자화, 프루닝)
- 도메인 적응을 위한 추가 학습
- 지속적 모니터링 및 성능 평가
"""
