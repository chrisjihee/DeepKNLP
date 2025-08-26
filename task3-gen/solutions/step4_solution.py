# === Step 4 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 4 TODO 해답:

1. GenerationTrainArguments 설정:
   args = GenerationTrainArguments(
       pretrained_model_name="skt/kogpt2-base-v2",
       downstream_model_dir="output/nsmc-gen/train_gen-by-kogpt2",
       downstream_corpus_name="nsmc",
       max_seq_length=32,
       batch_size=32 if torch.cuda.is_available() else 4,
       learning_rate=5e-5,
       epochs=3,
       tpu_cores=0 if torch.cuda.is_available() else 8,
       seed=7,
   )

2. 시드 설정:
   nlpbook.set_seed(args)

3. NSMC 데이터셋 다운로드:
   Korpora.fetch(
       corpus_name=args.downstream_corpus_name,
       root_dir=args.downstream_corpus_root_dir,
       force_download=args.force_download,
   )

4. 토크나이저 로드:
   tokenizer = PreTrainedTokenizerFast.from_pretrained(
       args.pretrained_model_name,
       eos_token="</s>",
   )

5. NSMC 코퍼스 객체 생성:
   corpus = NsmcCorpus()

6. 학습용 데이터셋 생성:
   train_dataset = GenerationDataset(
       args=args,
       corpus=corpus,
       tokenizer=tokenizer,
       mode="train",
   )

7. 학습용 데이터로더 생성:
   train_dataloader = DataLoader(
       train_dataset,
       batch_size=args.batch_size,
       sampler=RandomSampler(train_dataset, replacement=False),
       collate_fn=nlpbook.data_collator,
       drop_last=False,
       num_workers=args.cpu_workers,
   )

8. 검증용 데이터셋 생성:
   val_dataset = GenerationDataset(
       args=args,
       corpus=corpus,
       tokenizer=tokenizer,
       mode="test",
   )

9. 검증용 데이터로더 생성:
   val_dataloader = DataLoader(
       val_dataset,
       batch_size=args.batch_size,
       sampler=SequentialSampler(val_dataset),
       collate_fn=nlpbook.data_collator,
       drop_last=False,
       num_workers=args.cpu_workers,
   )

10. GPT2 모델 로드:
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name
    )

11. 생성 태스크 객체 생성:
    task = GenerationTask(model, args)

12. Trainer 객체 가져오기:
    trainer = nlpbook.get_trainer(args)

13. 학습 실행:
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

핵심 개념 설명:

1. Fine-tuning vs Pre-training:
   - Pre-training: 대량의 일반 텍스트로 언어 모델의 기본 능력 학습
   - Fine-tuning: 특정 도메인/태스크에 맞게 모델을 추가 학습
   - Transfer Learning의 핵심: 일반 지식 → 특화 지식

2. NSMC 데이터셋:
   - 네이버 영화 리뷰 감정 분류 데이터
   - 긍정/부정 리뷰 텍스트가 포함
   - 자연스러운 한국어 구어체 표현 풍부
   - 감정이 담긴 생동감 있는 텍스트 생성 학습

3. GenerationTrainArguments:
   - 모든 학습 관련 설정을 통합 관리
   - 재현 가능한 실험을 위한 필수 요소
   - 하드웨어 환경에 따른 자동 최적화

4. 데이터 파이프라인:
   - NsmcCorpus: 원본 NSMC 데이터 로드 및 전처리
   - GenerationDataset: 토큰화 및 모델 입력 형태 변환
   - DataLoader: 배치 단위 효율적 데이터 제공

5. GenerationTask:
   - PyTorch Lightning 기반 학습 태스크
   - Forward pass, Loss 계산, Optimization 자동화
   - 분산 학습, 체크포인트 저장 등 고급 기능

6. 학습 파라미터 의미:
   - max_seq_length=32: 짧은 문장 생성에 최적화
   - batch_size: GPU 메모리에 따라 조정
   - learning_rate=5e-5: GPT2 fine-tuning 표준 학습률
   - epochs=3: 과적합 방지를 위한 적절한 에포크 수

Fine-tuning 후 기대 효과:
- NSMC 리뷰 스타일의 감정 표현이 풍부한 텍스트 생성
- 영화, 드라마 등 엔터테인먼트 관련 표현 향상
- 한국어 구어체, 감탄사, 줄임말 등 자연스러운 표현
- 긍정/부정 감정이 뚜렷한 표현력 있는 문장 생성

학습 모니터링:
- Training Loss: 학습 진행 상황 확인
- Validation Loss: 과적합 여부 판단
- Learning Rate Scheduler: 학습률 자동 조정
- 체크포인트: 최적 모델 자동 저장

실무 고려사항:
- GPU 메모리 부족 시 batch_size 감소
- 학습 시간은 하드웨어에 따라 차이 (GPU 권장)
- 과적합 방지를 위한 Early Stopping 고려
- 다양한 도메인 데이터로 추가 fine-tuning 가능
"""
