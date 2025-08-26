### 목적과 전체 구조
- **역할**: 문장 생성(generation) 태스크를 위한 학습/추론/서빙 파이프라인.
- **프레임워크**: Hugging Face Transformers + ratsnlp(`nlpbook.generation`) + Korpora(NSMC) 활용.
- **두 축**: 
  - 학습(`train_gen-*.py`) → 체크포인트 산출
  - 추론(`infer_gen-*.py`) → 다양한 generation 설정 실습
  - 서빙(`serve_gen-*.py`) → 웹 UI로 인터랙티브 생성

### 버전별 차이
- **-1**: `skt/kogpt2-base-v2`, output: `output/nsmc-gen/train_gen-by-kogpt2`, 서버 포트 9001
- **-2**: `skt/ko-gpt-trinity-1.2B-v0.5`, output: `.../train_gen-by-kogpt-trinity`, 서버 포트 9002
- **-3**: `EleutherAI/polyglot-ko-1.3b`, output: `.../train_gen-by-polyglot-ko-1.3b`, 서버 포트 9003

### 학습 스크립트: `train_gen-1/2/3.py`
- **인자 세팅**: `GenerationTrainArguments`
  - `pretrained_model_name`, `downstream_model_dir`, `max_seq_length`, `batch_size`, `learning_rate`, `epochs`, `seed` 등.
- **데이터**: `Korpora.fetch("nsmc")` → `NsmcCorpus` → `GenerationDataset(mode="train"/"test")`
- **토크나이저/모델**: `PreTrainedTokenizerFast.from_pretrained(..., eos_token="</s>")`, `GPT2LMHeadModel.from_pretrained(...)`
- **DataLoader**: `RandomSampler(train)`, `SequentialSampler(val)`, `nlpbook.data_collator`
- **학습**: `GenerationTask(model, args)` → `nlpbook.get_trainer(args)` → `trainer.fit(...)`

핵심: NSMC 문장을 조건 없이/약하게 조건화하여 다음 토큰을 생성하는 LM 파인튜닝 구조.

### 추론 스크립트: `infer_gen-1/2/3.py`
- **로딩**: 지정 `pretrained`로 모델/토크나이저 로드.
- **입력**: `input_sentence`를 인코딩해 `model.generate(...)` 호출.
- **하이퍼 실습**: 16개 블록으로 다음을 비교 출력
  - Greedy vs Beam(`num_beams`)
  - `no_repeat_ngram_size`
  - `repetition_penalty`
  - Sampling: `top_k`, `top_p`(nucleus)
  - `temperature`
  - 복합 설정(top-k/p + temperature + repetition/no-repeat)
- 목적: 생성 품질과 다양성에 미치는 하이퍼파라미터 효과를 체감.

### 서빙 스크립트: `serve_gen-1/2/3.py`
- **체크포인트 로드**:
  - `GenerationDeployArguments`에서 `downstream_model_checkpoint_fpath`를 받아
  - `GPT2Config.from_pretrained(...)` → `GPT2LMHeadModel(config)` 생성 후
  - fine-tuned `state_dict`를 키 보정(`"model."` 제거)하여 `load_state_dict(...)`
- **토크나이저**: 동일 `PreTrainedTokenizerFast(..., eos_token="</s>")`
- **추론 함수**: `inference_fn(prompt, min_length, max_length, top_p, top_k, repetition_penalty, no_repeat_ngram_size, temperature)`
  - `do_sample=True`로 `generate` 수행 후 디코딩
  - 예외 시 한국어 가이드 메시지 반환(파라미터 유효 범위)
- **웹 앱**: `get_web_service_app(inference_fn, template_folder=templates, server_page="serve_gen.html")` → `app.run(host="0.0.0.0", port=900{1|2|3})`

### 실행 예시
```bash
# 학습 (버전 1: KoGPT2 base)
python task3-gen/train_gen-1.py

# 추론 파라미터 실습 (버전 2: KoGPT Trinity)
python task3-gen/infer_gen-2.py

# 서빙 (버전 3: Polyglot-ko 1.3b)
python task3-gen/serve_gen-3.py
```

### 실전 팁
- **길이 제어**: `min_length`, `max_length`로 문장 길이 하한/상한 설정.
- **반복 억제**: `repetition_penalty`↑, `no_repeat_ngram_size`≥3 조합이 유효.
- **다양성 제어**: `do_sample=True` + `top_k/top_p` + `temperature`로 샘플 다양성 조절.
- **성능/자원**: -2, -3 모델은 파라미터가 커서 GPU 메모리 요구가 큼. 배치 사이즈/seq 길이 조정 권장.

- 요약: `task3-gen`은 3개 사전학습 언어모델에 대해 동일한 학습 파이프라인과, 다양한 생성 하이퍼 실습용 추론/웹서빙 환경을 제공합니다. 학습은 ratsnlp의 GenerationTask/Trainer를 통해 간결하게 구성되고, 서빙은 체크포인트를 로드한 후 `generate` 파라미터로 품질/다양성을 조절합니다.