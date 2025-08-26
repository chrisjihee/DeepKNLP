# === Step 5 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 5 TODO 해답:

run_server:

1. WebAPI 클래스를 Flask 앱에 등록:
   NERModel.WebAPI.register(route_base="/", app=server, init_argument=self)

2. 서버 실행:
   server.run(*args, **kwargs)

WebAPI.index:

3. 템플릿 렌더링:
   return render_template(self.model.args.server.page)

WebAPI.api:

4. JSON 요청을 받아 NER 수행 후 결과 반환:
   response = self.model.infer_one(text=request.json)
   return jsonify(response)

train 명령어:

5. PyTorch 설정 초기화:
   torch.set_float32_matmul_precision("high")
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
   logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
   logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

6. 학습 인수 구성:
   args = TrainerArguments(
       env=ProjectEnv(
           project=project,
           job_name=job_name if job_name else pretrained.name,
           job_version=job_version,
           debugging=debugging,
           message_level=logging.DEBUG if debugging else logging.INFO,
           message_format=LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20,
       ),
       data=DataOption(
           home=data_home,
           name=data_name,
           files=DataFiles(train=train_file, valid=valid_file, test=test_file),
           num_check=num_check,
       ),
       model=ModelOption(
           pretrained=pretrained,
           finetuning=finetuning,
           name=model_name,
           seq_len=seq_len,
       ),
       hardware=HardwareOption(
           cpu_workers=cpu_workers,
           train_batch=train_batch,
           infer_batch=infer_batch,
           accelerator=accelerator,
           precision=precision,
           strategy=strategy,
           devices=device,
       ),
       printing=PrintingOption(
           print_rate_on_training=print_rate_on_training,
           print_rate_on_validate=print_rate_on_validate,
           print_rate_on_evaluate=print_rate_on_evaluate,
           print_step_on_training=print_step_on_training,
           print_step_on_validate=print_step_on_validate,
           print_step_on_evaluate=print_step_on_evaluate,
           tag_format_on_training=tag_format_on_training,
           tag_format_on_validate=tag_format_on_validate,
           tag_format_on_evaluate=tag_format_on_evaluate,
       ),
       learning=LearningOption(
           learning_rate=learning_rate,
           random_seed=random_seed,
           saving_mode=saving_mode,
           num_saving=num_saving,
           num_epochs=num_epochs,
           check_rate_on_training=check_rate_on_training,
           name_format_on_saving=name_format_on_saving,
       ),
   )

7. Fabric 초기화:
   fabric = Fabric(
       loggers=[args.prog.tb_logger, args.prog.csv_logger],
       devices=args.hardware.devices if args.hardware.accelerator in ["cuda", "gpu"] else args.hardware.cpu_workers if args.hardware.accelerator == "cpu" else "auto",
       strategy=args.hardware.strategy if args.hardware.accelerator in ["cuda", "gpu"] else "auto",
       precision=args.hardware.precision if args.hardware.accelerator in ["cuda", "gpu"] else None,
       accelerator=args.hardware.accelerator,
   )

핵심 개념:
- Flask-Classful: 클래스 기반 웹 API 구조화
- WebAPI 등록: Flask 앱에 라우트와 핸들러 자동 등록
- REST API: POST /api 엔드포인트로 NER 서비스 제공
- 웹 인터페이스: HTML 템플릿으로 사용자 친화적 UI 제공
- TrainerArguments: 모든 CLI 인수를 구조화된 설정 클래스로 매핑
- Lightning Fabric: 분산 학습, 혼합 정밀도, 로깅 등 통합 관리
- CLI 명령어: train, test, serve 세 가지 주요 기능
- NER 특화 설정: val_F1c 기준 저장, NER F1 메트릭 포함 로그 형식
- 환경 설정: TOKENIZERS_PARALLELISM, 로거 레벨 등 NER 최적화
"""
