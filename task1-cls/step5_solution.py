# === Step 5 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 5 TODO 해답:

1. WebAPI 클래스를 Flask 앱에 등록:
   NSMCModel.WebAPI.register(route_base="/", app=server, init_argument=self)

2. 서버 실행:
   server.run(*args, **kwargs)

3. 메인 페이지 템플릿 렌더링:
   return render_template(self.model.args.server.page)

4. 감성분석 API 엔드포인트:
   response = self.model.infer_one(text=request.json)
   return jsonify(response)

5. PyTorch 설정 초기화:
   torch.set_float32_matmul_precision("high")
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)

6. 학습 인수 구성 (TrainerArguments):
   args = TrainerArguments(
       env=EnvOption(
           project=project,
           job_name=job_name,
           job_version=job_version,
           debugging=debugging,
           logging_file=logging_file,
           argument_file=argument_file,
           message_level=logging.DEBUG if debugging else logging.INFO,
           message_format=LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20,
       ),
       data=DataOption(
           home=data_home,
           name=data_name,
           files=DataFiles(
               train=train_file,
               valid=valid_file,
               test=test_file,
           ),
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
- Flask-Classful: 클래스 기반 라우팅, @route 데코레이터
- Typer: 타입 힌트 기반 CLI 프레임워크
- TrainerArguments: 구조화된 설정 관리
- Lightning Fabric: 분산 학습 및 혼합 정밀도 지원
- JobTimer: 실행 시간 측정 및 로깅 컨텍스트
- 환경별 설정 분기: GPU/CPU에 따른 다른 Fabric 설정
"""
