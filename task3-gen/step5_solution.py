# === Step 5 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 5 TODO 해답:

1. 사용 가능한 디바이스 설정:
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

2. 배포용 인수 설정:
   args = GenerationDeployArguments(
       pretrained_model_name="skt/kogpt2-base-v2",
       downstream_model_dir="output/nsmc-gen/train_gen-by-kogpt2",
   )

3. 사전학습 모델 설정 로드:
   pretrained_model_config = GPT2Config.from_pretrained(
       args.pretrained_model_name,
   )

4. 모델 객체 생성:
   model = GPT2LMHeadModel(pretrained_model_config)

5. Fine-tuned 체크포인트 로드:
   fine_tuned_model_ckpt = torch.load(
       args.downstream_model_checkpoint_fpath,
       map_location=device,
   )

6. 모델에 Fine-tuned 가중치 로드:
   model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

7. 모델을 평가 모드로 설정:
   model.eval()

8. 토크나이저 로드:
   tokenizer = PreTrainedTokenizerFast.from_pretrained(
       args.pretrained_model_name,
       eos_token="</s>",
   )

9. 입력 프롬프트 토큰화:
   input_ids = tokenizer.encode(prompt, return_tensors="pt")

10. 모델을 사용하여 텍스트 생성:
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        top_p=float(top_p),
        top_k=int(top_k),
        min_length=int(min_length),
        max_length=int(max_length),
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        temperature=float(temperature),
    )

11. 생성된 토큰 ID들을 텍스트로 디코딩:
    generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])

12. 웹 서비스 애플리케이션 생성:
    app = get_web_service_app(
        inference_fn, 
        template_folder=Path("templates").resolve(), 
        server_page="serve_gen.html"
    )

13. 웹 서버 실행:
    app.run(host="0.0.0.0", port=9001)

핵심 개념 설명:

1. 모델 배포 (Model Deployment):
   - 학습된 모델을 실제 서비스 환경에 배치
   - 체크포인트 로드, 모델 초기화, 추론 최적화
   - 실시간 요청 처리를 위한 효율적 아키텍처

2. 체크포인트 관리:
   - state_dict: 모델의 가중치 정보
   - map_location: GPU/CPU 간 모델 이동
   - 키 이름 변환: Lightning 래퍼에서 순수 모델로

3. 웹 서비스 아키텍처:
   - Frontend: HTML/CSS/JavaScript 기반 사용자 인터페이스
   - Backend: Flask 기반 REST API 서버
   - Model: 텍스트 생성 추론 엔진

4. 실시간 추론 최적화:
   - torch.no_grad(): 기울기 계산 비활성화로 메모리 절약
   - model.eval(): 드롭아웃, 배치 정규화 등 평가 모드
   - 단일 요청 처리에 최적화된 배치 크기

5. 오류 처리 (Error Handling):
   - try-except 구문으로 안정적 서비스 제공
   - 사용자 친화적 오류 메시지
   - 파라미터 범위 검증 및 가이드

6. 웹 서비스 구성 요소:

   a) 추론 함수 (inference_fn):
      - 핵심 비즈니스 로직
      - 모델 입력 전처리
      - 생성 파라미터 적용
      - 결과 후처리 및 반환

   b) Flask 애플리케이션:
      - HTTP 요청/응답 처리
      - 템플릿 렌더링
      - API 엔드포인트 제공

   c) 사용자 인터페이스:
      - 프롬프트 입력 영역
      - 파라미터 조정 슬라이더
      - 실시간 결과 표시

7. 파라미터 타입 변환:
   - 웹에서 전달되는 문자열을 적절한 타입으로 변환
   - float(), int() 함수 활용
   - 타입 오류 방지를 위한 명시적 변환

8. 서비스 접근 방법:
   - 웹 브라우저에서 http://localhost:9001 접속
   - 프롬프트 입력 후 파라미터 조정
   - 실시간 텍스트 생성 결과 확인

9. 실무 응용 분야:
   - 창작 도구: 소설, 시, 에세이 작성 도움
   - 챗봇 백엔드: 대화형 AI 시스템
   - 콘텐츠 생성: 마케팅 문구, 제품 설명
   - 교육 도구: 언어 학습, 창의적 글쓰기

10. 성능 최적화 고려사항:
    - GPU 메모리 관리 (모델 크기에 따른 제약)
    - 동시 접속자 수 처리 능력
    - 응답 시간 최적화 (생성 길이 제한)
    - 서버 리소스 모니터링 및 알림

11. 확장 가능성:
    - 여러 모델 동시 서비스
    - 사용자별 개인화 설정
    - API 키 기반 접근 제어
    - 로그 기반 사용 패턴 분석

보안 고려사항:
- 입력 텍스트 필터링 (부적절한 내용 차단)
- 생성 길이 제한 (DoS 공격 방지)
- 요청 빈도 제한 (Rate Limiting)
- HTTPS 적용 (프로덕션 환경)
"""
