# === Step 5 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 5 TODO 해답:

GenerativeQAModel 클래스 구현:

1. T5 모델 로드:
   self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
   self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)

2. 모델을 평가 모드로 설정:
   self.model.eval()

3. WebAPI 클래스를 Flask 앱에 등록:
   GenerativeQAModel.WebAPI.register(route_base='/', app=server, init_argument=self)

4. 서버 실행:
   server.run(*args, **kwargs)

5. T5 입력 형식 구성:
   input_text = f"question: {question} context: {context}"

6. 입력 토크나이제이션:
   inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

7. 점수와 함께 생성:
   outputs = self.model.generate(
       input_ids=inputs["input_ids"],
       attention_mask=inputs["attention_mask"],
       max_length=self.max_length,
       num_beams=self.num_beams,
       return_dict_in_generate=True,
       output_scores=True
   )

8. 생성된 답변 디코딩:
   answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

9. 소프트맥스로 확률 변환:
   token_prob = F.softmax(outputs.scores[i-1], dim=-1)[0, token_id].item()

10. 전체 점수 계산:
    if token_probs:
        score = torch.prod(torch.tensor(token_probs)).item()

11. 창의적 생성을 위한 파라미터 설정:
    creative_num_beams = creative_params.get("num_beams", 3)
    creative_max_length = creative_params.get("max_length", 100)
    do_sample = creative_params.get("do_sample", True)
    temperature = creative_params.get("temperature", 0.8)
    top_p = creative_params.get("top_p", 0.9)

12. T5 입력 형식 구성:
    input_text = f"question: {question} context: {context}"
    inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

13. 창의적 파라미터로 생성:
    outputs = self.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=creative_max_length,
        num_beams=creative_num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )

14. 답변 디코딩:
    answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

WebAPI 클래스 구현:

15. HTML 템플릿 렌더링:
    return render_template(self.model.server_page)

16. JSON 요청 데이터 파싱:
    data = request.json
    question = data.get("question", "")
    context = data.get("context", "")

17. 생성형 QA 모델로 답변 생성:
    result = self.model.infer_one(question, context)

18. JSON 형태로 결과 반환:
    return jsonify(result)

19. 창의적 답변 생성:
    result = self.model.infer_creative(question, context, creative_params)

20. 표준 답변 생성:
    standard_result = self.model.infer_one(question, context)

21. 창의적 답변 생성:
    creative_result = self.model.infer_creative(question, context)

CLI 애플리케이션 구현:

22. CLI 옵션들 정의:
    pretrained: str = typer.Option(
        "paust/pko-t5-base-finetuned-korquad",
        help="사전학습된 T5 QA 모델 경로 또는 Hugging Face Hub ID"
    ),
    server_host: str = typer.Option("0.0.0.0"),
    server_port: int = typer.Option(9165),
    server_page: str = typer.Option("serve_qa_seq2seq.html"),
    num_beams: int = typer.Option(5),
    max_length: int = typer.Option(50),
    debug: bool = typer.Option(False),

23. 가장 최근 파일 선택:
    pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

24. 생성형 QA 모델 로드:
    model = GenerativeQAModel(pretrained=pretrained, server_page=server_page, num_beams=num_beams, max_length=max_length)

25. Flask 앱 생성:
    app = Flask(__name__, template_folder=Path("templates").resolve())

26. 웹 서비스 실행:
    model.run_server(app, host=server_host, port=server_port, debug=debug)

테스트 함수 구현:

27. 기본 모델 경로 설정:
    pretrained = "paust/pko-t5-base-finetuned-korquad"

28. 표준 생성:
    standard_result = model.infer_one(test_case["question"], test_case["context"])

29. 창의적 생성:
    creative_result = model.infer_creative(test_case["question"], test_case["context"])

핵심 개념:

1. Generative QA 서비스 아키텍처:
   - Model Loading: T5 모델 초기화
   - Generation Engine: 텍스트 생성 처리
   - Parameter Control: 생성 파라미터 조절
   - Creative Mode: 창의적 생성 지원

2. API 설계 특징:

   **표준 API** (POST /api):
   - 기본 생성 파라미터 사용
   - 안정적이고 일관된 답변
   - 빠른 응답 시간

   **창의적 API** (POST /api/creative):
   - 사용자 정의 생성 파라미터
   - 창의적이고 다양한 답변
   - 느린 응답 시간

   **비교 API** (POST /api/compare):
   - 두 가지 방식 동시 비교
   - 차이점 분석 제공
   - 교육/연구 목적

3. 창의적 생성 파라미터:
   - do_sample=True: 확률적 생성 활성화
   - temperature=0.8: 창의성 증가 (0.7-1.2)
   - top_p=0.9: Nucleus sampling (0.8-0.95)
   - num_beams=3: 적당한 품질과 다양성

4. 점수 계산 시스템:
   - 토큰별 확률 계산
   - 전체 시퀀스 점수 = 토큰 확률들의 곱
   - 길이 정규화 고려 필요
   - 신뢰도 지표로 활용

5. 서비스 모니터링:
   - /health 엔드포인트: 상태 확인
   - 모델 로드 상태 체크
   - 생성 파라미터 정보 제공
   - 실시간 성능 모니터링

6. Extractive vs Generative 서비스 비교:

   **Extractive QA 서비스**:
   - 단순한 입력-출력 구조
   - 고정된 추론 방식
   - 빠른 응답 (50-100ms)
   - 제한된 답변 형태
   - 높은 정확성

   **Generative QA 서비스**:
   - 유연한 생성 파라미터
   - 다양한 생성 모드
   - 느린 응답 (500-2000ms)
   - 다양한 답변 형태
   - 창의적 답변 가능

7. 실시간 최적화:
   - 모델 양자화 적용
   - 배치 처리 활용
   - 캐싱 전략 구현
   - GPU 메모리 관리

8. 품질 보장 시스템:
   - 생성 결과 필터링
   - 독성 콘텐츠 탐지
   - 사실성 검증 모듈
   - 사용자 피드백 수집

실무 배포 전략:

1. 성능 최적화:
   - 모델 서빙 최적화 (TensorRT, ONNX)
   - 동적 배치 처리
   - 요청 큐잉 시스템
   - 로드 밸런싱

2. 확장성 설계:
   - 마이크로서비스 아키텍처
   - 컨테이너 기반 배포
   - 오토 스케일링 설정
   - 다중 GPU 활용

3. 품질 관리:
   - A/B 테스트 프레임워크
   - 실시간 품질 모니터링
   - 사용자 만족도 추적
   - 지속적 모델 개선

4. 보안 고려사항:
   - 입력 검증 및 제한
   - Rate Limiting 적용
   - 콘텐츠 필터링
   - 사용자 인증/인가

사용 시나리오:

1. 교육 플랫폼:
   - "개념을 설명해주세요"
   - "예시를 들어주세요"
   - "차이점을 알려주세요"

2. 고객 지원:
   - "문제를 해결하는 방법은?"
   - "단계별로 안내해주세요"
   - "대안을 제시해주세요"

3. 연구 도구:
   - "논문을 요약해주세요"
   - "핵심 내용은 무엇인가요?"
   - "의미를 해석해주세요"

4. 창작 지원:
   - "아이디어를 제안해주세요"
   - "스토리를 만들어주세요"
   - "다양한 관점을 제시해주세요"

성능 벤치마크:
- 응답 시간: 평균 1-2초
- 동시 사용자: 100-500명
- 메모리 사용량: 4-8GB
- GPU 활용률: 70-90%

미래 발전 방향:
- 다국어 지원 확장
- 실시간 학습 기능
- 개인화 답변 생성
- 멀티모달 입력 지원
"""
