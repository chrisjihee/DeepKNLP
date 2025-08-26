# === Step 5 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 5 TODO 해답:

QAModel 클래스 구현:

1. 모델 로드:
   self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
   self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained)

2. 모델을 평가 모드로 설정:
   self.model.eval()

3. WebAPI 클래스를 Flask 앱에 등록:
   QAModel.WebAPI.register(route_base='/', app=server, init_argument=self)

4. 서버 실행:
   server.run(*args, **kwargs)

5. 입력 토크나이제이션:
   inputs = self.tokenizer.encode_plus(
       question, context, return_tensors="pt", truncation=True, padding=True
   )

6. 모델 추론:
   with torch.no_grad():
       outputs = self.model(**inputs)

7. start/end logits 추출:
   start_logits = outputs.start_logits
   end_logits = outputs.end_logits

8. 가장 높은 확률의 시작/끝 위치 찾기:
   start_index = torch.argmax(start_logits)
   end_index = torch.argmax(end_logits)

9. 예측된 답변 토큰들 추출:
   predict_answer_tokens = inputs["input_ids"][0, start_index:end_index + 1]

10. 토큰들을 텍스트로 디코딩:
    answer = self.tokenizer.decode(predict_answer_tokens)

11. Softmax 정규화된 확률 사용:
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    score = (torch.max(start_probs) * torch.max(end_probs)).item()

12. 원시 logit 점수 사용:
    score = float(torch.max(start_logits) + torch.max(end_logits))

WebAPI 클래스 구현:

13. HTML 템플릿 렌더링:
    return render_template(self.model.server_page)

14. JSON 요청 데이터 파싱:
    data = request.json
    question = data.get("question", "")
    context = data.get("context", "")

15. QA 모델로 답변 생성:
    result = self.model.infer_one(question, context)

16. JSON 형태로 결과 반환:
    return jsonify(result)

17. 개별 질문 처리:
    result = self.model.infer_one(question, context)

CLI 애플리케이션 구현:

18. CLI 옵션들 정의:
    pretrained: str = typer.Option(
        "monologg/koelectra-base-v3-finetuned-korquad",
        help="사전학습된 QA 모델 경로 또는 Hugging Face Hub ID"
    ),
    server_host: str = typer.Option("0.0.0.0"),
    server_port: int = typer.Option(9164),
    server_page: str = typer.Option("serve_qa.html"),
    normalized: bool = typer.Option(True),
    debug: bool = typer.Option(False),

19. 가장 최근 파일 선택:
    pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

20. QA 모델 로드:
    model = QAModel(pretrained=pretrained, server_page=server_page, normalized=normalized)

21. Flask 앱 생성:
    app = Flask(__name__, template_folder=Path("templates").resolve())

22. 웹 서비스 실행:
    model.run_server(app, host=server_host, port=server_port, debug=debug)

테스트 함수 구현:

23. 기본 모델 경로 설정:
    pretrained = "monologg/koelectra-base-v3-finetuned-korquad"

24. 질의응답 수행:
    result = model.infer_one(test_case["question"], test_case["context"])

핵심 개념:

1. 프로덕션 서비스 아키텍처:
   - Model Loading: 초기화 시 한 번만 로드
   - Stateless Design: 요청 간 상태 비공유
   - Error Handling: 안정적 서비스 운영
   - Health Check: 서비스 상태 모니터링

2. Flask-Classful 패턴:
   - 클래스 기반 뷰로 코드 구조화
   - 라우트 자동 등록으로 편의성 향상
   - MVC 패턴 적용으로 유지보수성 증대

3. REST API 설계:
   - POST /api: 단일 질의응답
   - POST /batch_api: 배치 처리
   - GET /health: 상태 확인
   - GET /: 웹 UI 제공

4. 성능 최적화:
   - torch.no_grad(): 메모리 절약
   - 모델 eval() 모드: 추론 최적화
   - 효율적 토크나이제이션: 캐싱 활용

5. 신뢰도 점수:
   - Normalized: Softmax 확률 기반 (0~1)
   - Raw: Logit 합계 기반 (실수값)
   - 사용자에게 답변 신뢰도 제공

API 사용법:

1. 단일 질의응답:
   ```bash
   curl -X POST http://localhost:9164/api \
     -H "Content-Type: application/json" \
     -d '{"question": "대한민국의 수도는?", "context": "대한민국의 수도는 서울이다."}'
   ```

2. 배치 처리:
   ```bash
   curl -X POST http://localhost:9164/batch_api \
     -H "Content-Type: application/json" \
     -d '{"questions": ["수도는?", "인구는?"], "context": "서울은 대한민국의 수도이다..."}'
   ```

3. 상태 확인:
   ```bash
   curl http://localhost:9164/health
   ```

실무 고려사항:

1. 확장성:
   - 로드 밸런서 적용
   - 다중 인스턴스 운영
   - 비동기 처리 도입

2. 보안:
   - 입력 검증 및 제한
   - Rate Limiting 적용
   - HTTPS 암호화

3. 모니터링:
   - 응답 시간 측정
   - 처리량 추적
   - 에러율 모니터링

4. 성능 튜닝:
   - 모델 양자화
   - 배치 처리 최적화
   - 캐싱 전략

배포 전략:

1. 컨테이너화:
   - Docker 이미지 구성
   - Kubernetes 오케스트레이션
   - 자동 스케일링 설정

2. CI/CD:
   - 자동 테스트 파이프라인
   - 점진적 배포 (Blue-Green)
   - 롤백 전략 수립

3. 인프라:
   - GPU 리소스 관리
   - 네트워크 최적화
   - 스토리지 전략

4. 운영:
   - 로그 집계 및 분석
   - 알림 시스템 구축
   - 백업 및 복구 계획
"""
