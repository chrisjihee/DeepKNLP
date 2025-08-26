# === Step 5: QA 웹 서비스 ===
# 수강생 과제: TODO 부분을 완성하여 Flask 기반 실시간 질의응답 웹 서비스를 구현하세요.

import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import typer
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)

class QAModel(LightningModule):
    """질의응답 모델 클래스"""
    
    def __init__(self, pretrained: str, server_page: str, normalized: bool = True):
        """
        QA 모델 초기화
        
        Args:
            pretrained: 사전학습된 QA 모델 경로 또는 Hugging Face Hub ID
            server_page: 웹 템플릿 파일명
            normalized: 점수 계산 시 softmax 정규화 사용 여부
        """
        super().__init__()
        self.server_page = server_page
        self.normalized = normalized

        # TODO: 모델 로드
        logger.info(f"Loading model from {pretrained}")
        # 힌트: AutoTokenizer.from_pretrained(), AutoModelForQuestionAnswering.from_pretrained()
        self.tokenizer = # TODO: 완성하세요
        self.model = # TODO: 완성하세요
        
        # TODO: 모델을 평가 모드로 설정
        # TODO: 완성하세요

    def run_server(self, server: Flask, *args, **kwargs):
        """Flask 웹 서버 실행"""
        # TODO: WebAPI 클래스를 Flask 앱에 등록
        # 힌트: QAModel.WebAPI.register() 사용
        # TODO: 완성하세요
        
        # TODO: 서버 실행
        # 힌트: server.run() 사용
        # TODO: 완성하세요

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        단일 질문-컨텍스트 쌍에 대한 답변 생성
        
        Args:
            question: 질문 문자열
            context: 컨텍스트 문자열
            
        Returns:
            dict: 답변과 관련 정보를 포함한 딕셔너리
        """
        # TODO: 입력 유효성 검사
        if not question.strip():
            return {"question": question, "context": context, "answer": "(질문이 비어있습니다.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(컨텍스트가 비어있습니다.)"}

        # TODO: 입력 토크나이제이션
        # 힌트: self.tokenizer.encode_plus() 사용, return_tensors="pt", truncation=True, padding=True
        inputs = # TODO: 완성하세요
        
        # TODO: 모델 추론 (gradient 계산 없이)
        with torch.no_grad():
            outputs = # TODO: 완성하세요

        # TODO: start/end logits 추출
        start_logits = # TODO: 완성하세요
        end_logits = # TODO: 완성하세요

        # TODO: 가장 높은 확률의 시작/끝 위치 찾기
        start_index = # TODO: torch.argmax() 사용
        end_index = # TODO: torch.argmax() 사용

        # TODO: 예측된 답변 토큰들 추출
        predict_answer_tokens = # TODO: 완성하세요
        
        # TODO: 토큰들을 텍스트로 디코딩
        answer = # TODO: 완성하세요

        # TODO: 신뢰도 점수 계산
        if self.normalized:
            # Softmax 정규화된 확률 사용
            start_probs = # TODO: F.softmax() 사용
            end_probs = # TODO: F.softmax() 사용
            score = # TODO: 최대 확률들의 곱
        else:
            # 원시 logit 점수 사용
            score = # TODO: 최대 logit들의 합

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "score": round(score, 4),
            "start": int(start_index),
            "end": int(end_index)
        }

    class WebAPI(FlaskView):
        """Flask 기반 웹 API 클래스"""
        
        def __init__(self, model: "QAModel"):
            """
            WebAPI 초기화
            
            Args:
                model: QAModel 인스턴스
            """
            self.model = model

        @route('/')
        def index(self):
            """메인 페이지 렌더링"""
            # TODO: HTML 템플릿 렌더링
            # 힌트: render_template() 사용, self.model.server_page
            return # TODO: 완성하세요

        @route('/api', methods=['POST'])
        def api(self):
            """
            QA API 엔드포인트
            
            POST /api
            Request Body: {"question": "질문", "context": "컨텍스트"}
            
            Returns:
                JSON: 답변과 관련 정보
            """
            # TODO: JSON 요청 데이터 파싱
            data = # TODO: request.json
            question = # TODO: data에서 "question" 추출, 기본값 ""
            context = # TODO: data에서 "context" 추출, 기본값 ""
            
            # TODO: QA 모델로 답변 생성
            result = # TODO: self.model.infer_one() 호출
            
            # TODO: JSON 형태로 결과 반환
            return # TODO: jsonify() 사용

        @route('/batch_api', methods=['POST'])
        def batch_api(self):
            """
            배치 QA API 엔드포인트 (여러 질문 동시 처리)
            
            POST /batch_api  
            Request Body: {"questions": ["질문1", "질문2"], "context": "컨텍스트"}
            
            Returns:
                JSON: 각 질문별 답변 리스트
            """
            # TODO: 배치 요청 데이터 파싱
            data = request.json
            questions = data.get("questions", [])
            context = data.get("context", "")
            
            # TODO: 각 질문에 대해 답변 생성
            results = []
            for question in questions:
                # TODO: 개별 질문 처리
                result = # TODO: self.model.infer_one() 호출
                results.append(result)
            
            return jsonify({"results": results})

        @route('/health')
        def health(self):
            """서비스 상태 확인 엔드포인트"""
            # TODO: 서비스 상태 정보 반환
            return jsonify({
                "status": "healthy",
                "model_loaded": self.model.model is not None,
                "tokenizer_loaded": self.model.tokenizer is not None
            })


# === CLI 애플리케이션 ===

main = typer.Typer()

@main.command()
def serve(
    # TODO: CLI 옵션들 정의
    pretrained: str = typer.Option(
        # TODO: 기본값 설정 (로컬 체크포인트 또는 Hub ID)
        help="사전학습된 QA 모델 경로 또는 Hugging Face Hub ID"
    ),
    server_host: str = typer.Option(
        # TODO: 기본값 "0.0.0.0",
        help="서버 호스트 주소"
    ),
    server_port: int = typer.Option(
        # TODO: 기본값 9164,
        help="서버 포트 번호"
    ),
    server_page: str = typer.Option(
        # TODO: 기본값 "serve_qa.html",
        help="웹 템플릿 파일명"
    ),
    normalized: bool = typer.Option(
        # TODO: 기본값 True,
        help="점수 계산 시 softmax 정규화 사용 여부"
    ),
    debug: bool = typer.Option(
        # TODO: 기본값 False,
        help="Flask 디버그 모드 활성화"
    ),
):
    """
    QA 웹 서비스 실행
    
    기능:
    - 실시간 질의응답 웹 인터페이스 제공
    - REST API 엔드포인트 제공
    - 배치 처리 지원
    - 서비스 상태 모니터링
    """
    # TODO: 로깅 설정
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # TODO: 로컬 체크포인트 경로 처리 (glob 패턴 지원)
    from pathlib import Path
    import glob
    
    if "*" in pretrained:
        # Glob 패턴으로 체크포인트 찾기
        checkpoint_paths = glob.glob(pretrained)
        if checkpoint_paths:
            # 가장 최근 파일 선택
            pretrained = # TODO: 완성하세요 (파일 수정 시간 기준 정렬)
        else:
            raise ValueError(f"No checkpoint found matching pattern: {pretrained}")

    print(f"Starting QA service with model: {pretrained}")
    
    # TODO: QA 모델 로드
    model = # TODO: QAModel 인스턴스 생성
    
    # TODO: Flask 앱 생성
    # 힌트: Flask(__name__, template_folder=Path("templates").resolve())
    app = # TODO: 완성하세요

    # TODO: 웹 서비스 실행
    # 힌트: model.run_server() 호출
    # TODO: 완성하세요


@main.command()
def test():
    """
    QA 모델 테스트 (간단한 예제로 동작 확인)
    """
    # TODO: 테스트용 모델 로드
    pretrained = # TODO: 기본 모델 경로 설정
    model = QAModel(pretrained=pretrained, server_page="", normalized=True)
    
    # TODO: 테스트 데이터
    test_cases = [
        {
            "question": "대한민국의 수도는?",
            "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다."
        },
        {
            "question": "대한민국의 국화는?", 
            "context": "대한민국의 국기는 태극기이며, 국가는 애국가, 국화는 무궁화이다."
        }
    ]
    
    print("=== QA 모델 테스트 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        
        # TODO: 질의응답 수행
        result = # TODO: model.infer_one() 호출
        
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']}")
        print(f"Position: [{result['start']}:{result['end']}]")


if __name__ == "__main__":
    main()

"""
학습 목표:
1. 학습된 QA 모델을 실제 웹 서비스로 배포하는 과정 이해
2. Flask와 Flask-Classful을 활용한 REST API 구현
3. 실시간 질의응답 시스템의 아키텍처 설계
4. 사용자 친화적 웹 인터페이스 구축

핵심 개념:

1. 모델 서빙 아키텍처:
   - Model Loading: 학습된 체크포인트 로드
   - Inference Engine: 실시간 질의응답 처리
   - Web Framework: Flask 기반 HTTP 서버
   - API Design: RESTful 엔드포인트 설계

2. Flask-Classful 패턴:
   - 클래스 기반 뷰 구성
   - 라우트 자동 등록
   - 코드 구조화 및 재사용성 향상
   - MVC 패턴 적용

3. QA 서비스 특화 기능:
   - 단일 질문 처리: /api 엔드포인트
   - 배치 처리: /batch_api 엔드포인트  
   - 상태 모니터링: /health 엔드포인트
   - 웹 UI: 사용자 친화적 인터페이스

4. 성능 최적화:
   - torch.no_grad(): 추론 시 gradient 계산 비활성화
   - 모델 eval() 모드: 드롭아웃 등 비활성화
   - 효율적 토크나이제이션: 캐싱 및 배치 처리

5. 신뢰도 점수:
   - Normalized Score: Softmax 확률 기반
   - Raw Score: 원시 logit 값 기반
   - 사용자에게 답변 신뢰도 제공

API 엔드포인트:

1. GET /:
   - 메인 웹 페이지
   - 질문/컨텍스트 입력 폼
   - 실시간 답변 표시

2. POST /api:
   - 단일 질의응답 처리
   - Input: {"question": "...", "context": "..."}
   - Output: {"answer": "...", "score": 0.95, ...}

3. POST /batch_api:
   - 여러 질문 동시 처리
   - Input: {"questions": ["...", "..."], "context": "..."}
   - Output: {"results": [{...}, {...}]}

4. GET /health:
   - 서비스 상태 확인
   - 모델 로드 상태 점검
   - 모니터링 시스템 연동

실무 배포 고려사항:

1. 확장성:
   - 로드 밸런서 뒤에서 다중 인스턴스 실행
   - GPU/CPU 리소스 효율적 활용
   - 요청 큐잉 및 비동기 처리

2. 안정성:
   - 에러 핸들링 및 예외 상황 대응
   - 요청 검증 및 입력 제한
   - 타임아웃 설정 및 리소스 보호

3. 모니터링:
   - 응답 시간 측정
   - 처리량 및 성공률 추적
   - 모델 성능 메트릭 수집

4. 보안:
   - 입력 검증 및 SQL 인젝션 방지
   - 속도 제한 (Rate Limiting)
   - HTTPS 적용 및 인증/인가

5. 성능 튜닝:
   - 모델 양자화 (Quantization)
   - 동적 배치 처리
   - 캐싱 전략 적용

사용 시나리오:
- 고객 지원 시스템: FAQ 자동 응답
- 교육 플랫폼: 학습 자료 기반 Q&A
- 문서 검색: 사내 문서에서 정보 추출
- 챗봇 백엔드: 대화형 AI 시스템
"""
