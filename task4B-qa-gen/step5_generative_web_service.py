# === Step 5: Generative QA 웹 서비스 ===
# 수강생 과제: TODO 부분을 완성하여 Flask 기반 생성형 QA 웹 서비스를 구현하세요.
# Extractive QA와의 차이점: 텍스트 생성 기반 응답과 창의적 답변 지원!

import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import typer
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

class GenerativeQAModel:
    """생성형 질의응답 모델 클래스"""
    
    def __init__(self, pretrained: str, server_page: str, num_beams: int = 5, max_length: int = 50):
        """
        Generative QA 모델 초기화
        
        Args:
            pretrained: 사전학습된 T5 QA 모델 경로 또는 Hugging Face Hub ID
            server_page: 웹 템플릿 파일명
            num_beams: Beam Search 폭
            max_length: 최대 생성 길이
        """
        self.server_page = server_page
        self.num_beams = num_beams
        self.max_length = max_length

        # TODO: T5 모델 로드
        logger.info(f"Loading T5 model from {pretrained}")
        # 힌트: AutoTokenizer, AutoModelForSeq2SeqLM 사용
        self.tokenizer = # TODO: 완성하세요
        self.model = # TODO: 완성하세요
        
        # TODO: 모델을 평가 모드로 설정
        # TODO: 완성하세요

    def run_server(self, server: Flask, *args, **kwargs):
        """Flask 웹 서버 실행"""
        # TODO: WebAPI 클래스를 Flask 앱에 등록
        # 힌트: GenerativeQAModel.WebAPI.register() 사용
        # TODO: 완성하세요
        
        # TODO: 서버 실행
        # TODO: 완성하세요

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        단일 질문-컨텍스트 쌍에 대한 생성형 답변
        
        Args:
            question: 질문 문자열
            context: 컨텍스트 문자열
            
        Returns:
            dict: 생성된 답변과 관련 정보를 포함한 딕셔너리
        """
        # TODO: 입력 유효성 검사
        if not question.strip():
            return {"question": question, "context": context, "answer": "(질문이 비어있습니다.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(컨텍스트가 비어있습니다.)"}

        # TODO: T5 입력 형식 구성
        # 힌트: "question: {질문} context: {컨텍스트}" 형태
        input_text = # TODO: 완성하세요
        
        # TODO: 입력 토크나이제이션
        # 힌트: self.tokenizer() 사용, return_tensors="pt", truncation=True, padding=True
        inputs = # TODO: 완성하세요
        
        # TODO: 텍스트 생성 (gradient 계산 없이)
        with torch.no_grad():
            # TODO: 점수와 함께 생성
            # 힌트: self.model.generate() 사용, return_dict_in_generate=True, output_scores=True
            outputs = self.model.generate(
                # TODO: 필요한 인수들을 완성하세요
                # input_ids, attention_mask, max_length, num_beams,
                # return_dict_in_generate, output_scores
            )

        # TODO: 생성된 답변 디코딩
        # 힌트: self.tokenizer.decode() 사용, skip_special_tokens=True
        answer = # TODO: 완성하세요

        # TODO: 신뢰도 점수 계산 (토큰 확률 기반)
        if hasattr(outputs, 'scores') and outputs.scores:
            token_probs = []
            for i, token_id in enumerate(outputs.sequences[0]):
                if i == 0:  # 시작 토큰 제외
                    continue
                if i-1 < len(outputs.scores):
                    # TODO: 소프트맥스로 확률 변환
                    token_prob = # TODO: F.softmax(outputs.scores[i-1], dim=-1)[0, token_id].item()
                    token_probs.append(token_prob)
            
            # TODO: 전체 점수 계산 (토큰 확률들의 곱)
            if token_probs:
                score = # TODO: torch.prod(torch.tensor(token_probs)).item()
            else:
                score = 0.0
        else:
            score = 0.0

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "score": round(score, 4),
            "model_type": "generative",
            "generation_params": {
                "num_beams": self.num_beams,
                "max_length": self.max_length
            }
        }

    def infer_creative(self, question: str, context: str, creative_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        창의적 질문에 대한 답변 생성 (Generative QA의 장점!)
        
        Args:
            question: 질문 문자열
            context: 컨텍스트 문자열
            creative_params: 창의적 생성을 위한 파라미터들
            
        Returns:
            dict: 창의적 답변과 관련 정보
        """
        if creative_params is None:
            creative_params = {}
        
        # TODO: 창의적 생성을 위한 파라미터 설정
        creative_num_beams = creative_params.get("num_beams", 3)
        creative_max_length = creative_params.get("max_length", 100)
        do_sample = creative_params.get("do_sample", True)
        temperature = creative_params.get("temperature", 0.8)
        top_p = creative_params.get("top_p", 0.9)

        # TODO: T5 입력 형식 구성
        input_text = # TODO: 완성하세요
        inputs = # TODO: 완성하세요
        
        with torch.no_grad():
            # TODO: 창의적 파라미터로 생성
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # TODO: 창의적 생성 파라미터들을 완성하세요
                # max_length, num_beams, do_sample, temperature, top_p
            )

        # TODO: 답변 디코딩
        answer = # TODO: 완성하세요

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "model_type": "generative_creative",
            "generation_params": {
                "num_beams": creative_num_beams,
                "max_length": creative_max_length,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p
            }
        }

    class WebAPI(FlaskView):
        """Flask 기반 웹 API 클래스"""
        
        def __init__(self, model: "GenerativeQAModel"):
            """
            WebAPI 초기화
            
            Args:
                model: GenerativeQAModel 인스턴스
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
            표준 생성형 QA API 엔드포인트
            
            POST /api
            Request Body: {"question": "질문", "context": "컨텍스트"}
            
            Returns:
                JSON: 생성된 답변과 관련 정보
            """
            # TODO: JSON 요청 데이터 파싱
            data = # TODO: request.json
            question = # TODO: data에서 "question" 추출, 기본값 ""
            context = # TODO: data에서 "context" 추출, 기본값 ""
            
            # TODO: 생성형 QA 모델로 답변 생성
            result = # TODO: self.model.infer_one() 호출
            
            # TODO: JSON 형태로 결과 반환
            return # TODO: jsonify() 사용

        @route('/api/creative', methods=['POST'])
        def api_creative(self):
            """
            창의적 생성형 QA API 엔드포인트 (Generative QA의 고유 기능!)
            
            POST /api/creative
            Request Body: {
                "question": "질문", 
                "context": "컨텍스트",
                "creative_params": {...}
            }
            
            Returns:
                JSON: 창의적 답변과 생성 파라미터
            """
            # TODO: 창의적 요청 데이터 파싱
            data = request.json
            question = data.get("question", "")
            context = data.get("context", "")
            creative_params = data.get("creative_params", {})
            
            # TODO: 창의적 답변 생성
            result = # TODO: self.model.infer_creative() 호출
            
            return jsonify(result)

        @route('/api/compare', methods=['POST'])
        def api_compare(self):
            """
            표준 vs 창의적 생성 비교 API
            
            POST /api/compare  
            Request Body: {"question": "질문", "context": "컨텍스트"}
            
            Returns:
                JSON: 두 가지 방식의 답변 비교
            """
            # TODO: 비교 요청 데이터 파싱
            data = request.json
            question = data.get("question", "")
            context = data.get("context", "")
            
            # TODO: 표준 답변 생성
            standard_result = # TODO: self.model.infer_one() 호출
            
            # TODO: 창의적 답변 생성
            creative_result = # TODO: self.model.infer_creative() 호출
            
            return jsonify({
                "question": question,
                "context": context,
                "standard_answer": standard_result,
                "creative_answer": creative_result,
                "comparison": {
                    "standard_length": len(standard_result["answer"].split()),
                    "creative_length": len(creative_result["answer"].split()),
                    "length_difference": len(creative_result["answer"].split()) - len(standard_result["answer"].split())
                }
            })

        @route('/health')
        def health(self):
            """서비스 상태 확인 엔드포인트"""
            # TODO: 서비스 상태 정보 반환
            return jsonify({
                "status": "healthy",
                "model_type": "generative_qa",
                "model_loaded": self.model.model is not None,
                "tokenizer_loaded": self.model.tokenizer is not None,
                "generation_params": {
                    "num_beams": self.model.num_beams,
                    "max_length": self.model.max_length
                }
            })


# === CLI 애플리케이션 ===

main = typer.Typer()

@main.command()
def serve(
    # TODO: CLI 옵션들 정의
    pretrained: str = typer.Option(
        # TODO: 기본값 설정 (로컬 체크포인트 또는 Hub ID)
        help="사전학습된 T5 QA 모델 경로 또는 Hugging Face Hub ID"
    ),
    server_host: str = typer.Option(
        # TODO: 기본값 "0.0.0.0",
        help="서버 호스트 주소"
    ),
    server_port: int = typer.Option(
        # TODO: 기본값 9165,
        help="서버 포트 번호"
    ),
    server_page: str = typer.Option(
        # TODO: 기본값 "serve_qa_seq2seq.html",
        help="웹 템플릿 파일명"
    ),
    num_beams: int = typer.Option(
        # TODO: 기본값 5,
        help="Beam Search 폭"
    ),
    max_length: int = typer.Option(
        # TODO: 기본값 50,
        help="최대 생성 길이"
    ),
    debug: bool = typer.Option(
        # TODO: 기본값 False,
        help="Flask 디버그 모드 활성화"
    ),
):
    """
    생성형 QA 웹 서비스 실행
    
    기능:
    - 실시간 생성형 질의응답 웹 인터페이스 제공
    - 표준 및 창의적 생성 API 엔드포인트 제공
    - 생성 파라미터 조정 지원
    - Extractive vs Generative 비교 가능
    """
    # TODO: 로깅 설정
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # TODO: 로컬 체크포인트 경로 처리 (glob 패턴 지원)
    import glob
    
    if "*" in pretrained:
        # Glob 패턴으로 체크포인트 찾기
        checkpoint_paths = glob.glob(pretrained)
        if checkpoint_paths:
            # TODO: 가장 최근 파일 선택
            pretrained = # TODO: 완성하세요 (파일 수정 시간 기준)
        else:
            raise ValueError(f"No checkpoint found matching pattern: {pretrained}")

    print(f"Starting Generative QA service with model: {pretrained}")
    
    # TODO: 생성형 QA 모델 로드
    model = # TODO: GenerativeQAModel 인스턴스 생성
    
    # TODO: Flask 앱 생성
    app = # TODO: Flask(__name__, template_folder=Path("templates").resolve())

    # TODO: 웹 서비스 실행
    # TODO: model.run_server() 호출


@main.command()
def test():
    """
    생성형 QA 모델 테스트 (표준 vs 창의적 생성 비교)
    """
    # TODO: 테스트용 모델 로드
    pretrained = # TODO: 기본 모델 경로 설정
    model = GenerativeQAModel(pretrained=pretrained, server_page="", num_beams=5, max_length=50)
    
    # TODO: 테스트 데이터
    test_cases = [
        {
            "question": "대한민국의 수도는?",
            "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다."
        },
        {
            "question": "대한민국에 대해 설명해주세요",  # 창의적 질문!
            "context": "대한민국은 동아시아의 한반도 남부에 위치한 나라이다. 수도는 서울특별시이며, 인구는 약 5천만명이다."
        }
    ]
    
    print("=== 생성형 QA 모델 테스트 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        
        # TODO: 표준 생성
        standard_result = # TODO: model.infer_one() 호출
        
        # TODO: 창의적 생성
        creative_result = # TODO: model.infer_creative() 호출
        
        print(f"Question: {standard_result['question']}")
        print(f"Standard Answer: {standard_result['answer']}")
        print(f"Standard Score: {standard_result['score']}")
        print(f"Creative Answer: {creative_result['answer']}")
        print(f"Creative Params: {creative_result['generation_params']}")


@main.command()
def compare_qa_types():
    """
    Extractive QA vs Generative QA 비교 데모
    """
    print("=== Extractive QA vs Generative QA 비교 ===")
    print()
    print("📊 특징 비교:")
    print("┌─────────────────┬─────────────────────┬─────────────────────┐")
    print("│ 특징            │ Extractive QA       │ Generative QA       │")
    print("├─────────────────┼─────────────────────┼─────────────────────┤")
    print("│ 답변 방식       │ 컨텍스트에서 추출   │ 새로운 텍스트 생성  │")
    print("│ 모델 구조       │ BERT (Encoder)      │ T5 (Encoder-Decoder)│")
    print("│ 후처리 복잡도   │ 매우 높음 ⭐⭐⭐⭐⭐   │ 매우 낮음 ⭐        │")
    print("│ 창의적 답변     │ 불가능              │ 가능 ✅             │")
    print("│ 사실성 보장     │ 높음 ✅             │ 낮음 (검증 필요)   │")
    print("│ 처리 속도       │ 빠름 ✅             │ 느림               │")
    print("│ 요약/설명       │ 불가능              │ 가능 ✅             │")
    print("└─────────────────┴─────────────────────┴─────────────────────┘")
    print()
    print("🎯 사용 시나리오:")
    print("Extractive QA: 정확한 정보 추출, 팩트 체킹, 빠른 응답")
    print("Generative QA: 설명, 요약, 창의적 답변, 교육 도구")


if __name__ == "__main__":
    main()

"""
학습 목표:
1. 생성형 QA의 독특한 웹 서비스 구현 방법 이해
2. Extractive QA와의 서비스 설계 차이점 파악
3. 창의적 생성 파라미터의 실제 활용 체험
4. 두 가지 QA 방식의 장단점 비교 분석

핵심 개념:

1. Generative QA 서비스 특징:
   - 텍스트 생성 기반 답변
   - 창의적 생성 파라미터 지원
   - 요약, 설명, 추론 등 복합 답변 가능
   - 사용자 맞춤형 생성 옵션

2. 서비스 아키텍처 비교:

   **Extractive QA 서비스**:
   - 단순한 입력-출력 구조
   - 고정된 추론 방식
   - 빠른 응답 시간
   - 제한된 답변 형태

   **Generative QA 서비스**:
   - 유연한 생성 파라미터
   - 다양한 생성 모드 (표준/창의적)
   - 느린 응답 시간
   - 다양한 답변 형태

3. 창의적 생성 파라미터:
   - do_sample=True: 확률적 생성 활성화
   - temperature: 창의성 조절 (0.7-1.2)
   - top_p: Nucleus sampling (0.8-0.95)
   - num_beams: 품질과 다양성 균형

4. API 설계 차이점:

   **Extractive QA API**:
   ```json
   POST /api
   {"question": "...", "context": "..."}
   →
   {"answer": "...", "score": 0.95, "start": 10, "end": 15}
   ```

   **Generative QA API**:
   ```json
   POST /api
   {"question": "...", "context": "..."}
   →
   {"answer": "...", "score": 0.85, "generation_params": {...}}
   
   POST /api/creative
   {"question": "...", "context": "...", "creative_params": {...}}
   →
   {"answer": "...", "generation_params": {...}}
   ```

5. 실시간 생성 최적화:
   - Beam Search 크기 조절
   - 최대 길이 제한
   - 배치 처리 활용
   - GPU 메모리 관리

6. 서비스 모니터링:
   - 생성 시간 측정
   - 답변 품질 추적
   - 사용자 만족도 수집
   - 생성 파라미터 최적화

7. 창의적 활용 사례:
   - 교육: "설명해주세요", "요약해주세요"
   - 고객지원: "해결 방법을 알려주세요"
   - 연구: "이것의 의미는 무엇인가요?"
   - 창작: "이야기를 만들어주세요"

8. 품질 관리:
   - 생성 결과 필터링
   - 부적절한 내용 탐지
   - 사실성 검증 시스템
   - 사용자 피드백 수집

실무 배포 고려사항:

1. 성능 최적화:
   - 모델 양자화 적용
   - 캐싱 전략 구현
   - 비동기 처리 도입
   - 로드 밸런싱 설정

2. 품질 보장:
   - 생성 결과 검증
   - 독성 콘텐츠 필터링
   - 사실성 체크 시스템
   - 편향성 모니터링

3. 사용자 경험:
   - 실시간 생성 진행률 표시
   - 다양한 생성 옵션 제공
   - 결과 만족도 피드백
   - 개인화 설정 지원

4. 확장성:
   - 다중 모델 지원
   - A/B 테스트 프레임워크
   - 실시간 모델 업데이트
   - 글로벌 서비스 대응

활용 분야:
- 교육 플랫폼: 설명형 QA 시스템
- 고객 지원: 복합 문제 해결 가이드
- 콘텐츠 플랫폼: 창의적 답변 생성
- 연구 도구: 논문 요약 및 해석
- 언어 학습: 대화형 튜터 시스템
"""
