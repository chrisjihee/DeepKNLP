# === Step 5: 텍스트 생성 웹 서비스 ===
# 수강생 과제: TODO 부분을 완성하여 Flask 기반 텍스트 생성 웹 서비스를 구현하세요.

from pathlib import Path
import torch

from ratsnlp.nlpbook.generation import GenerationDeployArguments
from ratsnlp.nlpbook.generation import get_web_service_app
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    # === Step 5-1: 디바이스 및 배포 인수 설정 ===
    
    # TODO: 사용 가능한 디바이스 설정 (GPU 우선, CPU 대체)
    # 힌트: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = # TODO: 완성하세요
    
    # TODO: 배포용 인수 설정
    # 힌트: GenerationDeployArguments 사용
    args = GenerationDeployArguments(
        # TODO: 사전학습 모델명 지정 (fine-tuning에 사용한 것과 동일)
        pretrained_model_name=# TODO: 완성하세요,
        
        # TODO: Fine-tuned 모델이 저장된 디렉토리 지정
        downstream_model_dir=# TODO: 완성하세요,
    )

    # === Step 5-2: 모델 설정 및 로드 ===
    
    # TODO: 사전학습 모델 설정 로드
    # 힌트: GPT2Config.from_pretrained() 사용
    pretrained_model_config = # TODO: 완성하세요
    
    # TODO: 모델 객체 생성 (가중치 로드 전)
    # 힌트: GPT2LMHeadModel(pretrained_model_config) 사용
    model = # TODO: 완성하세요
    
    # TODO: Fine-tuned 체크포인트 로드
    # 힌트: torch.load()를 사용하여 args.downstream_model_checkpoint_fpath 로드
    fine_tuned_model_ckpt = # TODO: 완성하세요
    
    # TODO: 모델에 Fine-tuned 가중치 로드
    # 힌트: model.load_state_dict() 사용, 키 이름 변환 필요
    # model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    # TODO: 완성하세요
    
    # TODO: 모델을 평가 모드로 설정
    # TODO: 완성하세요

    # === Step 5-3: 토크나이저 로드 ===
    
    # TODO: 토크나이저 로드 (학습 시와 동일한 설정)
    # 힌트: PreTrainedTokenizerFast.from_pretrained(), eos_token="</s>"
    tokenizer = # TODO: 완성하세요

    # === Step 5-4: 추론 함수 정의 ===
    
    def inference_fn(
            prompt,                     # 입력 프롬프트
            min_length=10,             # 최소 생성 길이
            max_length=20,             # 최대 생성 길이
            top_p=1.0,                 # Top-p (nucleus) sampling
            top_k=50,                  # Top-k sampling
            repetition_penalty=1.0,     # 반복 패널티
            no_repeat_ngram_size=0,    # n-gram 반복 방지 크기
            temperature=1.0,           # 온도 조절
    ):
        """
        웹 서비스에서 호출될 텍스트 생성 함수
        
        Args:
            prompt: 사용자가 입력한 텍스트 프롬프트
            기타: 생성 파라미터들 (웹 UI에서 조정 가능)
            
        Returns:
            Dict: {'result': 생성된 텍스트} 형태의 결과
        """
        try:
            # TODO: 입력 프롬프트를 토큰화
            # 힌트: tokenizer.encode() 사용, return_tensors="pt"
            input_ids = # TODO: 완성하세요
            
            with torch.no_grad():
                # TODO: 모델을 사용하여 텍스트 생성
                # 힌트: model.generate() 사용, 모든 파라미터 적용
                generated_ids = model.generate(
                    input_ids,
                    # TODO: 생성 파라미터들을 완성하세요
                    # do_sample, top_p, top_k, min_length, max_length,
                    # repetition_penalty, no_repeat_ngram_size, temperature
                )
            
            # TODO: 생성된 토큰 ID들을 텍스트로 디코딩
            # 힌트: tokenizer.decode() 사용
            generated_sentence = # TODO: 완성하세요
            
        except Exception as e:
            # 오류 발생 시 사용자 친화적 메시지 반환
            generated_sentence = f"""처리 중 오류가 발생했습니다. <br>
                오류 내용: {str(e)} <br><br>
                변수의 입력 범위를 확인하세요. <br><br> 
                min_length: 1 이상의 정수 <br>
                max_length: 1 이상의 정수 <br>
                top-p: 0 이상 1 이하의 실수 <br>
                top-k: 1 이상의 정수 <br>
                repetition_penalty: 1 이상의 실수 <br>
                no_repeat_ngram_size: 0 이상의 정수 <br>
                temperature: 0 초과의 실수
                """
        
        return {
            'result': generated_sentence,
        }

    # === Step 5-5: 웹 애플리케이션 실행 ===
    
    # TODO: 웹 서비스 애플리케이션 생성
    # 힌트: get_web_service_app() 사용
    app = get_web_service_app(
        # TODO: 필요한 인수들을 완성하세요
        # inference_fn, template_folder, server_page
    )
    
    # TODO: 웹 서버 실행
    # 힌트: app.run() 사용, host="0.0.0.0", port=9001
    # TODO: 완성하세요

"""
학습 목표:
1. 학습된 모델을 실제 서비스로 배포하는 과정 이해
2. Flask 기반 웹 애플리케이션 구조 학습
3. 실시간 텍스트 생성 API 구현 체험
4. 사용자 인터페이스와 백엔드 로직 연동 이해

핵심 개념:

1. 모델 배포 (Model Deployment):
   - 학습된 모델을 실제 서비스에서 사용할 수 있도록 배치
   - 체크포인트 로드, 모델 초기화, 추론 최적화
   - 실시간 요청 처리를 위한 효율적 구조

2. GenerationDeployArguments:
   - 배포 전용 설정 관리 클래스
   - 학습 설정과 분리된 배포 특화 인수들
   - 모델 경로, 서버 설정 등 포함

3. 웹 서비스 아키텍처:
   - Frontend: 사용자 인터페이스 (HTML/CSS/JavaScript)
   - Backend: Flask 기반 API 서버
   - Model: 텍스트 생성 추론 엔진

4. 실시간 추론 최적화:
   - torch.no_grad(): 기울기 계산 비활성화로 메모리 절약
   - model.eval(): 평가 모드로 드롭아웃 등 비활성화
   - 배치 처리보다는 단일 요청 최적화

5. 오류 처리 (Error Handling):
   - 사용자 입력 검증
   - 모델 추론 오류 대응
   - 친화적 오류 메시지 제공

웹 서비스 구성 요소:

1. 추론 함수 (inference_fn):
   - 사용자 입력을 받아 텍스트 생성
   - 다양한 생성 파라미터 지원
   - 오류 처리 및 결과 반환

2. 웹 애플리케이션 (Flask):
   - HTML 템플릿 렌더링
   - REST API 엔드포인트 제공
   - 사용자 요청-응답 관리

3. 사용자 인터페이스:
   - 프롬프트 입력 창
   - 생성 파라미터 조정 슬라이더
   - 실시간 결과 표시 영역

서비스 활용 방법:
1. 웹 브라우저에서 http://localhost:9001 접속
2. 텍스트 프롬프트 입력 (예: "오늘 날씨가")
3. 생성 파라미터 조정 (temperature, top-k 등)
4. 생성 버튼 클릭하여 결과 확인
5. 다양한 설정으로 실험 반복

실무 응용:
- 챗봇 백엔드 시스템
- 창작 도구 웹 서비스
- API 기반 텍스트 생성 서비스
- 프로토타입 데모 시스템

성능 고려사항:
- GPU 메모리 관리
- 동시 요청 처리 능력
- 응답 시간 최적화
- 서버 리소스 모니터링
"""
