# === Step 1: 텍스트 생성 기본 개념 ===
# 수강생 과제: TODO 부분을 완성하여 GPT2 모델의 기본 텍스트 생성을 이해하세요.

# === 라이브러리 Import ===
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    # === 텍스트 생성 모델과 토크나이저 설정 ===
    
    # TODO: 사전학습된 한국어 GPT2 모델명을 지정하세요
    # 힌트: "skt/kogpt2-base-v2" 사용
    pretrained = # TODO: 완성하세요

    # TODO: GPT2LMHeadModel을 로드하고 평가 모드로 설정하세요
    # 힌트: GPT2LMHeadModel.from_pretrained()와 model.eval() 사용
    model = # TODO: 완성하세요
    # TODO: 평가 모드 설정

    # TODO: PreTrainedTokenizerFast를 로드하세요
    # 힌트: eos_token="</s>" 설정 필요
    tokenizer = # TODO: 완성하세요

    # === 입력 텍스트 준비 ===
    
    # TODO: 생성을 시작할 프롬프트를 지정하세요
    # 힌트: "안녕하세요" 또는 "대한민국의 수도는" 등 사용
    input_sentence = # TODO: 완성하세요
    
    # TODO: 입력 텍스트를 토큰화하세요
    # 힌트: tokenizer.encode() 사용, return_tensors="pt" 설정
    input_ids = # TODO: 완성하세요

    # === 기본 텍스트 생성 (Greedy Search) ===
    print(f"입력 프롬프트: '{input_sentence}'")
    print(f"{'=' * 80}")
    
    with torch.no_grad():
        # TODO: 모델의 generate 메소드를 사용하여 텍스트를 생성하세요
        # 힌트: do_sample=False (Greedy Search), min_length=10, max_length=50
        generated_ids = # TODO: 완성하세요
        
        # TODO: 생성된 토큰 ID들을 텍스트로 디코딩하세요
        # 힌트: tokenizer.decode() 사용
        generated_text = # TODO: 완성하세요
        
        print("생성된 텍스트:")
        print(generated_text)

"""
학습 목표:
1. GPT2 언어 모델의 기본 구조 이해
2. 토크나이저의 역할과 사용법 학습
3. 기본적인 텍스트 생성 과정 체험
4. Greedy Search의 특징 파악

핵심 개념:
- GPT2LMHeadModel: 다음 토큰 예측을 위한 언어 모델
- PreTrainedTokenizerFast: 텍스트-토큰 변환 도구
- model.generate(): 텍스트 생성 메인 메소드
- do_sample=False: 확률이 가장 높은 토큰만 선택 (Greedy)
- min_length/max_length: 생성 길이 제어
- eos_token: 문장 종료 토큰 ("</s>")

예상 결과:
- 동일한 입력에 대해 항상 같은 출력 생성
- 자연스러운 한국어 문장 생성
- 프롬프트와 연관성 있는 내용 생성
"""
