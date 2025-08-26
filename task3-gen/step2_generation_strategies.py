# === Step 2: 다양한 텍스트 생성 전략 ===
# 수강생 과제: TODO 부분을 완성하여 Greedy Search, Beam Search, Sampling의 차이점을 학습하세요.

import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    # 모델과 토크나이저 로드 (Step 1에서 완성한 코드 재사용)
    pretrained = "skt/kogpt2-base-v2"
    
    model = GPT2LMHeadModel.from_pretrained(pretrained)
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained,
        eos_token="</s>",
    )

    # 다양한 프롬프트로 실험
    input_sentence = "안녕하세요" or "대한민국의 수도는"
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

    print(f"입력 프롬프트: '{input_sentence}'")
    print()

    # === 1. Greedy Search (기본) ===
    print("[1] Greedy Search" + "-" * 60)
    with torch.no_grad():
        # TODO: Greedy Search로 텍스트 생성
        # 힌트: do_sample=False, min_length=10, max_length=50
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 2. Beam Search (num_beams=3) ===
    print("\n[2] Beam Search (num_beams=3)" + "-" * 50)
    with torch.no_grad():
        # TODO: Beam Search로 텍스트 생성
        # 힌트: do_sample=False, num_beams=3 추가
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 3. Beam Search (num_beams=1) - Greedy와 동일 ===
    print("\n[3] Beam Search (num_beams=1 = Greedy)" + "-" * 40)
    with torch.no_grad():
        # TODO: num_beams=1로 설정하여 Greedy와 같은 결과 확인
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 4. Top-k Sampling (k=50) ===
    print("\n[4] Top-k Sampling (k=50)" + "-" * 50)
    with torch.no_grad():
        # TODO: Top-k Sampling으로 텍스트 생성
        # 힌트: do_sample=True, top_k=50
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 5. Top-k Sampling (k=1) - Greedy와 유사 ===
    print("\n[5] Top-k Sampling (k=1 ≈ Greedy)" + "-" * 45)
    with torch.no_grad():
        # TODO: k=1로 설정하여 Greedy와 유사한 결과 확인
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 6. Top-p Sampling (Nucleus Sampling) ===
    print("\n[6] Top-p Sampling (p=0.92)" + "-" * 50)
    with torch.no_grad():
        # TODO: Top-p Sampling으로 텍스트 생성
        # 힌트: do_sample=True, top_p=0.92
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 7. Top-p Sampling (p=0.01) - Greedy와 유사 ===
    print("\n[7] Top-p Sampling (p=0.01 ≈ Greedy)" + "-" * 45)
    with torch.no_grad():
        # TODO: p=0.01로 설정하여 Greedy와 유사한 결과 확인
        generated_ids = # TODO: 완성하세요
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 8. 동일한 설정으로 여러 번 실행하여 차이점 확인 ===
    print("\n[8] Top-k Sampling 3회 반복 (다양성 확인)" + "-" * 35)
    for i in range(3):
        with torch.no_grad():
            # TODO: 동일한 Top-k 설정으로 3번 생성하여 다양성 확인
            # 힌트: do_sample=True, top_k=50
            generated_ids = # TODO: 완성하세요
            print(f"실행 {i+1}: {tokenizer.decode([el.item() for el in generated_ids[0]])}")

"""
학습 목표:
1. Greedy Search vs Beam Search vs Sampling의 차이점 이해
2. 각 생성 전략의 장단점 파악
3. do_sample 파라미터의 역할 이해
4. 생성 결과의 다양성과 품질 트레이드오프 체험

핵심 개념:

1. Greedy Search (do_sample=False):
   - 매번 가장 높은 확률의 토큰 선택
   - 빠르고 안정적이지만 단조로운 결과
   - 항상 동일한 출력 생성

2. Beam Search (num_beams > 1):
   - 여러 경로를 동시에 탐색
   - Greedy보다 더 좋은 전체 확률 문장 생성 가능
   - 계산량이 많지만 품질 향상

3. Top-k Sampling (do_sample=True, top_k):
   - 상위 k개 토큰 중에서 확률적 샘플링
   - 다양한 결과 생성 가능
   - k=1이면 Greedy와 동일

4. Top-p Sampling (Nucleus Sampling):
   - 누적 확률 p까지의 토큰들 중에서 샘플링
   - 동적으로 후보 토큰 수 조절
   - 품질과 다양성의 균형

파라미터 효과:
- num_beams=1: Greedy Search와 동일
- top_k=1: Greedy Search와 유사
- top_p=0.01: Greedy Search와 유사
- top_p=1.0: 전체 어휘에서 샘플링 (품질 저하 가능)
"""
