# === Step 3: 생성 파라미터 조정 ===
# 수강생 과제: TODO 부분을 완성하여 다양한 생성 파라미터의 효과를 실험하세요.

import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    # 모델과 토크나이저 로드
    pretrained = "skt/kogpt2-base-v2"
    
    model = GPT2LMHeadModel.from_pretrained(pretrained)
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained,
        eos_token="</s>",
    )

    input_sentence = "안녕하세요" or "대한민국의 수도는"
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

    print(f"입력 프롬프트: '{input_sentence}'")
    print()

    # === 1. 반복 방지: no_repeat_ngram_size ===
    print("[1] 반복 방지: no_repeat_ngram_size" + "-" * 45)
    
    print("  1-1) 기본 설정 (반복 가능성 있음)")
    with torch.no_grad():
        # TODO: 기본 Greedy Search로 생성 (반복 발생 가능)
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    print("  1-2) no_repeat_ngram_size=3 적용")
    with torch.no_grad():
        # TODO: no_repeat_ngram_size=3을 추가하여 3-gram 반복 방지
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 2. 반복 패널티: repetition_penalty ===
    print("\n[2] 반복 패널티: repetition_penalty" + "-" * 45)
    
    repetition_penalties = [1.0, 1.1, 1.2, 1.5]
    for penalty in repetition_penalties:
        print(f"  2-{repetition_penalties.index(penalty)+1}) repetition_penalty={penalty}")
        with torch.no_grad():
            # TODO: 각각의 repetition_penalty 값으로 텍스트 생성
            # 힌트: do_sample=False, repetition_penalty=penalty
            generated_ids = # TODO: 완성하세요
            print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 3. 온도 조절: temperature ===
    print("\n[3] 온도 조절: temperature" + "-" * 55)
    
    temperatures = [0.01, 1.0, 2.0]
    for temp in temperatures:
        print(f"  3-{temperatures.index(temp)+1}) temperature={temp}")
        with torch.no_grad():
            # TODO: 각각의 temperature 값으로 텍스트 생성
            # 힌트: do_sample=True, top_k=50, temperature=temp
            generated_ids = # TODO: 완성하세요
            print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 4. Top-k 값 변화 ===
    print("\n[4] Top-k 값 변화" + "-" * 60)
    
    top_k_values = [1, 10, 50]
    for k in top_k_values:
        print(f"  4-{top_k_values.index(k)+1}) top_k={k}")
        with torch.no_grad():
            # TODO: 각각의 top_k 값으로 텍스트 생성
            # 힌트: do_sample=True, top_k=k
            generated_ids = # TODO: 완성하세요
            print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 5. Top-p 값 변화 ===
    print("\n[5] Top-p 값 변화" + "-" * 60)
    
    top_p_values = [0.01, 0.5, 0.92]
    for p in top_p_values:
        print(f"  5-{top_p_values.index(p)+1}) top_p={p}")
        with torch.no_grad():
            # TODO: 각각의 top_p 값으로 텍스트 생성
            # 힌트: do_sample=True, top_p=p
            generated_ids = # TODO: 완성하세요
            print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 6. 파라미터 조합 실험 ===
    print("\n[6] 최적 파라미터 조합" + "-" * 55)
    
    print("  6-1) 보수적 설정 (안정적, 반복 적음)")
    with torch.no_grad():
        # TODO: 보수적 파라미터 조합으로 생성
        # 힌트: do_sample=True, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    print("  6-2) 창의적 설정 (다양성 높음)")
    with torch.no_grad():
        # TODO: 창의적 파라미터 조합으로 생성
        # 힌트: do_sample=True, temperature=1.2, top_k=100, top_p=0.95, repetition_penalty=1.3
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    print("  6-3) 극단적 창의 설정 (매우 높은 다양성)")
    with torch.no_grad():
        # TODO: 극단적 파라미터 조합으로 생성
        # 힌트: do_sample=True, temperature=2.0, top_k=200, repetition_penalty=1.5
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

    # === 7. 통합 최적화 설정 (infer_gen-1.py 마지막 설정) ===
    print("\n[7] 통합 최적화 설정" + "-" * 55)
    with torch.no_grad():
        # TODO: 모든 파라미터를 종합한 최적화 설정으로 생성
        # 힌트: do_sample=True, repetition_penalty=1.5, no_repeat_ngram_size=3, 
        #       temperature=0.9, top_k=50, top_p=0.92
        generated_ids = # TODO: 완성하세요
        print("      ", tokenizer.decode([el.item() for el in generated_ids[0]]))

"""
학습 목표:
1. 각 생성 파라미터의 역할과 효과 이해
2. 파라미터 조합이 생성 품질에 미치는 영향 분석
3. 다양성과 품질 간의 트레이드오프 체험
4. 실무에서 사용할 수 있는 최적 설정 발견

핵심 파라미터 효과:

1. no_repeat_ngram_size:
   - n-gram 단위 반복 방지
   - 값이 클수록 더 긴 패턴의 반복 방지
   - 3-4가 일반적으로 효과적

2. repetition_penalty:
   - 이미 생성된 토큰의 재출현 확률 감소
   - 1.0: 패널티 없음, >1.0: 패널티 적용
   - 1.1-1.5 범위가 일반적

3. temperature:
   - 확률 분포의 "날카로움" 조절
   - <1.0: 더 확실한 선택 (보수적)
   - =1.0: 원래 분포 유지
   - >1.0: 더 다양한 선택 (창의적)

4. top_k:
   - 상위 k개 토큰만 고려
   - 작을수록 보수적, 클수록 다양함
   - 40-100 범위가 일반적

5. top_p (nucleus sampling):
   - 누적 확률 p까지의 토큰들만 고려
   - 0.9-0.95 범위가 일반적
   - 동적으로 후보 수 조절

실무 팁:
- 안정적 생성: temperature=0.8, top_k=40, top_p=0.9
- 창의적 생성: temperature=1.2, top_k=100, top_p=0.95
- 반복 방지: repetition_penalty=1.1-1.3, no_repeat_ngram_size=3
"""
