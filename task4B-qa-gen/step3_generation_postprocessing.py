# === Step 3: 텍스트 생성 및 후처리 ===
# 수강생 과제: TODO 부분을 완성하여 Seq2Seq QA의 생성과 후처리 과정을 이해하세요.
# Extractive QA와의 차이점: 복잡한 위치 기반 후처리 대신 간단한 텍스트 디코딩!

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

logger = logging.getLogger(__name__)

def generate_answers_with_scores(
    model,
    tokenizer, 
    examples,
    max_length: int = 50,
    num_beams: int = 5,
    return_scores: bool = True
) -> Dict[str, Any]:
    """
    Seq2Seq QA 모델을 사용하여 답변을 생성하고 점수를 계산하는 함수
    
    Args:
        model: T5 QA 모델
        tokenizer: T5 토크나이저
        examples: QA 예제들
        max_length: 생성할 최대 길이
        num_beams: Beam Search 폭
        return_scores: 점수 계산 여부
        
    Returns:
        dict: 예측 결과와 점수들
    """
    predictions = {}
    detailed_results = []
    
    for example in examples:
        # TODO: T5 입력 형식 구성
        # 힌트: "question: {질문} context: {컨텍스트}" 형태
        input_text = # TODO: 완성하세요
        
        # TODO: 입력 토크나이제이션
        # 힌트: tokenizer() 사용, return_tensors="pt"
        inputs = # TODO: 완성하세요
        
        with torch.no_grad():
            if return_scores:
                # TODO: 점수와 함께 생성
                # 힌트: model.generate() with return_dict_in_generate=True, output_scores=True
                outputs = model.generate(
                    # TODO: 필요한 인수들을 완성하세요
                    # input_ids, attention_mask, max_length, num_beams,
                    # return_dict_in_generate, output_scores
                )
                
                # TODO: 생성된 시퀀스에서 답변 추출
                generated_sequence = # TODO: outputs.sequences[0]
                
                # TODO: 답변 텍스트 디코딩
                answer = # TODO: tokenizer.decode() 사용, skip_special_tokens=True
                
                # TODO: 토큰별 점수 계산
                if return_scores and hasattr(outputs, 'scores'):
                    token_scores = []
                    for i, token_id in enumerate(generated_sequence):
                        if i == 0:  # 첫 번째 토큰(시작 토큰) 제외
                            continue
                        if i-1 < len(outputs.scores):
                            # TODO: 소프트맥스로 확률 변환
                            token_prob = # TODO: F.softmax(outputs.scores[i-1], dim=-1)[0, token_id].item()
                            token_scores.append(token_prob)
                    
                    # TODO: 전체 점수 계산 (토큰 확률들의 곱)
                    overall_score = # TODO: torch.prod(torch.tensor(token_scores)).item() if token_scores else 0.0
                else:
                    overall_score = 0.0
            else:
                # TODO: 점수 없이 간단한 생성
                outputs = model.generate(
                    # TODO: 필요한 인수들을 완성하세요
                )
                
                # TODO: 답변 추출 및 디코딩
                answer = # TODO: tokenizer.decode() 사용
                overall_score = 0.0
        
        # 결과 저장
        predictions[example["id"]] = answer
        detailed_results.append({
            "id": example["id"],
            "question": example["question"],
            "context": example["context"],
            "predicted_answer": answer,
            "expected_answer": example["answers"]["text"][0] if example["answers"]["text"] else "",
            "score": overall_score
        })
    
    return {
        "predictions": predictions,
        "detailed_results": detailed_results
    }

def simple_postprocess_seq2seq_predictions(
    examples,
    predictions: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Seq2Seq QA 예측 결과의 간단한 후처리
    
    Args:
        examples: 원본 예제들
        predictions: {example_id: predicted_answer} 딕셔너리
        
    Returns:
        list: 후처리된 결과들
    """
    # TODO: 후처리된 결과 리스트 초기화
    processed_results = []
    
    for example in examples:
        example_id = example["id"]
        predicted_answer = predictions.get(example_id, "")
        expected_answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        
        # TODO: 간단한 후처리 (텍스트 정제)
        # 힌트: strip(), 특수문자 제거 등
        cleaned_prediction = # TODO: predicted_answer.strip()
        cleaned_expected = # TODO: expected_answer.strip()
        
        # TODO: 정확도 계산 (완전 일치)
        exact_match = # TODO: cleaned_prediction.lower() == cleaned_expected.lower()
        
        # TODO: 단순 F1 점수 계산 (토큰 기반)
        pred_tokens = # TODO: cleaned_prediction.split()
        expected_tokens = # TODO: cleaned_expected.split()
        
        if not pred_tokens and not expected_tokens:
            f1_score = 1.0
        elif not pred_tokens or not expected_tokens:
            f1_score = 0.0
        else:
            # TODO: 공통 토큰 계산
            common_tokens = # TODO: set(pred_tokens) & set(expected_tokens)
            
            # TODO: Precision, Recall, F1 계산
            precision = # TODO: len(common_tokens) / len(pred_tokens)
            recall = # TODO: len(common_tokens) / len(expected_tokens) 
            f1_score = # TODO: 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        processed_results.append({
            "id": example_id,
            "question": example["question"],
            "predicted_answer": cleaned_prediction,
            "expected_answer": cleaned_expected,
            "exact_match": exact_match,
            "f1_score": f1_score
        })
    
    return processed_results

if __name__ == "__main__":
    # === 1. 모델과 토크나이저 로드 ===
    
    # TODO: 사전학습된 T5 QA 모델 로드
    pretrained = # TODO: "paust/pko-t5-base-finetuned-korquad"
    
    print(f"Loading model from {pretrained}")
    # TODO: 토크나이저와 모델 로드
    tokenizer = # TODO: AutoTokenizer.from_pretrained()
    model = # TODO: AutoModelForSeq2SeqLM.from_pretrained()
    model.eval()

    # === 2. 테스트 예제 준비 ===
    
    # TODO: 테스트용 QA 데이터 구성
    test_examples = [
        {
            "id": "test_001",
            "question": # TODO: "대한민국의 수도는?",
            "context": # TODO: "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다.",
            "answers": {"text": [# TODO: "서울특별시"]}
        },
        # TODO: 더 많은 테스트 예제 추가
    ]

    # === 3. 다양한 생성 전략 실험 ===
    
    print("=== 생성 전략 비교 ===")
    
    # TODO: Greedy Search (num_beams=1)
    print("\n1. Greedy Search:")
    greedy_results = # TODO: generate_answers_with_scores() 호출 (num_beams=1)
    
    for result in greedy_results["detailed_results"]:
        print(f"Q: {result['question']}")
        print(f"A: {result['predicted_answer']} (Score: {result['score']:.4f})")

    # TODO: Beam Search (num_beams=5)
    print("\n2. Beam Search:")
    beam_results = # TODO: generate_answers_with_scores() 호출 (num_beams=5)
    
    for result in beam_results["detailed_results"]:
        print(f"Q: {result['question']}")
        print(f"A: {result['predicted_answer']} (Score: {result['score']:.4f})")

    # === 4. 생성 길이 실험 ===
    
    print("\n=== 생성 길이 실험 ===")
    
    test_question = test_examples[0]
    
    # TODO: 다양한 최대 길이로 실험
    for max_len in [10, 30, 50, 100]:
        print(f"\nMax Length = {max_len}:")
        # TODO: generate_answers_with_scores() 호출 (max_length 변경)
        length_results = # TODO: 완성하세요
        
        answer = length_results["detailed_results"][0]["predicted_answer"]
        score = length_results["detailed_results"][0]["score"]
        print(f"Answer: {answer}")
        print(f"Score: {score:.4f}")
        print(f"Actual length: {len(answer.split())} words")

    # === 5. 후처리 및 평가 ===
    
    print("\n=== 후처리 및 평가 ===")
    
    # TODO: 후처리 함수 실행
    processed_results = # TODO: simple_postprocess_seq2seq_predictions() 호출
    
    # TODO: 전체 성능 계산
    total_exact_match = sum(r["exact_match"] for r in processed_results)
    total_f1_score = sum(r["f1_score"] for r in processed_results)
    num_examples = len(processed_results)
    
    print(f"전체 Exact Match: {total_exact_match}/{num_examples} ({total_exact_match/num_examples*100:.1f}%)")
    print(f"평균 F1 Score: {total_f1_score/num_examples:.3f}")
    
    # 개별 결과 출력
    for result in processed_results:
        print(f"\nID: {result['id']}")
        print(f"Question: {result['question']}")
        print(f"Predicted: '{result['predicted_answer']}'")
        print(f"Expected: '{result['expected_answer']}'")
        print(f"Exact Match: {result['exact_match']}")
        print(f"F1 Score: {result['f1_score']:.3f}")

    # === 6. Extractive vs Generative 후처리 비교 ===
    
    print("\n=== Extractive vs Generative 후처리 비교 ===")
    print("Extractive QA 후처리:")
    print("- N-best 후보 생성")
    print("- 복잡한 점수 계산")
    print("- 위치 기반 답변 추출")
    print("- 유효성 검사")
    print("- SQuAD v2.0 null 답변 처리")
    print()
    print("Generative QA 후처리:")
    print("- 직접 텍스트 디코딩")
    print("- 간단한 점수 계산")
    print("- 텍스트 정제")
    print("- 토큰 기반 평가")
    print("- 매우 단순!")

    # === 7. 창의적 질문 실험 ===
    
    print("\n=== 창의적 질문 실험 (Generative QA의 장점) ===")
    
    creative_examples = [
        {
            "id": "creative_001",
            "question": # TODO: "대한민국에 대해 간략히 설명해주세요",
            "context": # TODO: 긴 컨텍스트,
            "answers": {"text": [# TODO: "요약형 답변"]}
        },
        # TODO: 더 많은 창의적 질문 추가
    ]
    
    # TODO: 창의적 질문들에 대한 답변 생성
    creative_results = # TODO: generate_answers_with_scores() 호출
    
    for result in creative_results["detailed_results"]:
        print(f"\nCreative Q: {result['question']}")
        print(f"Generated A: {result['predicted_answer']}")
        print(f"Score: {result['score']:.4f}")

    # === 8. 점수 분석 ===
    
    print("\n=== 점수 분석 ===")
    
    all_scores = [r["score"] for r in beam_results["detailed_results"]]
    
    print(f"평균 점수: {np.mean(all_scores):.4f}")
    print(f"최고 점수: {np.max(all_scores):.4f}")
    print(f"최저 점수: {np.min(all_scores):.4f}")
    print(f"점수 표준편차: {np.std(all_scores):.4f}")
    
    print("\n점수 해석:")
    print("- 높은 점수: 모델이 확신 있는 답변")
    print("- 낮은 점수: 불확실하거나 어려운 질문")
    print("- 점수는 토큰 확률들의 곱이므로 길수록 낮아짐")

"""
학습 목표:
1. Seq2Seq QA의 간단하고 직관적인 후처리 과정 이해
2. Beam Search와 생성 파라미터의 효과 체험
3. 토큰 확률 기반 점수 계산 방법 학습
4. Extractive QA와의 후처리 복잡도 차이 체감

핵심 개념:

1. Generative QA 후처리의 단순함:
   - Extractive QA: 10단계 이상의 복잡한 후처리
   - Generative QA: 3단계 간단한 후처리
   1) 텍스트 디코딩
   2) 텍스트 정제 
   3) 평가 메트릭 계산

2. 텍스트 생성 과정:
   - model.generate(): 핵심 생성 메소드
   - return_dict_in_generate=True: 상세 정보 반환
   - output_scores=True: 토큰별 점수 정보 포함
   - skip_special_tokens=True: 특수 토큰 제거

3. Beam Search vs Greedy:
   - Greedy (num_beams=1): 빠르지만 단조로운 결과
   - Beam Search (num_beams>1): 더 나은 품질, 느린 속도
   - 실무에서는 num_beams=3~5가 적절

4. 점수 계산 방식:
   - 토큰별 확률: F.softmax(logits, dim=-1)
   - 전체 점수: 모든 토큰 확률의 곱
   - 긴 답변일수록 점수 낮아짐 (확률 곱셈 특성)

5. 평가 메트릭:
   - Exact Match: 완전 일치 여부
   - F1 Score: 토큰 단위 겹침 정도
   - BLEU, ROUGE: 더 정교한 생성 평가 (추가 고려)

6. 후처리 복잡도 비교:

   **Extractive QA 후처리**:
   - 10+ 단계의 복잡한 로직
   - N-best 후보 생성
   - 복잡한 점수 계산  
   - 위치 유효성 검사
   - SQuAD v2.0 처리
   - 문자-토큰 변환
   
   **Generative QA 후처리**:
   - 3단계 간단한 로직
   - 직접 텍스트 디코딩
   - 간단한 점수 계산
   - 텍스트 정제
   - 평가 메트릭 계산
   - 완료!

7. 장점:
   - 구현이 매우 간단
   - 이해하기 쉬움
   - 디버깅이 용이
   - 확장성이 높음
   - 다양한 답변 형태 지원

8. 주의사항:
   - Hallucination 가능성
   - 사실성 검증 필요
   - 생성 속도 고려
   - 점수 해석 주의

실무 활용:
- 교육: 설명형 답변 제공
- 고객지원: 복합 문제 해결
- 연구: 논문 요약 및 설명
- 창작: 스토리텔링 지원

성능 최적화:
- 생성 길이 조절로 속도 향상
- Beam 크기 조정으로 품질-속도 균형
- 배치 처리로 처리량 증대
- 모델 양자화로 메모리 절약
"""
