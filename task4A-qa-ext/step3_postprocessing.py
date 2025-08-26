# === Step 3: QA 후처리 (Post-processing) ===
# 수강생 과제: TODO 부분을 완성하여 QA 모델의 복잡한 후처리 과정을 이해하세요.
# 이 단계는 QA에서 가장 복잡하고 중요한 부분입니다!

import collections
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def postprocess_qa_predictions(
    examples,
    features, 
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    version_2_with_negative: bool = False,
    null_score_diff_threshold: float = 0.0,
):
    """
    QA 모델의 start/end logits을 실제 답변 텍스트로 변환하는 후처리 함수
    
    Args:
        examples: 원본 QA 데이터
        features: 전처리된 features
        predictions: (start_logits, end_logits) 튜플
        n_best_size: 고려할 최고 답변 후보 수
        max_answer_length: 최대 답변 길이
        version_2_with_negative: SQuAD v2.0 형식 (답변 없음 허용) 여부
        null_score_diff_threshold: "답변 없음" 판단 임계값
        
    Returns:
        dict: {example_id: predicted_answer} 형태의 결과
    """
    # TODO: start_logits과 end_logits 분리
    # 힌트: predictions는 (start_logits, end_logits) 튜플
    start_logits, end_logits = # TODO: 완성하세요
    
    # TODO: example_id를 키로 하는 feature 매핑 생성
    # 힌트: features의 "example_id"를 사용하여 딕셔너리 구성
    all_features = {}
    for feature in features:
        example_id = # TODO: feature에서 example_id 추출
        if example_id not in all_features:
            all_features[example_id] = []
        all_features[example_id].append(feature)
    
    # 최종 예측 결과를 저장할 딕셔너리
    all_predictions = {}
    
    # TODO: 각 원본 예제에 대해 후처리 수행
    for example_index, example in enumerate(examples):
        example_id = # TODO: example에서 id 추출
        
        # TODO: 현재 예제에 해당하는 features 가져오기
        context = example["context"]
        features = # TODO: all_features에서 해당 예제의 features 가져오기
        
        # 유효한 답변 후보들을 저장할 리스트
        valid_answers = []
        
        # TODO: 각 feature에 대해 답변 후보 생성
        for feature_index, feature in enumerate(features):
            # TODO: 현재 feature의 start/end logits 가져오기
            # 힌트: start_logits[feature_index], end_logits[feature_index]
            start_logit = # TODO: 완성하세요
            end_logit = # TODO: 완성하세요
            
            # TODO: offset_mapping 가져오기
            # 힌트: feature["offset_mapping"]
            offset_mapping = # TODO: 완성하세요
            
            # TODO: 가능한 시작 위치들 찾기 (상위 n_best_size개)
            # 힌트: np.argsort()를 사용하여 내림차순 정렬 후 상위 n개 선택
            start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = # TODO: end_logit에 대해서도 동일하게 처리
            
            # TODO: 모든 시작-끝 조합에 대해 검증
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # TODO: 기본 유효성 검사
                    # 1. 시작 인덱스가 끝 인덱스보다 작거나 같아야 함
                    # 2. 답변 길이가 max_answer_length 이하여야 함
                    # 3. offset_mapping이 None이 아니어야 함 (컨텍스트 부분)
                    if (start_index >= len(offset_mapping) or 
                        end_index >= len(offset_mapping) or
                        offset_mapping[start_index] is None or 
                        offset_mapping[end_index] is None):
                        continue
                    
                    if # TODO: 추가 유효성 검사 조건들:
                        continue
                    
                    # TODO: 문자 단위 시작/끝 위치 계산
                    # 힌트: offset_mapping[start_index][0], offset_mapping[end_index][1]
                    start_char = # TODO: 완성하세요
                    end_char = # TODO: 완성하세요
                    
                    # TODO: 컨텍스트에서 답변 텍스트 추출
                    # 힌트: context[start_char:end_char]
                    answer_text = # TODO: 완성하세요
                    
                    # TODO: 답변 점수 계산
                    # 힌트: start_logit[start_index] + end_logit[end_index]
                    score = # TODO: 완성하세요
                    
                    # 유효한 답변 후보로 추가
                    valid_answers.append({
                        "score": score,
                        "text": answer_text,
                        "start_char": start_char,
                        "end_char": end_char
                    })
        
        # TODO: 답변 후보들을 점수 순으로 정렬
        # 힌트: sorted() 함수와 key=lambda x: x["score"] 사용, reverse=True
        valid_answers = # TODO: 완성하세요
        
        # TODO: SQuAD v2.0의 경우 "답변 없음" 처리
        if version_2_with_negative:
            # CLS 토큰 점수 계산 (답변 없음의 점수)
            cls_score = # TODO: start_logits[0][0] + end_logits[0][0] 계산
            
            # 최고 답변과 CLS 점수 비교
            if valid_answers and valid_answers[0]["score"] > cls_score + null_score_diff_threshold:
                # TODO: 최고 점수 답변 선택
                best_answer = # TODO: 완성하세요
            else:
                # TODO: "답변 없음" 반환
                best_answer = # TODO: 빈 문자열 또는 특별한 답변 없음 표시
        else:
            # TODO: SQuAD v1.0의 경우 최고 점수 답변 선택
            if valid_answers:
                best_answer = # TODO: 완성하세요
            else:
                best_answer = ""  # 백업 답변
        
        # TODO: 최종 결과에 추가
        all_predictions[example_id] = # TODO: best_answer의 텍스트 부분
    
    return all_predictions


def simple_postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, str]:
    """
    간단한 버전의 QA 후처리 함수 (학습용)
    
    Args:
        examples: 원본 QA 데이터
        features: 전처리된 features  
        predictions: (start_logits, end_logits)
        
    Returns:
        dict: {example_id: predicted_answer}
    """
    # TODO: start/end logits 분리
    start_logits, end_logits = # TODO: 완성하세요
    
    predictions = {}
    
    # TODO: 각 feature에 대해 단순 후처리
    for i, feature in enumerate(features):
        # TODO: 가장 높은 확률의 시작/끝 위치 찾기
        start_index = # TODO: np.argmax() 사용
        end_index = # TODO: np.argmax() 사용
        
        # TODO: 기본 검증
        if start_index <= end_index and end_index - start_index < 30:
            # TODO: offset_mapping 가져오기
            offset_mapping = # TODO: 완성하세요
            
            # TODO: 컨텍스트에서 해당하는 예제 찾기
            example_id = feature["example_id"]
            
            # 해당 예제의 컨텍스트 가져오기
            context = None
            for example in examples:
                if example["id"] == example_id:
                    context = example["context"]
                    break
            
            if context and start_index < len(offset_mapping) and end_index < len(offset_mapping):
                if offset_mapping[start_index] is not None and offset_mapping[end_index] is not None:
                    # TODO: 문자 위치 계산 및 답변 추출
                    start_char = # TODO: 완성하세요
                    end_char = # TODO: 완성하세요
                    answer = # TODO: 완성하세요
                    
                    predictions[example_id] = answer
    
    return predictions


if __name__ == "__main__":
    # === 후처리 테스트를 위한 예제 데이터 ===
    
    # TODO: 예제 데이터 구성
    examples = [
        {
            "id": "test_001",
            "question": "대한민국의 수도는?",
            "context": "대한민국은 동아시아에 위치한 나라이다. 수도는 서울특별시이다.",
            "answers": {"answer_start": [25], "text": ["서울특별시"]}
        }
    ]
    
    # TODO: 가상의 features 구성 (실제로는 전처리에서 생성)
    features = [
        {
            "example_id": "test_001",
            "offset_mapping": [
                None, None, None, None, None,  # [CLS] + 질문 토큰들
                (0, 3), (3, 6), (6, 8), (8, 12), (12, 14),  # 컨텍스트 시작
                (14, 16), (16, 18), (18, 22), (22, 24), (24, 25),  # "수도는"
                (25, 27), (27, 29), (29, 31), (31, 33), (33, 34),  # "서울특별시"
                (34, 35), None  # "이다" + [SEP]
            ]
        }
    ]
    
    # TODO: 가상의 predictions 구성 (실제로는 모델 출력)
    # start_logits과 end_logits를 numpy 배열로 구성
    start_logits = np.array([
        # TODO: 각 토큰 위치별 시작 확률 로그 값
        # 힌트: "서울특별시"의 시작 위치 (인덱스 10)에서 높은 값
        [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 
         0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 인덱스 10에서 최고값
    ])
    
    end_logits = np.array([
        # TODO: 각 토큰 위치별 끝 확률 로그 값  
        # 힌트: "서울특별시"의 끝 위치 (인덱스 14)에서 높은 값
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]  # 인덱스 14에서 최고값
    ])
    
    predictions = (start_logits, end_logits)
    
    # === 후처리 실행 및 결과 확인 ===
    
    print("=== 간단한 후처리 결과 ===")
    # TODO: 간단한 후처리 함수 실행
    simple_results = # TODO: 완성하세요
    print(f"Simple results: {simple_results}")
    
    print("\n=== 전체 후처리 결과 ===")
    # TODO: 전체 후처리 함수 실행
    full_results = # TODO: 완성하세요
    print(f"Full results: {full_results}")
    
    # 결과 비교 및 검증
    for example in examples:
        example_id = example["id"]
        expected_answer = example["answers"]["text"][0]
        
        simple_pred = simple_results.get(example_id, "")
        full_pred = full_results.get(example_id, "")
        
        print(f"\nExample ID: {example_id}")
        print(f"Expected: '{expected_answer}'")
        print(f"Simple prediction: '{simple_pred}'")
        print(f"Full prediction: '{full_pred}'")
        print(f"Simple match: {simple_pred == expected_answer}")
        print(f"Full match: {full_pred == expected_answer}")

"""
학습 목표:
1. QA 모델의 복잡한 후처리 과정 완전 이해
2. Start/End Logits을 실제 답변 텍스트로 변환하는 과정 체험
3. N-best 답변 후보 생성과 점수 기반 선택 방법 학습
4. SQuAD v2.0의 "답변 없음" 처리 방식 이해

핵심 개념:

1. 후처리의 복잡성:
   - QA에서 가장 복잡하고 중요한 단계
   - 단순한 argmax로는 최적 답변 보장 안됨
   - 다양한 제약 조건과 검증 필요

2. N-best 후보 생성:
   - 단일 최고 점수가 아닌 상위 N개 조합 고려
   - 시작×끝 모든 조합의 점수 계산
   - 유효성 검사를 통한 후보 필터링

3. 유효성 검사:
   - start_index <= end_index
   - 답변 길이 <= max_answer_length  
   - offset_mapping이 None이 아님 (컨텍스트 부분)
   - 실제 컨텍스트 범위 내 위치

4. 점수 계산:
   - start_logit + end_logit 합산
   - 로그 확률의 덧셈 = 확률의 곱셈
   - 높은 점수 = 높은 신뢰도

5. SQuAD v2.0 특별 처리:
   - CLS 토큰 점수 = "답변 없음" 점수
   - null_score_diff_threshold로 임계값 조정
   - 답변 신뢰도가 낮으면 "답변 없음" 반환

6. 문자-토큰 변환:
   - offset_mapping을 통한 정확한 위치 계산
   - 토큰 경계와 문자 경계의 매핑
   - 서브워드 토크나이제이션 고려

실무 중요성:
- 후처리 품질이 최종 QA 성능 크게 좌우
- 잘못된 후처리로 인한 성능 손실 빈발
- 언어별, 도메인별 최적화 필요
- 실시간 서비스에서 속도와 정확성 균형

디버깅 팁:
- offset_mapping 정확성 확인
- 예상 답변 위치와 예측 위치 비교
- 점수 분포와 임계값 조정
- 다양한 길이의 답변에 대한 테스트
"""
