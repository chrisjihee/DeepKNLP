# === Step 4: QA 모델 학습 ===
# 수강생 과제: TODO 부분을 완성하여 KorQuAD 데이터로 BERT 기반 QA 모델을 학습하세요.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# 이전 단계에서 구현한 함수들 import (실제로는 별도 파일에서)
from step2_data_preprocessing import prepare_train_features, prepare_validation_features
from step3_postprocessing import postprocess_qa_predictions

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """모델 관련 인수들"""
    
    # TODO: 사전학습 모델 경로 설정
    model_name_or_path: str = field(
        default=# TODO: "monologg/koelectra-base-v3" 또는 다른 한국어 BERT 모델,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    # TODO: 설정 파일 경로 (선택사항)
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    # TODO: 토크나이저 경로 (선택사항)  
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataTrainingArguments:
    """데이터 관련 인수들"""
    
    # TODO: 데이터셋 관련 설정
    dataset_name: Optional[str] = field(
        default=# TODO: None 또는 "squad_kor_v1",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    # TODO: 학습 파일 경로
    train_file: Optional[str] = field(
        default=# TODO: "data/KorQuAD_v1.0_train.json",
        metadata={"help": "The input training data file (a json file)."}
    )
    
    # TODO: 검증 파일 경로
    validation_file: Optional[str] = field(
        default=# TODO: "data/KorQuAD_v1.0_dev.json", 
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a json file)."}
    )
    
    # TODO: QA 특화 파라미터들
    max_seq_length: int = field(
        default=# TODO: 384,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    
    doc_stride: int = field(
        default=# TODO: 128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."}
    )
    
    n_best_size: int = field(
        default=# TODO: 20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."}
    )
    
    max_answer_length: int = field(
        default=# TODO: 30,
        metadata={"help": "The maximum length of an answer that can be generated."}
    )
    
    version_2_with_negative: bool = field(
        default=# TODO: False,
        metadata={"help": "If true, some of the examples do not have an answer."}
    )

def main():
    # === 1. 인수 파싱 ===
    
    # TODO: HfArgumentParser를 사용하여 인수 파서 생성
    # 힌트: HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = # TODO: 완성하세요
    
    # TODO: 명령행 인수 파싱
    # 힌트: parser.parse_args_into_dataclasses() 사용
    model_args, data_args, training_args = # TODO: 완성하세요

    # === 2. 로깅 설정 ===
    
    # TODO: 로깅 기본 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # TODO: 로그 레벨 설정
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # === 3. 시드 설정 ===
    
    # TODO: 재현 가능한 실험을 위한 시드 설정
    # 힌트: set_seed() 함수 사용
    # TODO: 완성하세요

    # === 4. 데이터셋 로드 ===
    
    # TODO: 데이터셋 로드 방식 결정
    if data_args.dataset_name is not None:
        # Hugging Face Hub에서 로드
        raw_datasets = # TODO: load_dataset() 사용
    else:
        # 로컬 파일에서 로드
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = # TODO: data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = # TODO: data_args.validation_file
            
        # TODO: JSON 형식으로 데이터셋 로드
        raw_datasets = # TODO: load_dataset("json", data_files=data_files)

    # === 5. 모델과 토크나이저 로드 ===
    
    # TODO: 모델 설정 로드
    # 힌트: AutoConfig.from_pretrained() 사용
    config = # TODO: 완성하세요
    
    # TODO: 토크나이저 로드
    # 힌트: AutoTokenizer.from_pretrained(), use_fast=True 설정
    tokenizer = # TODO: 완성하세요
    
    # TODO: QA 모델 로드
    # 힌트: AutoModelForQuestionAnswering.from_pretrained() 사용
    model = # TODO: 완성하세요

    # === 6. 데이터 전처리 ===
    
    # TODO: 컬럼명 확인
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
        
    # TODO: 컬럼명 매핑
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # TODO: 최대 시퀀스 길이 조정
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # === 7. 전처리 함수 정의 ===
    
    def prepare_train_features_with_args(examples):
        """학습용 전처리 함수 (인수 적용)"""
        # TODO: Step 2에서 구현한 전처리 함수 호출
        # 힌트: 필요한 인수들을 전달하여 호출
        return prepare_train_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=data_args.doc_stride,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name
        )

    def prepare_validation_features_with_args(examples):
        """검증용 전처리 함수 (인수 적용)"""
        # TODO: Step 2에서 구현한 전처리 함수 호출
        return prepare_validation_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=data_args.doc_stride,
            question_column_name=question_column_name,
            context_column_name=context_column_name
        )

    # === 8. 학습 데이터셋 전처리 ===
    
    if training_args.do_train:
        # TODO: 학습 데이터셋 가져오기
        train_dataset = # TODO: raw_datasets["train"]
        
        # TODO: 전처리 적용
        # 힌트: dataset.map() 사용, batched=True, remove_columns=column_names
        train_dataset = train_dataset.map(
            # TODO: 필요한 인수들을 완성하세요
        )

    # === 9. 검증 데이터셋 전처리 ===
    
    if training_args.do_eval:
        # TODO: 검증 데이터셋 가져오기 및 전처리
        eval_examples = # TODO: raw_datasets["validation"]
        eval_dataset = eval_examples.map(
            # TODO: 필요한 인수들을 완성하세요
        )

    # === 10. 데이터 콜레이터 설정 ===
    
    # TODO: 데이터 콜레이터 선택
    # 힌트: 패딩 필요 시 DataCollatorWithPadding, 아니면 default_data_collator
    data_collator = (
        default_data_collator 
        if data_args.pad_to_max_length 
        else # TODO: DataCollatorWithPadding() 사용
    )

    # === 11. 후처리 함수 정의 ===
    
    def post_processing_function(examples, features, predictions, stage="eval"):
        """후처리 함수"""
        # TODO: Step 3에서 구현한 후처리 함수 호출
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            # TODO: 필요한 인수들 추가
        )
        
        # TODO: 평가 형식에 맞게 변환
        formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
        
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # === 12. 평가 메트릭 설정 ===
    
    # TODO: SQuAD 메트릭 로드
    # 힌트: evaluate.load("squad" 또는 "squad_v2") 사용
    metric = evaluate.load(
        # TODO: version_2_with_negative에 따라 "squad" 또는 "squad_v2" 선택
    )

    def compute_metrics(p: EvalPrediction):
        # TODO: 메트릭 계산
        return # TODO: metric.compute() 호출

    # === 13. 커스텀 Trainer 클래스 (간소화된 버전) ===
    
    from transformers import Trainer
    
    class QuestionAnsweringTrainer(Trainer):
        def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_examples = eval_examples
            self.post_process_function = post_process_function

        def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            # TODO: 평가 로직 구현 (간소화)
            # 실제로는 더 복잡한 로직이 필요하지만 여기서는 기본 evaluate 사용
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    # === 14. Trainer 초기화 ===
    
    # TODO: QuestionAnsweringTrainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        # TODO: 필요한 인수들을 완성하세요
        # train_dataset, eval_dataset, tokenizer, data_collator, 
        # post_process_function, compute_metrics, eval_examples
    )

    # === 15. 학습 실행 ===
    
    if training_args.do_train:
        print("=== 학습 시작 ===")
        # TODO: 학습 실행
        # 힌트: trainer.train() 호출
        train_result = # TODO: 완성하세요
        
        # TODO: 모델 저장
        # 힌트: trainer.save_model() 사용
        # TODO: 완성하세요
        
        # TODO: 학습 메트릭 출력
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # === 16. 평가 실행 ===
    
    if training_args.do_eval:
        print("=== 평가 시작 ===")
        # TODO: 평가 실행
        # 힌트: trainer.evaluate() 호출
        metrics = # TODO: 완성하세요
        
        # TODO: 평가 메트릭 출력
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print(f"평가 결과: {metrics}")

if __name__ == "__main__":
    main()

"""
학습 목표:
1. Hugging Face Transformers를 활용한 QA 모델 학습 파이프라인 구축
2. KorQuAD 데이터셋의 구조와 특성 이해  
3. QA 특화 학습 설정과 하이퍼파라미터 튜닝
4. 전체 QA 시스템의 통합적 이해

핵심 개념:

1. QA 모델 구조:
   - BERT/ELECTRA + QA Head (Linear layers)
   - Start/End position 예측을 위한 분류 헤드
   - 기존 언어 모델에 task-specific 헤드 추가

2. KorQuAD 데이터셋:
   - 한국어 Reading Comprehension 데이터
   - SQuAD 형식: {id, question, context, answers}
   - answers: {answer_start: [위치], text: [답변]}

3. 학습 설정:
   - max_seq_length: 메모리와 성능의 트레이드오프
   - doc_stride: 긴 문서 처리를 위한 겹침 정도
   - learning_rate: QA는 보통 2e-5 ~ 5e-5 사용
   - 에포크: 3-5 에포크면 충분 (과적합 주의)

4. 평가 메트릭:
   - Exact Match (EM): 정확히 일치하는 답변 비율
   - F1 Score: 토큰 단위 겹침 기반 점수
   - SQuAD v1.0: EM, F1만 사용
   - SQuAD v2.0: HasAns/NoAns 분리 평가

5. 데이터 처리 파이프라인:
   - 전처리: 토크나이제이션, 위치 라벨링
   - 학습: start/end 위치 예측 학습
   - 후처리: logits → 실제 답변 텍스트 변환
   - 평가: 예측 답변과 정답 비교

6. 메모리 최적화:
   - Gradient Accumulation: 작은 배치 크기로 큰 배치 효과
   - Mixed Precision: FP16으로 메모리 절약
   - Dynamic Padding: 실제 길이에 맞춰 패딩

실무 고려사항:
- 긴 문서 처리: doc_stride와 max_seq_length 조정
- 도메인 적응: 일반 QA → 특정 도메인 QA
- 다국어 확장: 언어별 토크나이저와 모델 선택
- 실시간 서비스: 추론 속도 최적화

하이퍼파라미터 튜닝:
- learning_rate: 2e-5, 3e-5, 5e-5 비교
- batch_size: GPU 메모리에 따라 8, 16, 32
- max_seq_length: 384, 512 (긴 문서는 512)
- warmup_ratio: 0.1 (전체 스텝의 10%)

성능 향상 팁:
- 데이터 증강: 패러프레이징, 역번역
- 앙상블: 여러 모델의 예측 결합
- 후처리 최적화: n_best_size, threshold 조정
- 모델 선택: RoBERTa, ELECTRA, DeBERTa 비교
"""
