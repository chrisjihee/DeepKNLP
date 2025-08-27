# === Step 4: Seq2Seq QA 모델 학습 ===
# 수강생 과제: TODO 부분을 완성하여 T5 기반 Generative QA 모델을 학습하세요.
# Extractive QA와의 차이점: Seq2SeqTrainer와 생성 기반 평가!

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

# 이전 단계에서 구현한 함수들 import (실제로는 별도 파일에서)
from step2_seq2seq_preprocessing import preprocess_squad_batch

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Seq2Seq 모델 관련 인수들"""
    
    # TODO: 사전학습 T5 모델 경로 설정
    model_name_or_path: str = field(
        default=# TODO: "paust/pko-t5-base" 또는 다른 한국어 T5 모델,
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
    
    # TODO: Fast tokenizer 사용 여부
    use_fast_tokenizer: bool = field(
        default=# TODO: True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )

@dataclass  
class DataTrainingArguments:
    """Seq2Seq QA 데이터 관련 인수들"""
    
    # TODO: 데이터셋 관련 설정
    dataset_name: Optional[str] = field(
        default=# TODO: None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    # TODO: 컬럼명 설정
    context_column: Optional[str] = field(
        default=# TODO: "context",
        metadata={"help": "The name of the column in the datasets containing the contexts."}
    )
    
    question_column: Optional[str] = field(
        default=# TODO: "question", 
        metadata={"help": "The name of the column in the datasets containing the questions."}
    )
    
    answer_column: Optional[str] = field(
        default=# TODO: "answers",
        metadata={"help": "The name of the column in the datasets containing the answers."}
    )
    
    # TODO: 학습/검증 파일 경로
    train_file: Optional[str] = field(
        default=# TODO: "data/KorQuAD_v1.0_train.json",
        metadata={"help": "The input training data file (a json file)."}
    )
    
    validation_file: Optional[str] = field(
        default=# TODO: "data/KorQuAD_v1.0_dev.json",
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a json file)."}
    )
    
    # TODO: Seq2Seq QA 특화 파라미터들
    max_seq_length: int = field(
        default=# TODO: 512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    
    max_answer_length: int = field(
        default=# TODO: 50,
        metadata={"help": "The maximum length of an answer that can be generated."}
    )
    
    val_max_answer_length: Optional[int] = field(
        default=# TODO: None,
        metadata={"help": "The maximum total sequence length for validation target text after tokenization."}
    )
    
    # TODO: 생성 관련 파라미터
    num_beams: Optional[int] = field(
        default=# TODO: None,
        metadata={"help": "Number of beams to use for evaluation."}
    )
    
    ignore_pad_token_for_loss: bool = field(
        default=# TODO: True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )

    def __post_init__(self):
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length

def main():
    # === 1. 인수 파싱 ===
    
    # TODO: HfArgumentParser를 사용하여 Seq2Seq 인수 파서 생성
    # 힌트: Seq2SeqTrainingArguments 사용 (TrainingArguments 아님!)
    parser = # TODO: 완성하세요
    
    # TODO: 명령행 인수 파싱
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

    # === 5. T5 모델과 토크나이저 로드 ===
    
    # TODO: T5 모델 설정 로드
    config = # TODO: AutoConfig.from_pretrained() 사용
    
    # TODO: T5 토크나이저 로드
    tokenizer = # TODO: AutoTokenizer.from_pretrained() 사용
    
    # TODO: T5 Seq2Seq 모델 로드
    # 힌트: AutoModelForSeq2SeqLM.from_pretrained() 사용
    model = # TODO: 완성하세요

    # TODO: 임베딩 크기 조정 (필요시)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        # TODO: 토큰 임베딩 크기 조정
        # TODO: 완성하세요

    # TODO: 디코더 시작 토큰 확인
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # === 6. 데이터 전처리 설정 ===
    
    # TODO: 컬럼명 확인 및 설정
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # TODO: 컬럼명 매핑
    question_column = data_args.question_column
    context_column = data_args.context_column  
    answer_column = data_args.answer_column

    # TODO: 최대 시퀀스 길이 조정
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    max_answer_length = data_args.max_answer_length
    padding = # TODO: "max_length" if pad_to_max_length else False

    # === 7. 전처리 함수 정의 ===
    
    def preprocess_function(examples):
        """학습용 Seq2Seq QA 전처리 함수"""
        # TODO: T5 입력 형식으로 변환
        inputs, targets = # TODO: preprocess_squad_batch() 호출
        
        # TODO: 입력 토크나이제이션
        model_inputs = tokenizer(
            # TODO: 필요한 인수들을 완성하세요
        )
        
        # TODO: 타겟 토크나이제이션
        labels = tokenizer(
            # TODO: text_target=targets 사용
        )

        # TODO: 패딩 토큰을 -100으로 변경
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                # TODO: 완성하세요
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_validation_function(examples):
        """검증용 Seq2Seq QA 전처리 함수"""
        # TODO: T5 입력 형식으로 변환
        inputs, targets = # TODO: preprocess_squad_batch() 호출

        # TODO: 입력 토크나이제이션 (overflow 토큰 포함)
        model_inputs = tokenizer(
            # TODO: return_overflowing_tokens=True 포함
        )
        
        # TODO: 타겟 토크나이제이션
        labels = tokenizer(
            # TODO: text_target=targets 사용
        )

        # TODO: 패딩 토큰 처리
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                # TODO: 완성하세요
            ]

        # TODO: 샘플 매핑과 example_id 처리
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        model_inputs["example_id"] = []
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
            labels_out.append(labels["input_ids"][sample_index])

        model_inputs["labels"] = labels_out
        return model_inputs

    # === 8. 학습/검증 데이터셋 전처리 ===
    
    if training_args.do_train:
        # TODO: 학습 데이터셋 전처리
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            # TODO: 필요한 인수들을 완성하세요
        )

    if training_args.do_eval:
        # TODO: 검증 데이터셋 전처리
        eval_examples = raw_datasets["validation"]
        eval_dataset = eval_examples.map(
            # TODO: 필요한 인수들을 완성하세요
        )

    # === 9. 데이터 콜레이터 설정 ===
    
    # TODO: Seq2Seq 전용 데이터 콜레이터 설정
    # 힌트: DataCollatorForSeq2Seq 사용
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = # TODO: 완성하세요

    # === 10. 평가 메트릭 설정 ===
    
    # TODO: SQuAD 메트릭 로드
    metric = # TODO: evaluate.load("squad") 사용

    def compute_metrics(p: EvalPrediction):
        # TODO: 메트릭 계산
        return # TODO: metric.compute() 호출

    # === 11. 후처리 함수 정의 ===
    
    def post_processing_function(examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"):
        """Seq2Seq QA 후처리 함수"""
        # TODO: 예측 결과 디코딩
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # TODO: -100을 패딩 토큰으로 변경
        preds = # TODO: np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        # TODO: 배치 디코딩
        decoded_preds = # TODO: tokenizer.batch_decode() 사용

        # TODO: 예측 결과 매핑 구성
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        
        # TODO: 각 예제별 예측 결과 매핑
        for example_index, example in enumerate(examples):
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        # TODO: 예측 결과 저장 (선택사항)
        output_path = os.path.join(training_args.output_dir, "eval_predictions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved evaluation predictions to {output_path}")

        # TODO: 평가 형식에 맞게 변환
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]
        
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # === 12. 커스텀 Seq2SeqTrainer 클래스 (간소화된 버전) ===
    
    from transformers import Seq2SeqTrainer
    
    class QuestionAnsweringSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_examples = eval_examples
            self.post_process_function = post_process_function

        def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval", **gen_kwargs):
            # TODO: 생성 파라미터 설정
            gen_kwargs = gen_kwargs.copy()
            if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
                gen_kwargs["max_length"] = self.args.generation_max_length
            gen_kwargs["num_beams"] = (
                gen_kwargs.get("num_beams") if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
            )
            self._gen_kwargs = gen_kwargs
            
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    # === 13. Trainer 초기화 ===
    
    # TODO: QuestionAnsweringSeq2SeqTrainer 초기화
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        # TODO: 필요한 인수들을 완성하세요
        # train_dataset, eval_dataset, tokenizer, data_collator,
        # compute_metrics, post_process_function, eval_examples
    )

    # === 14. 학습 실행 ===
    
    if training_args.do_train:
        print("=== Seq2Seq QA 학습 시작 ===")
        # TODO: 학습 실행
        train_result = # TODO: trainer.train() 호출
        
        # TODO: 모델 저장
        # TODO: trainer.save_model() 호출
        
        # TODO: 학습 메트릭 출력
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # === 15. 평가 실행 ===
    
    if training_args.do_eval:
        print("=== Seq2Seq QA 평가 시작 ===")
        # TODO: 평가 실행 (생성 파라미터 포함)
        # 힌트: max_length, num_beams 파라미터 전달
        max_length = training_args.generation_max_length or data_args.val_max_answer_length
        num_beams = data_args.num_beams or training_args.generation_num_beams
        
        metrics = # TODO: trainer.evaluate() 호출
        
        # TODO: 평가 메트릭 출력
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print(f"평가 결과: {metrics}")

if __name__ == "__main__":
    main()

"""
학습 목표:
1. T5 기반 Seq2Seq QA 모델의 학습 파이프라인 구축
2. Seq2SeqTrainer와 생성 기반 평가 방식 이해
3. Extractive QA와의 학습 과정 차이점 파악
4. 텍스트 생성 모델의 fine-tuning 특성 학습

핵심 개념:

1. Seq2Seq vs Extractive QA 학습:

   **Extractive QA**:
   - QuestionAnsweringTrainer 사용
   - 위치 예측 손실 (CrossEntropyLoss)
   - 복잡한 후처리 기반 평가
   - BERT 기반 인코더 모델
   
   **Seq2Seq QA**:
   - Seq2SeqTrainer 사용
   - 생성 손실 (Language Modeling Loss)
   - 텍스트 생성 기반 평가
   - T5 인코더-디코더 모델

2. T5 모델 특성:
   - Text-to-Text Transfer Transformer
   - 모든 태스크를 텍스트 생성으로 통일
   - Encoder-Decoder 구조
   - Prefix를 통한 태스크 구분

3. Seq2SeqTrainingArguments:
   - 일반 TrainingArguments 확장
   - 생성 관련 파라미터 추가:
     - generation_max_length: 생성 최대 길이
     - generation_num_beams: Beam Search 크기
     - predict_with_generate: 생성 기반 평가

4. DataCollatorForSeq2Seq:
   - 인코더와 디코더 입력 모두 처리
   - 동적 패딩 지원
   - 라벨 마스킹 (-100) 처리
   - T5 특화 데이터 처리

5. 생성 기반 평가:
   - model.generate() 사용
   - 실제 텍스트 생성 후 평가
   - BLEU, ROUGE, SQuAD 메트릭 활용
   - 느리지만 실제 성능 반영

6. 학습 과정 비교:

   **전처리**:
   - Extractive: 복잡한 위치 매핑 ⭐⭐⭐⭐⭐
   - Seq2Seq: 간단한 텍스트 변환 ⭐⭐

   **모델**:
   - Extractive: BERT + QA Head ⭐⭐⭐
   - Seq2Seq: T5 Encoder-Decoder ⭐⭐⭐⭐

   **손실 함수**:
   - Extractive: 위치 분류 손실 ⭐⭐⭐
   - Seq2Seq: 언어 모델링 손실 ⭐⭐

   **평가**:
   - Extractive: 후처리 기반 ⭐⭐⭐⭐⭐
   - Seq2Seq: 생성 기반 ⭐⭐⭐

   **후처리**:
   - Extractive: 매우 복잡 ⭐⭐⭐⭐⭐
   - Seq2Seq: 매우 간단 ⭐

7. 학습 파라미터:
   - learning_rate: 1e-4 ~ 5e-4 (T5는 높게)
   - max_seq_length: 512 (T5는 길게 가능)
   - max_answer_length: 50 (생성 길이)
   - num_beams: 4-8 (평가 시 beam search)

8. 메모리 고려사항:
   - 인코더-디코더로 메모리 사용량 높음
   - 생성 과정에서 추가 메모리 필요
   - Gradient Checkpointing 활용
   - 더 작은 배치 사이즈 필요

실무 장단점:

**장점**:
- 구현이 단순하고 직관적
- 다양한 형태의 답변 생성 가능
- 요약, 설명, 추론 등 복잡한 태스크 지원
- 새로운 정보 생성 가능

**단점**:
- 학습 시간이 오래 걸림
- 메모리 사용량이 많음
- Hallucination 위험성
- 사실성 보장 어려움

활용 전략:
- 교육: 설명형 QA 시스템
- 고객지원: 복합 문제 해결
- 연구: 논문 요약 및 해석
- 창작: 스토리텔링 지원
"""
