# === Step 3: NER 학습/평가 스텝 ===
# 수강생 과제: TODO 부분을 완성하여 NER 특화 학습/평가 로직을 구현하세요.
# 참고: validation_step은 매우 복잡하므로 핵심 부분만 구현합니다.

# === 라이브러리 Import ===
import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Tuple, Dict, Mapping, Any

import torch
import typer
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, make_dir, files, hr
from chrisbase.util import mute_tqdm_cls, tupled
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    CharSpan,
)
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from DeepKNLP.arguments import (
    DataFiles,
    DataOption,
    ModelOption,
    ServerOption,
    HardwareOption,
    PrintingOption,
    LearningOption,
)
from DeepKNLP.arguments import (
    TrainerArguments,
    TesterArguments,
    ServerArguments,
)
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier
from DeepKNLP.metrics import (
    accuracy,
    NER_Char_MacroF1,
    NER_Entity_MacroF1,
)
from DeepKNLP.ner import (
    NERCorpus,
    NERDataset,
    NEREncodedExample,
)

logger = logging.getLogger(__name__)
main = AppTyper()


class NERModel(LightningModule):
    """NER(Named Entity Recognition) 모델을 위한 LightningModule 클래스"""

    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        """NERModel 초기화"""
        super().__init__()
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        self.data: NERCorpus = NERCorpus(args)

        self.labels: List[str] = self.data.labels
        self._label_to_id: Dict[str, int] = {
            label: i for i, label in enumerate(self.labels)
        }
        self._id_to_label: Dict[int, str] = {
            i: label for i, label in enumerate(self.labels)
        }

        self._infer_dataset: NERDataset | None = None

        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"

        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=self.data.num_labels,
        )

        self.lm_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )
        assert isinstance(
            self.lm_tokenizer, PreTrainedTokenizerFast
        ), f"Our code support only PreTrainedTokenizerFast, not {type(self.lm_tokenizer)}"

        self.lang_model: PreTrainedModel = (
            AutoModelForTokenClassification.from_pretrained(
                args.model.pretrained,
                config=self.lm_config,
            )
        )

    @staticmethod
    def label_to_char_labels(label, num_char):
        """토큰 레벨 라벨을 문자 레벨 라벨 시퀀스로 변환"""
        for i in range(num_char):
            if i > 0 and ("-" in label):
                yield "I-" + label.split("-", maxsplit=1)[-1]
            else:
                yield label

    def label_to_id(self, x):
        """라벨 문자열을 ID로 변환"""
        return self._label_to_id[x]

    def id_to_label(self, x):
        """ID를 라벨 문자열로 변환"""
        return self._id_to_label[x]

    def to_checkpoint(self) -> Dict[str, Any]:
        """체크포인트 저장을 위한 상태 딕셔너리 생성"""
        return {
            "lang_model": self.lang_model.state_dict(),
            "args_prog": self.args.prog,
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        """체크포인트에서 모델 상태 복원"""
        self.lang_model.load_state_dict(ckpt_state["lang_model"])
        self.args.prog = ckpt_state["args_prog"]
        self.eval()

    def load_checkpoint_file(self, checkpoint_file):
        """체크포인트 파일에서 모델 로드"""
        assert Path(
            checkpoint_file
        ).exists(), f"Model file not found: {checkpoint_file}"
        self.fabric.print(f"Loading model from {checkpoint_file}")
        self.from_checkpoint(self.fabric.load(checkpoint_file))

    def load_last_checkpoint_file(self, checkpoints_glob):
        """glob 패턴으로 찾은 체크포인트 중 가장 최근 파일 로드"""
        checkpoint_files = files(checkpoints_glob)
        assert checkpoint_files, f"No model file found: {checkpoints_glob}"
        self.load_checkpoint_file(checkpoint_files[-1])

    def configure_optimizers(self):
        """AdamW 옵티마이저 설정"""
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        """학습용 데이터로더 생성"""
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        train_dataset = NERDataset("train", data=self.data, tokenizer=self.lm_tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=False),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.train_batch,
            collate_fn=self.data.encoded_examples_to_batch,
            drop_last=False,
        )

        self.fabric.print(
            f"Created train_dataset providing {len(train_dataset)} examples"
        )
        self.fabric.print(
            f"Created train_dataloader providing {len(train_dataloader)} batches"
        )
        return train_dataloader

    def val_dataloader(self):
        """검증용 데이터로더 생성"""
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        val_dataset = NERDataset("valid", data=self.data, tokenizer=self.lm_tokenizer)

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=self.data.encoded_examples_to_batch,
            drop_last=False,
        )

        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(
            f"Created val_dataloader providing {len(val_dataloader)} batches"
        )

        self._infer_dataset = val_dataset
        return val_dataloader

    def test_dataloader(self):
        """테스트용 데이터로더 생성"""
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        test_dataset = NERDataset("test", data=self.data, tokenizer=self.lm_tokenizer)

        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=self.data.encoded_examples_to_batch,
            drop_last=False,
        )

        self.fabric.print(
            f"Created test_dataset providing {len(test_dataset)} examples"
        )
        self.fabric.print(
            f"Created test_dataloader providing {len(test_dataloader)} batches"
        )

        self._infer_dataset = test_dataset
        return test_dataloader

    def training_step(self, inputs, batch_idx):
        """
        학습 단계에서 한 배치 처리

        Args:
            inputs: 토크나이즈된 입력 데이터 (input_ids, attention_mask, labels, example_ids)
            batch_idx: 배치 인덱스

        Returns:
            Dict: loss와 accuracy를 포함한 딕셔너리
        """
        # TODO: example_ids는 학습에 불필요하므로 제거
        # 힌트: inputs.pop("example_ids") 사용
        # TODO: 완성하세요

        # TODO: 토큰 분류 모델 순전파
        # 힌트: self.lang_model(**inputs)를 사용하여 TokenClassifierOutput 획득
        outputs: TokenClassifierOutput = # TODO: 완성하세요

        # TODO: 라벨과 예측값 추출
        # 힌트: inputs["labels"]와 outputs.logits.argmax(dim=-1) 사용
        labels: torch.Tensor = # TODO: 완성하세요
        preds: torch.Tensor = # TODO: 완성하세요

        # TODO: 정확도 계산 (패딩 토큰 ignore_index=0 제외)
        # 힌트: accuracy(preds=preds, labels=labels, ignore_index=0) 함수 사용
        acc: torch.Tensor = # TODO: 완성하세요

        return {
            "loss": outputs.loss,  # 토큰 분류 손실
            "acc": acc,  # 토큰 레벨 정확도
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """
        검증 단계에서 한 배치 처리 - NER 특화 복잡한 토큰-문자 매핑 수행

        NER의 핵심: 토큰 레벨 예측을 문자 레벨로 변환하여 정확한 평가
        - 토큰 경계와 문자 경계가 다름 (서브워드 토크나이제이션)
        - BIO 태깅 규칙에 따른 라벨 변환
        - 문자 단위 정확한 평가를 위한 오프셋 매핑

        이 메소드는 매우 복잡하므로 핵심 부분만 구현합니다.
        """
        # TODO: 예제 ID 추출 (토큰-문자 매핑을 위해 필요)
        # 힌트: inputs.pop("example_ids").tolist() 사용
        example_ids: List[int] = # TODO: 완성하세요

        # TODO: 토큰 분류 모델 순전파
        # 힌트: self.lang_model(**inputs) 사용
        outputs: TokenClassifierOutput = # TODO: 완성하세요
        
        # TODO: 예측값 추출
        # 힌트: outputs.logits.argmax(dim=-1) 사용
        preds: torch.Tensor = # TODO: 완성하세요

        # 복잡한 토큰-문자 매핑 과정 (간소화된 버전)
        # 실제 구현에서는 매우 복잡한 오프셋 계산과 BIO 태깅 변환이 필요
        list_of_char_pred_ids: List[int] = []
        list_of_char_label_ids: List[int] = []
        
        # 간단한 더미 구현 (실제로는 복잡한 토큰-문자 매핑 필요)
        for pred_batch, example_id in zip(preds.tolist(), example_ids):
            # 실제 구현에서는 encoded_example을 통한 복잡한 매핑 과정
            encoded_example: NEREncodedExample = self._infer_dataset[example_id]
            
            # 간소화: 패딩 제외하고 처리
            for pred_id, label_id in zip(pred_batch[:50], encoded_example.label_ids[:50]):  # 임시 간소화
                if label_id != 0:  # 패딩이 아닌 경우만
                    list_of_char_pred_ids.append(pred_id)
                    list_of_char_label_ids.append(label_id)

        return {
            "loss": outputs.loss,
            "preds": list_of_char_pred_ids,  # 문자 레벨 예측값들
            "labels": list_of_char_label_ids,  # 문자 레벨 라벨들
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        """
        테스트 단계에서 한 배치 처리 (검증과 동일)

        Args:
            inputs: 토크나이즈된 입력 데이터
            batch_idx: 배치 인덱스

        Returns:
            Dict: validation_step과 동일한 출력
        """
        # TODO: validation_step과 동일한 처리를 하세요
        # 힌트: self.validation_step(inputs, batch_idx)를 호출하여 반환
        return # TODO: 완성하세요

    @torch.no_grad()
    def infer_one(self, text: str):
        """
        단일 텍스트에 대한 NER 추론

        Args:
            text: 개체명 인식을 수행할 텍스트

        Returns:
            Dict: 토큰별 개체명 라벨과 확률을 포함한 결과
        """
        # TODO: 텍스트를 튜플로 감싸서 토크나이즈 (batch dimension)
        # 힌트: self.lm_tokenizer를 사용하여 토크나이즈, return_tensors="pt" 설정
        inputs = # TODO: 완성하세요

        # TODO: 토큰 분류 모델 추론
        outputs: TokenClassifierOutput = # TODO: 완성하세요

        # TODO: 각 토큰에 대한 라벨 확률 계산
        # 힌트: outputs.logits[0].softmax(dim=1), torch.topk 사용
        all_probs: Tensor = # TODO: 완성하세요
        top_probs, top_preds = # TODO: 완성하세요

        # 토큰과 라벨 정보 추출
        tokens = self.lm_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        top_labels = [self.id_to_label(pred[0].item()) for pred in top_preds]

        # 특수 토큰 제외하고 결과 구성
        result = []
        for token, label, top_prob in zip(tokens, top_labels, top_probs):
            if token in self.lm_tokenizer.all_special_tokens:
                continue  # [CLS], [SEP], [PAD] 등 제외
            result.append(
                {
                    "token": token,
                    "label": label,
                    "prob": f"{round(top_prob[0].item(), 4):.4f}",
                }
            )

        return {
            "sentence": text,
            "result": result,
        }

    # TODO: 다음 단계에서 완성할 메소드들
    # def run_server(self, server: Flask, *args, **kwargs): pass
    # class WebAPI(FlaskView): pass


if __name__ == "__main__":
    main()
