# === Step 3: 학습/평가 스텝 ===
# 수강생 과제: TODO 부분을 완성하여 순전파/역전파 과정을 구현하세요.

# === 라이브러리 Import ===
import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Dict, Mapping, Any

import torch
import typer
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, make_dir, files
from chrisbase.util import mute_tqdm_cls, tupled
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from DeepKNLP.arguments import (
    DataFiles,
    DataOption,
    ModelOption,
    ServerOption,
    HardwareOption,
    PrintingOption,
    LearningOption,
)
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.cls import (
    ClassificationDataset,
    NsmcCorpus,
)
from DeepKNLP.helper import (
    CheckpointSaver,
    epsilon,
    data_collator,
    fabric_barrier,
)
from DeepKNLP.metrics import accuracy

logger = logging.getLogger(__name__)
main = AppTyper()


class NSMCModel(LightningModule):
    """NSMC(네이버 영화 리뷰) 감성분석을 위한 LightningModule 클래스"""

    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        """NSMCModel 초기화"""
        super().__init__()
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        self.data: NsmcCorpus = NsmcCorpus(args)

        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"
        
        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained, num_labels=self.data.num_labels
        )
        
        self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )
        
        self.lang_model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                args.model.pretrained,
                config=self.lm_config,
            )
        )

    def configure_optimizers(self):
        """AdamW 옵티마이저 설정"""
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        """학습용 데이터로더 생성"""
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        train_dataset = ClassificationDataset(
            "train", data=self.data, tokenizer=self.lm_tokenizer
        )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=False),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.train_batch,
            collate_fn=data_collator,
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

        val_dataset = ClassificationDataset(
            "valid", data=self.data, tokenizer=self.lm_tokenizer
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=data_collator,
            drop_last=False,
        )

        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(
            f"Created val_dataloader providing {len(val_dataloader)} batches"
        )
        return val_dataloader

    def test_dataloader(self):
        """테스트용 데이터로더 생성"""
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        test_dataset = ClassificationDataset(
            "test", data=self.data, tokenizer=self.lm_tokenizer
        )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=data_collator,
            drop_last=False,
        )

        self.fabric.print(
            f"Created test_dataset providing {len(test_dataset)} examples"
        )
        self.fabric.print(
            f"Created test_dataloader providing {len(test_dataloader)} batches"
        )
        return test_dataloader

    def training_step(self, inputs, batch_idx):
        """
        학습 단계에서 한 배치 처리

        Args:
            inputs: 토크나이즈된 입력 데이터 (input_ids, attention_mask, labels)
            batch_idx: 배치 인덱스

        Returns:
            Dict: loss와 accuracy를 포함한 딕셔너리
        """
        # TODO: 모델에 입력을 전달하여 출력을 얻으세요
        # 힌트: self.lang_model(**inputs)를 사용하여 SequenceClassifierOutput 획득
        outputs: SequenceClassifierOutput = # TODO: 완성하세요
        
        # TODO: 라벨과 예측값을 추출하세요
        # 힌트: inputs["labels"]와 outputs.logits.argmax(dim=-1) 사용
        labels: torch.Tensor = # TODO: 완성하세요
        preds: torch.Tensor = # TODO: 완성하세요
        
        # TODO: 정확도를 계산하세요
        # 힌트: accuracy(preds=preds, labels=labels) 함수 사용
        acc: torch.Tensor = # TODO: 완성하세요
        
        return {
            "loss": outputs.loss,  # 손실값
            "acc": acc,           # 정확도
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """
        검증 단계에서 한 배치 처리 (gradient 계산 없음)

        Args:
            inputs: 토크나이즈된 입력 데이터
            batch_idx: 배치 인덱스

        Returns:
            Dict: loss, 예측값들, 라벨들을 포함한 딕셔너리
        """
        # TODO: 모델에 입력을 전달하여 출력을 얻으세요
        outputs: SequenceClassifierOutput = # TODO: 완성하세요
        
        # TODO: 라벨과 예측값을 리스트로 변환하세요
        # 힌트: .tolist() 메소드 사용
        labels: List[int] = # TODO: 완성하세요
        preds: List[int] = # TODO: 완성하세요
        
        return {
            "loss": outputs.loss,
            "preds": preds,    # 전체 검증을 위해 예측값들 수집
            "labels": labels,  # 전체 검증을 위해 라벨들 수집
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
        단일 텍스트에 대한 감성분석 추론

        Args:
            text: 분석할 텍스트

        Returns:
            Dict: 예측 결과와 확률들을 포함한 딕셔너리
        """
        # TODO: 텍스트를 토크나이즈하세요
        # 힌트: self.lm_tokenizer를 사용하여 토크나이즈, return_tensors="pt" 설정
        inputs = # TODO: 완성하세요

        # TODO: 모델 추론을 수행하세요
        outputs: SequenceClassifierOutput = # TODO: 완성하세요
        
        # TODO: 확률로 변환하세요
        # 힌트: outputs.logits.softmax(dim=1) 사용
        prob = # TODO: 완성하세요

        # 예측 결과 해석
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
        positive_prob = round(prob[0][1].item(), 4)  # 긍정 확률
        negative_prob = round(prob[0][0].item(), 4)  # 부정 확률

        # 웹 UI용 결과 포맷
        return {
            "sentence": text,
            "prediction": pred,
            "positive_data": f"긍정 {positive_prob * 100:.1f}%",
            "negative_data": f"부정 {negative_prob * 100:.1f}%",
            "positive_width": f"{positive_prob * 100:.2f}%",  # 바 차트용
            "negative_width": f"{negative_prob * 100:.2f}%",  # 바 차트용
        }

    # TODO: 다음 단계에서 완성할 메소드들
    # def to_checkpoint(self): pass
    # def from_checkpoint(self): pass


if __name__ == "__main__":
    main()
