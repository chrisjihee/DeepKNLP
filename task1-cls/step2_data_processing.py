# === Step 2: 데이터 처리 ===
# 수강생 과제: TODO 부분을 완성하여 PyTorch 데이터 파이프라인을 구현하세요.

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
    """
    NSMC(네이버 영화 리뷰) 감성분석을 위한 LightningModule 클래스
    """

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
        """
        AdamW 옵티마이저 설정

        Returns:
            AdamW: 설정된 학습률을 가진 AdamW 옵티마이저
        """
        # TODO: AdamW 옵티마이저를 반환하세요
        # 힌트: self.lang_model.parameters()와 self.args.learning.learning_rate 사용
        return # TODO: 완성하세요

    def train_dataloader(self):
        """
        학습용 데이터로더 생성

        Returns:
            DataLoader: 학습용 데이터로더 (랜덤 샘플링)
        """
        # 분산 학습 시 로깅 설정 (rank 0에서만 info 레벨)
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: 학습 데이터셋 생성
        # 힌트: ClassificationDataset을 사용하여 "train" 데이터 생성
        train_dataset = # TODO: 완성하세요

        # TODO: 학습용 데이터로더 생성 (랜덤 셔플)
        # 힌트: RandomSampler, batch_size=self.args.hardware.train_batch 사용
        train_dataloader = DataLoader(
            # TODO: 필요한 인수들을 완성하세요
            # dataset, sampler, num_workers, batch_size, collate_fn, drop_last
        )

        # 정보 출력
        self.fabric.print(
            f"Created train_dataset providing {len(train_dataset)} examples"
        )
        self.fabric.print(
            f"Created train_dataloader providing {len(train_dataloader)} batches"
        )
        return train_dataloader

    def val_dataloader(self):
        """
        검증용 데이터로더 생성

        Returns:
            DataLoader: 검증용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: 검증 데이터셋 생성
        # 힌트: ClassificationDataset을 사용하여 "valid" 데이터 생성
        val_dataset = # TODO: 완성하세요

        # TODO: 검증용 데이터로더 생성 (순차 순서)
        # 힌트: SequentialSampler, batch_size=self.args.hardware.infer_batch 사용
        val_dataloader = DataLoader(
            # TODO: 필요한 인수들을 완성하세요
            # dataset, sampler, num_workers, batch_size, collate_fn, drop_last
        )

        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(
            f"Created val_dataloader providing {len(val_dataloader)} batches"
        )
        return val_dataloader

    def test_dataloader(self):
        """
        테스트용 데이터로더 생성

        Returns:
            DataLoader: 테스트용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: 테스트 데이터셋 생성
        # 힌트: ClassificationDataset을 사용하여 "test" 데이터 생성
        test_dataset = # TODO: 완성하세요

        # TODO: 테스트용 데이터로더 생성 (순차 순서)
        # 힌트: SequentialSampler, batch_size=self.args.hardware.infer_batch 사용  
        test_dataloader = DataLoader(
            # TODO: 필요한 인수들을 완성하세요
            # dataset, sampler, num_workers, batch_size, collate_fn, drop_last
        )

        self.fabric.print(
            f"Created test_dataset providing {len(test_dataset)} examples"
        )
        self.fabric.print(
            f"Created test_dataloader providing {len(test_dataloader)} batches"
        )
        return test_dataloader

    # TODO: 다음 단계에서 완성할 메소드들
    # def training_step(self, inputs, batch_idx): pass
    # def validation_step(self, inputs, batch_idx): pass
    # def test_step(self, inputs, batch_idx): pass


if __name__ == "__main__":
    main()
