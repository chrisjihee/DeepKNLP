# === Step 2: NER 데이터 처리 ===
# 수강생 과제: TODO 부분을 완성하여 NER 전용 데이터 파이프라인을 구현하세요.

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
            DataLoader: NER 학습용 데이터로더 (랜덤 샘플링)
        """
        # 분산 학습 시 로깅 설정
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: NER 학습 데이터셋 생성
        # 힌트: NERDataset을 사용하여 "train" 데이터 생성
        train_dataset = # TODO: 완성하세요

        # TODO: 학습용 데이터로더 생성 (랜덤 셔플, NER 전용 collate 함수)
        # 힌트: RandomSampler, self.data.encoded_examples_to_batch (NER 전용 collate_fn)
        train_dataloader = DataLoader(
            # TODO: 필요한 인수들을 완성하세요
            # dataset, sampler, num_workers, batch_size, collate_fn, drop_last
        )

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
            DataLoader: NER 검증용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: NER 검증 데이터셋 생성
        # 힌트: NERDataset을 사용하여 "valid" 데이터 생성
        val_dataset = # TODO: 완성하세요

        # TODO: 검증용 데이터로더 생성 (순차 순서)
        # 힌트: SequentialSampler, self.data.encoded_examples_to_batch 사용
        val_dataloader = DataLoader(
            # TODO: 필요한 인수들을 완성하세요
            # dataset, sampler, num_workers, batch_size, collate_fn, drop_last
        )

        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(
            f"Created val_dataloader providing {len(val_dataloader)} batches"
        )

        # TODO: validation_step에서 토큰-문자 매핑을 위해 데이터셋 저장
        # 힌트: self._infer_dataset에 val_dataset 할당
        # TODO: 완성하세요
        
        return val_dataloader

    def test_dataloader(self):
        """
        테스트용 데이터로더 생성

        Returns:
            DataLoader: NER 테스트용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # TODO: NER 테스트 데이터셋 생성
        # 힌트: NERDataset을 사용하여 "test" 데이터 생성
        test_dataset = # TODO: 완성하세요

        # TODO: 테스트용 데이터로더 생성 (순차 순서)
        # 힌트: SequentialSampler, self.data.encoded_examples_to_batch 사용
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

        # TODO: test_step에서 토큰-문자 매핑을 위해 데이터셋 저장
        # 힌트: self._infer_dataset에 test_dataset 할당
        # TODO: 완성하세요
        
        return test_dataloader

    # TODO: 다음 단계에서 완성할 메소드들
    # def training_step(self, inputs, batch_idx): pass
    # def validation_step(self, inputs, batch_idx): pass
    # def test_step(self, inputs, batch_idx): pass


if __name__ == "__main__":
    main()
