# === Step 1: 기본 구조 이해 ===
# 수강생 과제: TODO 부분을 완성하여 NSMCModel 클래스의 기본 초기화를 구현하세요.

# === 라이브러리 Import ===
import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Dict, Mapping, Any

# PyTorch 관련
import torch
# Typer CLI 프레임워크
import typer
# ChrisBase 유틸리티 (프로젝트 내부 라이브러리)
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, make_dir, files
from chrisbase.util import mute_tqdm_cls, tupled
# Flask 웹 프레임워크
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
# Lightning 분산 학습 프레임워크
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
# PyTorch 옵티마이저, 데이터로더
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# Hugging Face Transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

# 프로젝트 내부 모듈
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
)  # 분류 데이터셋과 NSMC 코퍼스
from DeepKNLP.helper import (
    CheckpointSaver,
    epsilon,
    data_collator,
    fabric_barrier,
)  # 헬퍼 함수들
from DeepKNLP.metrics import accuracy  # 정확도 메트릭

# 로거 및 CLI 앱 초기화
logger = logging.getLogger(__name__)
main = AppTyper()


class NSMCModel(LightningModule):
    """
    NSMC(네이버 영화 리뷰) 감성분석을 위한 LightningModule 클래스

    주요 기능:
    - BERT 계열 사전학습 모델을 활용한 이진 분류
    - 체크포인트 저장/로드 기능
    - 학습/검증/테스트 데이터로더 제공
    - 단일 텍스트 추론 및 웹 서비스 API
    """

    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        """
        NSMCModel 초기화

        Args:
            args: 학습/테스트/서빙을 위한 설정 인수들
        """
        super().__init__()
        # 설정 저장
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        
        # TODO: NSMC 데이터 코퍼스 초기화
        # 힌트: NsmcCorpus 클래스를 사용하여 self.data를 초기화하세요
        self.data: NsmcCorpus = # TODO: 완성하세요

        # 라벨 수 검증 (이진 분류: 2개)
        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"

        # TODO: 사전학습 모델 설정 로드 (라벨 수 설정 포함)
        # 힌트: AutoConfig.from_pretrained를 사용하여 args.model.pretrained와 num_labels를 설정
        self.lm_config: PretrainedConfig = # TODO: 완성하세요

        # TODO: 토크나이저 로드 (빠른 토크나이저 사용)
        # 힌트: AutoTokenizer.from_pretrained를 사용하여 use_fast=True로 설정
        self.lm_tokenizer: PreTrainedTokenizer = # TODO: 완성하세요

        # TODO: 분류용 사전학습 모델 로드
        # 힌트: AutoModelForSequenceClassification.from_pretrained를 사용
        self.lang_model: PreTrainedModel = # TODO: 완성하세요

    # TODO: 다음 단계에서 완성할 메소드들
    # def configure_optimizers(self): pass
    # def train_dataloader(self): pass
    # def val_dataloader(self): pass
    # def test_dataloader(self): pass


# TODO: 나머지 함수들과 CLI 명령어들은 다음 단계에서 추가됩니다
if __name__ == "__main__":
    main()
