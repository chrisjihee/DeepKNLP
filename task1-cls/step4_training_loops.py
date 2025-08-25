# === Step 4: 학습 루프 ===
# 수강생 과제: TODO 부분을 완성하여 전체 학습 과정을 구현하세요.

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

    def to_checkpoint(self) -> Dict[str, Any]:
        """
        체크포인트 저장을 위한 상태 딕셔너리 생성

        Returns:
            Dict: 모델 상태와 진행 정보를 포함한 딕셔너리
        """
        return {
            "lang_model": self.lang_model.state_dict(),
            "args_prog": self.args.prog,
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        """
        체크포인트에서 모델 상태 복원

        Args:
            ckpt_state: 저장된 체크포인트 상태 딕셔너리
        """
        self.lang_model.load_state_dict(ckpt_state["lang_model"])
        self.args.prog = ckpt_state["args_prog"]
        self.eval()

    def load_checkpoint_file(self, checkpoint_file):
        """
        체크포인트 파일에서 모델 로드

        Args:
            checkpoint_file: 체크포인트 파일 경로
        """
        assert Path(
            checkpoint_file
        ).exists(), f"Model file not found: {checkpoint_file}"
        self.fabric.print(f"Loading model from {checkpoint_file}")
        self.from_checkpoint(self.fabric.load(checkpoint_file))

    def load_last_checkpoint_file(self, checkpoints_glob):
        """
        glob 패턴으로 찾은 체크포인트 중 가장 최근 파일 로드

        Args:
            checkpoints_glob: 체크포인트 파일 찾기 패턴 (예: "output/**/*.ckpt")
        """
        checkpoint_files = files(checkpoints_glob)
        assert checkpoint_files, f"No model file found: {checkpoints_glob}"
        self.load_checkpoint_file(checkpoint_files[-1])

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
        """학습 단계에서 한 배치 처리"""
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """검증 단계에서 한 배치 처리"""
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        labels: List[int] = inputs["labels"].tolist()
        preds: List[int] = outputs.logits.argmax(dim=-1).tolist()
        return {
            "loss": outputs.loss,
            "preds": preds,
            "labels": labels,
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        """테스트 단계에서 한 배치 처리"""
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        """단일 텍스트에 대한 감성분석 추론"""
        inputs = self.lm_tokenizer(
            text,
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        prob = outputs.logits.softmax(dim=1)

        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)

        return {
            "sentence": text,
            "prediction": pred,
            "positive_data": f"긍정 {positive_prob * 100:.1f}%",
            "negative_data": f"부정 {negative_prob * 100:.1f}%",
            "positive_width": f"{positive_prob * 100:.2f}%",
            "negative_width": f"{negative_prob * 100:.2f}%",
        }


def train_loop(
    model: NSMCModel,
    optimizer: OptimizerLRScheduler,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    checkpoint_saver: CheckpointSaver | None = None,
):
    """
    모델 학습 루프 - 전체 에포크에 걸친 학습 진행

    Args:
        model: 학습할 NSMCModel
        optimizer: 옵티마이저
        dataloader: 학습 데이터로더
        val_dataloader: 검증 데이터로더
        checkpoint_saver: 체크포인트 저장 관리자 (선택사항)
    """
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug

    # 학습 스케줄링 계산
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_training * num_batch - epsilon
        if model.args.printing.print_step_on_training < 1
        else model.args.printing.print_step_on_training
    )
    check_interval = model.args.learning.check_rate_on_training * num_batch - epsilon

    # 학습 진행 상태 초기화
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0

    # TODO: 에포크별 학습 루프를 구현하세요
    for epoch in range(# TODO: 에포크 수 가져오기):
        # 진행률 표시바 초기화
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(
            range(num_batch), unit=f"x{dataloader.batch_size}b", desc="training"
        )

        # TODO: 배치별 학습 루프를 구현하세요
        for i, batch in enumerate(dataloader, start=1):
            # TODO: 모델을 학습 모드로 설정하세요
            # 힌트: model.train() 사용

            # 진행 상태 업데이트
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch

            # TODO: 역전파 및 가중치 업데이트를 구현하세요
            # 1. optimizer.zero_grad() - 기울기 초기화
            # 2. model.training_step(batch, i) - Forward pass
            # 3. fabric.backward(outputs["loss"]) - Backward pass  
            # 4. optimizer.step() - 가중치 업데이트

            progress.update()
            fabric.barrier()

            # 메트릭 계산 및 로깅 (gradient 계산 없음)
            with torch.no_grad():
                model.eval()

                # TODO: 분산 환경에서 메트릭 수집 및 평균화
                # 힌트: fabric.all_gather()를 사용하여 모든 프로세스의 값을 수집
                metrics: Mapping[str, Any] = {
                    "step": round(
                        fabric.all_gather(
                            torch.tensor(model.args.prog.global_step * 1.0)
                        )
                        .mean()
                        .item()
                    ),
                    "epoch": round(
                        fabric.all_gather(torch.tensor(model.args.prog.global_epoch))
                        .mean()
                        .item(),
                        4,
                    ),
                    # TODO: loss와 acc도 all_gather로 수집하여 평균내기
                    "loss": # TODO: 완성하세요
                    "acc": # TODO: 완성하세요
                }

                # 메트릭 로깅 (TensorBoard, CSV)
                fabric.log_dict(metrics=metrics, step=metrics["step"])

                # 주기적 출력
                if i % print_interval < 1:
                    fabric.print(
                        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                        f" | {model.args.printing.tag_format_on_training.format(**metrics)}"
                    )

                # TODO: 주기적 검증 및 체크포인트 저장
                # 힌트: model.args.prog.global_step % check_interval < 1 조건 확인
                if # TODO: 조건 완성:
                    # TODO: val_loop 호출
                    pass

        fabric_barrier(fabric, "[after-epoch]", c="=")

    fabric_barrier(fabric, "[after-train]")


@torch.no_grad()
def val_loop(
    model: NSMCModel,
    dataloader: DataLoader,
    checkpoint_saver: CheckpointSaver | None = None,
):
    """
    검증 루프 - 전체 검증 데이터에 대한 성능 평가

    Args:
        model: 평가할 NSMCModel
        dataloader: 검증 데이터로더
        checkpoint_saver: 체크포인트 저장 관리자 (선택사항)
    """
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug

    # 검증 스케줄링 계산
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_validate * num_batch - epsilon
        if model.args.printing.print_step_on_validate < 1
        else model.args.printing.print_step_on_validate
    )

    # 예측 결과 수집을 위한 리스트 초기화
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []

    # 진행률 표시바 초기화
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(
        range(num_batch), unit=f"x{dataloader.batch_size}b", desc="checking"
    )

    # TODO: 배치별 검증 루프를 구현하세요
    for i, batch in enumerate(dataloader, start=1):
        # TODO: 검증 단계 실행
        # 힌트: model.validation_step(batch, i) 호출
        outputs = # TODO: 완성하세요

        # TODO: 결과 수집
        # 힌트: preds.extend(), labels.extend(), losses.append() 사용
        # TODO: 완성하세요

        progress.update()

        # 주기적 출력
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")

    fabric.barrier()

    # TODO: 분산 환경에서 모든 예측 결과 수집
    # 힌트: fabric.all_gather(torch.tensor(preds)).flatten() 사용
    all_preds: torch.Tensor = # TODO: 완성하세요
    all_labels: torch.Tensor = # TODO: 완성하세요

    # TODO: 전체 검증 메트릭 계산
    metrics: Mapping[str, Any] = {
        "step": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0))
            .mean()
            .item()
        ),
        "epoch": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(),
            4,
        ),
        # TODO: val_loss와 val_acc 계산
        "val_loss": # TODO: 완성하세요
        "val_acc": # TODO: 완성하세요 (accuracy 함수 사용)
    }

    # 메트릭 로깅
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(
        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
        f" | {model.args.printing.tag_format_on_validate.format(**metrics)}"
    )

    fabric_barrier(fabric, "[after-check]")

    # 체크포인트 저장 (성능 기준으로)
    if checkpoint_saver:
        checkpoint_saver.save_checkpoint(
            metrics=metrics, ckpt_state=model.to_checkpoint()
        )


@torch.no_grad()
def test_loop(
    model: NSMCModel,
    dataloader: DataLoader,
    checkpoint_path: str | Path | None = None,
):
    """
    테스트 루프 - 최종 테스트 데이터에 대한 성능 평가

    Args:
        model: 평가할 NSMCModel
        dataloader: 테스트 데이터로더
        checkpoint_path: 로드할 체크포인트 경로 (선택사항)
    """
    # 체크포인트 로드 (제공된 경우)
    if checkpoint_path:
        model.load_checkpoint_file(checkpoint_path)

    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug

    # 테스트 스케줄링 계산
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_evaluate * num_batch - epsilon
        if model.args.printing.print_step_on_evaluate < 1
        else model.args.printing.print_step_on_evaluate
    )

    # 예측 결과 수집을 위한 리스트 초기화
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []

    # 진행률 표시바 초기화
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(
        range(num_batch), unit=f"x{dataloader.batch_size}b", desc="testing"
    )

    # TODO: test_loop의 나머지 부분을 val_loop과 유사하게 구현하세요
    # 차이점: test_step 사용, test_loss/test_acc 메트릭 이름


if __name__ == "__main__":
    main()
