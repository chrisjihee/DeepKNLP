# 기본 라이브러리 imports
import logging  # 로깅 시스템
import os  # 운영체제 인터페이스
from pathlib import Path  # 경로 조작
from time import sleep  # 대기 함수
from typing import List, Dict, Mapping, Any  # 타입 힌트

# PyTorch 관련 imports
import torch  # PyTorch 핵심 라이브러리
import typer  # CLI 명령어 인터페이스 생성
from chrisbase.data import AppTyper, JobTimer, ProjectEnv  # 프로젝트 환경 설정
from chrisbase.io import LoggingFormat, make_dir, files  # I/O 유틸리티
from chrisbase.util import mute_tqdm_cls, tupled  # 유틸리티 함수들
from flask import Flask, request, jsonify, render_template  # 웹 서버 프레임워크
from flask_classful import FlaskView, route  # Flask 클래스 기반 뷰

# Lightning 관련 imports (분산 학습 및 로깅)
from lightning import LightningModule  # PyTorch Lightning 모듈 베이스
from lightning.fabric import Fabric  # 경량화된 분산 학습 프레임워크
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger  # 로깅 시스템
from lightning.pytorch.utilities.types import OptimizerLRScheduler  # 옵티마이저 타입

# PyTorch 컴포넌트
from torch.optim import AdamW  # AdamW 옵티마이저
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler  # 데이터 로더

# Transformers 라이브러리 (사전학습 모델)
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
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        self.data: NsmcCorpus = NsmcCorpus(args)  # NSMC 데이터셋 로드

        # 라벨 수 검증 (이진 분류: 2개)
        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"

        # 사전학습 모델 설정 로드 (라벨 수 설정 포함)
        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained, num_labels=self.data.num_labels
        )

        # 토크나이저 로드 (빠른 토크나이저 사용)
        self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )

        # 분류용 사전학습 모델 로드
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
            "lang_model": self.lang_model.state_dict(),  # 모델 가중치
            "args_prog": self.args.prog,  # 학습 진행 상태 (step, epoch 등)
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        """
        체크포인트에서 모델 상태 복원

        Args:
            ckpt_state: 저장된 체크포인트 상태 딕셔너리
        """
        self.lang_model.load_state_dict(ckpt_state["lang_model"])  # 모델 가중치 로드
        self.args.prog = ckpt_state["args_prog"]  # 진행 상태 복원
        self.eval()  # 평가 모드로 설정

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
        self.load_checkpoint_file(checkpoint_files[-1])  # 가장 최근 파일 선택

    def configure_optimizers(self):
        """
        AdamW 옵티마이저 설정

        Returns:
            AdamW: 설정된 학습률을 가진 AdamW 옵티마이저
        """
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        """
        학습용 데이터로더 생성

        Returns:
            DataLoader: 학습용 데이터로더 (랜덤 샘플링)
        """
        # 분산 학습 시 로깅 설정 (rank 0에서만 info 레벨)
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # 학습 데이터셋 생성
        train_dataset = ClassificationDataset(
            "train", data=self.data, tokenizer=self.lm_tokenizer
        )

        # 학습용 데이터로더 생성 (랜덤 셔플)
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
        """
        검증용 데이터로더 생성

        Returns:
            DataLoader: 검증용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # 검증 데이터셋 생성
        val_dataset = ClassificationDataset(
            "valid", data=self.data, tokenizer=self.lm_tokenizer
        )

        # 검증용 데이터로더 생성 (순차 순서)
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
        """
        테스트용 데이터로더 생성

        Returns:
            DataLoader: 테스트용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # 테스트 데이터셋 생성
        test_dataset = ClassificationDataset(
            "test", data=self.data, tokenizer=self.lm_tokenizer
        )

        # 테스트용 데이터로더 생성 (순차 순서)
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
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)  # 모델 forward
        labels: torch.Tensor = inputs["labels"]  # 정답 라벨
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)  # 예측 결과 (argmax)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels)  # 정확도 계산
        return {
            "loss": outputs.loss,  # 손실값
            "acc": acc,  # 정확도
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
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        labels: List[int] = inputs["labels"].tolist()  # 라벨을 리스트로 변환
        preds: List[int] = outputs.logits.argmax(
            dim=-1
        ).tolist()  # 예측을 리스트로 변환
        return {
            "loss": outputs.loss,
            "preds": preds,  # 전체 검증을 위해 예측값들 수집
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
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        """
        단일 텍스트에 대한 감성분석 추론

        Args:
            text: 분석할 텍스트

        Returns:
            Dict: 예측 결과와 확률들을 포함한 딕셔너리
        """
        # 텍스트 토크나이즈
        inputs = self.lm_tokenizer(
            tupled(text),  # 단일 텍스트를 튜플로 변환
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 모델 추론
        outputs: SequenceClassifierOutput = self.lang_model(**inputs)
        prob = outputs.logits.softmax(dim=1)  # 확률로 변환

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

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Flask 웹 서버 실행

        Args:
            server: Flask 앱 인스턴스
            *args, **kwargs: Flask 서버 실행 옵션들
        """
        NSMCModel.WebAPI.register(route_base="/", app=server, init_argument=self)
        server.run(*args, **kwargs)

    class WebAPI(FlaskView):
        """
        Flask 기반 웹 API 클래스 - 감성분석 서비스 제공
        """

        def __init__(self, model: "NSMCModel"):
            """
            WebAPI 초기화

            Args:
                model: 학습된 NSMCModel 인스턴스
            """
            self.model = model

        @route("/")
        def index(self):
            """
            메인 페이지 렌더링

            Returns:
                HTML: 웹 인터페이스 페이지
            """
            return render_template(self.model.args.server.page)

        @route("/api", methods=["POST"])
        def api(self):
            """
            감성분석 API 엔드포인트

            POST /api
            Request Body: JSON 형태의 텍스트

            Returns:
                JSON: 분석 결과 (예측, 확률 등)
            """
            response = self.model.infer_one(text=request.json)
            return jsonify(response)


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
    # Fabric 초기화 및 설정
    fabric = model.fabric
    fabric.barrier()  # 분산 학습 동기화
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

    # 에포크별 학습 루프
    for epoch in range(model.args.learning.num_epochs):
        # 진행률 표시바 초기화
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(
            range(num_batch), unit=f"x{dataloader.batch_size}b", desc="training"
        )

        # 배치별 학습 루프
        for i, batch in enumerate(dataloader, start=1):
            model.train()  # 학습 모드 설정

            # 진행 상태 업데이트
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()  # 기울기 초기화
            outputs = model.training_step(batch, i)  # Forward pass
            fabric.backward(outputs["loss"])  # Backward pass
            optimizer.step()  # 가중치 업데이트

            progress.update()  # 진행률 업데이트
            fabric.barrier()  # 분산 학습 동기화

            # 메트릭 계산 및 로깅 (gradient 계산 없음)
            with torch.no_grad():
                model.eval()

                # 분산 환경에서 메트릭 수집 및 평균화
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
                    "loss": fabric.all_gather(outputs["loss"]).mean().item(),
                    "acc": fabric.all_gather(outputs["acc"]).mean().item(),
                }

                # 메트릭 로깅 (TensorBoard, CSV)
                fabric.log_dict(metrics=metrics, step=metrics["step"])

                # 주기적 출력
                if i % print_interval < 1:
                    fabric.print(
                        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                        f" | {model.args.printing.tag_format_on_training.format(**metrics)}"
                    )

                # 주기적 검증 및 체크포인트 저장
                if model.args.prog.global_step % check_interval < 1:
                    val_loop(model, val_dataloader, checkpoint_saver)

        fabric_barrier(fabric, "[after-epoch]", c="=")  # 에포크 종료 동기화

    fabric_barrier(fabric, "[after-train]")  # 전체 학습 종료 동기화


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
    # Fabric 설정
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

    # 배치별 검증 루프
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.validation_step(batch, i)  # 검증 단계 실행

        # 결과 수집
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])

        progress.update()

        # 주기적 출력
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")

    fabric.barrier()  # 분산 환경 동기화

    # 분산 환경에서 모든 예측 결과 수집
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()

    # 전체 검증 메트릭 계산
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
        "val_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "val_acc": accuracy(all_preds, all_labels).item(),
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
    # 체크포인트에서 모델 로드 (지정된 경우)
    if checkpoint_path:
        model.load_checkpoint_file(checkpoint_path)

    # Fabric 설정
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

    # 배치별 테스트 루프
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.test_step(batch, i)  # 테스트 단계 실행

        # 결과 수집
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])

        progress.update()

        # 주기적 출력
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")

    fabric.barrier()  # 분산 환경 동기화

    # 분산 환경에서 모든 예측 결과 수집
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()

    # 최종 테스트 메트릭 계산
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
        "test_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "test_acc": accuracy(all_preds, all_labels).item(),
    }

    # 최종 결과 로깅
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(
        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
        f" | {model.args.printing.tag_format_on_evaluate.format(**metrics)}"
    )

    fabric_barrier(fabric, "[after-test]")


@main.command()
def train(
    # === 실행 환경 설정 ===
    verbose: int = typer.Option(
        default=2, help="출력 상세도 (0: 최소, 1: 기본, 2: 상세)"
    ),
    # env - 프로젝트 환경 설정
    project: str = typer.Option(default="DeepKNLP", help="프로젝트 이름"),
    job_name: str = typer.Option(default=None, help="작업 이름 (기본값: 모델명)"),
    job_version: int = typer.Option(default=None, help="작업 버전 (기본값: 자동 할당)"),
    debugging: bool = typer.Option(default=False, help="디버깅 모드 활성화"),
    logging_file: str = typer.Option(default="logging.out", help="로그 파일명"),
    argument_file: str = typer.Option(
        default="arguments.json", help="인수 저장 파일명"
    ),
    # data
    data_home: str = typer.Option(default="data"),
    data_name: str = typer.Option(default="nsmc"),
    train_file: str = typer.Option(default="ratings_train.txt"),
    valid_file: str = typer.Option(default="ratings_valid.txt"),
    test_file: str = typer.Option(default=None),
    num_check: int = typer.Option(default=3),
    # model
    pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
    finetuning: str = typer.Option(default="output"),
    model_name: str = typer.Option(default=None),
    seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
    # hardware
    cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
    train_batch: int = typer.Option(default=50),
    infer_batch: int = typer.Option(default=50),
    accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
    precision: str = typer.Option(
        default="16-mixed"
    ),  # TODO: -> 32-true, bf16-mixed, 16-mixed
    strategy: str = typer.Option(default="ddp"),
    device: List[int] = typer.Option(default=[0]),  # TODO: -> [0], [0,1], [0,1,2,3]
    # printing
    print_rate_on_training: float = typer.Option(
        default=1 / 20
    ),  # TODO: -> 1/10, 1/20, 1/40, 1/100
    print_rate_on_validate: float = typer.Option(default=1 / 2),  # TODO: -> 1/2, 1/3
    print_rate_on_evaluate: float = typer.Option(default=1 / 2),  # TODO: -> 1/2, 1/3
    print_step_on_training: int = typer.Option(default=-1),
    print_step_on_validate: int = typer.Option(default=-1),
    print_step_on_evaluate: int = typer.Option(default=-1),
    tag_format_on_training: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, loss={loss:06.4f}, acc={acc:06.4f}"
    ),
    tag_format_on_validate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}"
    ),
    tag_format_on_evaluate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"
    ),
    # learning
    learning_rate: float = typer.Option(default=5e-5),
    random_seed: int = typer.Option(default=7),
    saving_mode: str = typer.Option(default="max val_acc"),
    num_saving: int = typer.Option(default=1),  # TODO: -> 2, 3
    num_epochs: int = typer.Option(default=1),  # TODO: -> 2, 3
    check_rate_on_training: float = typer.Option(default=1 / 5),  # TODO: -> 1/5, 1/10
    name_format_on_saving: str = typer.Option(
        default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}"
    ),
):
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TrainerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                train=train_file,
                valid=valid_file,
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_training=print_rate_on_training,
            print_rate_on_validate=print_rate_on_validate,
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_training=print_step_on_training,
            print_step_on_validate=print_step_on_validate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_training=tag_format_on_training,
            tag_format_on_validate=tag_format_on_validate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
        learning=LearningOption(
            learning_rate=learning_rate,
            random_seed=random_seed,
            saving_mode=saving_mode,
            num_saving=num_saving,
            num_epochs=num_epochs,
            check_rate_on_training=check_rate_on_training,
            name_format_on_saving=name_format_on_saving,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = (
        f"{args.tag}={args.env.job_name}={args.env.hostname}"
        if not args.model.name
        else args.model.name
    )
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.tb_logger = TensorBoardLogger(
        finetuning_home, output_name, args.env.job_version
    )  # tensorboard --logdir finetuning --bind_all
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    sleep(0.3)
    fabric = Fabric(
        loggers=[args.prog.tb_logger, args.prog.csv_logger],
        devices=(
            args.hardware.devices
            if args.hardware.accelerator in ["cuda", "gpu"]
            else (
                args.hardware.cpu_workers
                if args.hardware.accelerator == "cpu"
                else "auto"
            )
        ),
        strategy=(
            args.hardware.strategy
            if args.hardware.accelerator in ["cuda", "gpu"]
            else "auto"
        ),
        precision=(
            args.hardware.precision
            if args.hardware.accelerator in ["cuda", "gpu"]
            else None
        ),
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert (
        job_versions.min() == job_versions.max()
    ), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.seed_everything(args.learning.random_seed)
    fabric.barrier()

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
        verbose=verbose > 0 and fabric.local_rank == 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NSMCModel(args=args)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, optimizer)
        fabric_barrier(fabric, "[after-model]", c="=")

        assert args.data.files.train, "No training file found"
        train_dataloader = model.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        fabric_barrier(fabric, "[after-train_dataloader]", c="=")
        assert args.data.files.valid, "No validation file found"
        val_dataloader = model.val_dataloader()
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
        fabric_barrier(fabric, "[after-val_dataloader]", c="=")
        checkpoint_saver = CheckpointSaver(
            fabric=fabric,
            output_home=model.args.env.logging_home,
            name_format=model.args.learning.name_format_on_saving,
            saving_mode=model.args.learning.saving_mode,
            num_saving=model.args.learning.num_saving,
        )
        train_loop(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_saver=checkpoint_saver,
        )

        if args.data.files.test:
            test_dataloader = model.test_dataloader()
            test_dataloader = fabric.setup_dataloaders(test_dataloader)
            fabric_barrier(fabric, "[after-test_dataloader]", c="=")
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_saver.best_model_path,
            )


@main.command()
def test(
    verbose: int = typer.Option(default=2),
    # env
    project: str = typer.Option(default="DeepKNLP"),
    job_name: str = typer.Option(default=None),
    job_version: int = typer.Option(default=None),
    debugging: bool = typer.Option(default=False),
    logging_file: str = typer.Option(default="logging.out"),
    argument_file: str = typer.Option(default="arguments.json"),
    # data
    data_home: str = typer.Option(default="data"),
    data_name: str = typer.Option(default="nsmc"),
    test_file: str = typer.Option(default="ratings_test.txt"),
    num_check: int = typer.Option(default=3),
    # model
    pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
    finetuning: str = typer.Option(default="output"),
    model_name: str = typer.Option(default="train=*"),
    seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
    # hardware
    cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
    infer_batch: int = typer.Option(default=10),
    accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
    precision: str = typer.Option(
        default=None
    ),  # TODO: -> 32-true, bf16-mixed, 16-mixed
    strategy: str = typer.Option(default="auto"),
    device: List[int] = typer.Option(default=[0]),
    # printing
    print_rate_on_evaluate: float = typer.Option(
        default=1 / 10
    ),  # TODO: -> 1/2, 1/3, 1/5, 1/10, 1/50, 1/100
    print_step_on_evaluate: int = typer.Option(default=-1),
    tag_format_on_evaluate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"
    ),
):
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TesterArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    sleep(0.3)
    fabric = Fabric(
        devices=(
            args.hardware.devices
            if args.hardware.accelerator in ["cuda", "gpu"]
            else (
                args.hardware.cpu_workers
                if args.hardware.accelerator == "cpu"
                else "auto"
            )
        ),
        strategy=(
            args.hardware.strategy
            if args.hardware.accelerator in ["cuda", "gpu"]
            else "auto"
        ),
        precision=(
            args.hardware.precision
            if args.hardware.accelerator in ["cuda", "gpu"]
            else None
        ),
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert (
        job_versions.min() == job_versions.max()
    ), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.barrier()

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
        verbose=verbose > 0 and fabric.local_rank == 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NSMCModel(args=args)
        model = fabric.setup(model)
        fabric_barrier(fabric, "[after-model]", c="=")

        assert args.data.files.test, "No test file found"
        test_dataloader = model.test_dataloader()
        test_dataloader = fabric.setup_dataloaders(test_dataloader)
        fabric_barrier(fabric, "[after-test_dataloader]", c="=")

        for checkpoint_path in files(finetuning_home / args.model.name / "**/*.ckpt"):
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_path,
            )


@main.command()
def serve(
    verbose: int = typer.Option(default=2),
    # env
    project: str = typer.Option(default="DeepKNLP"),
    job_name: str = typer.Option(default=None),
    job_version: int = typer.Option(default=None),
    debugging: bool = typer.Option(default=False),
    logging_file: str = typer.Option(default="logging.out"),
    argument_file: str = typer.Option(default="arguments.json"),
    # data
    data_name: str = typer.Option(default="nsmc"),
    # model
    pretrained: str = typer.Option(default="beomi/KcELECTRA-base"),
    finetuning: str = typer.Option(default="output"),
    model_name: str = typer.Option(default="train=*"),
    seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
    # server
    server_port: int = typer.Option(default=9164),
    server_host: str = typer.Option(default="0.0.0.0"),
    server_temp: str = typer.Option(default="templates"),
    server_page: str = typer.Option(default="serve_cls.html"),
):
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = ServerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            name=data_name,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        server=ServerOption(
            port=server_port,
            host=server_host,
            temp=server_temp,
            page=server_page,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.print = logger.info
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) else None,
        verbose=verbose > 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NSMCModel(args=args)
        model = fabric.setup(model)
        model.load_last_checkpoint_file(finetuning_home / args.model.name / "**/*.ckpt")
        fabric_barrier(fabric, "[after-model]", c="=")

        model.run_server(
            server=Flask(output_name, template_folder=args.server.temp),
            host=args.server.host,
            port=args.server.port,
            debug=debugging,
        )


if __name__ == "__main__":
    main()
