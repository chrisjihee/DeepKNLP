# === Step 5: CLI와 서빙 ===
# 수강생 과제: TODO 부분을 완성하여 실제 사용 가능한 시스템을 구현하세요.

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

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Flask 웹 서버 실행

        Args:
            server: Flask 앱 인스턴스
            *args, **kwargs: Flask 서버 실행 옵션들
        """
        # TODO: WebAPI 클래스를 Flask 앱에 등록하세요
        # 힌트: NSMCModel.WebAPI.register() 사용
        # TODO: 완성하세요
        
        # TODO: 서버를 실행하세요
        # 힌트: server.run() 사용
        # TODO: 완성하세요

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
            # TODO: 템플릿을 렌더링하여 반환하세요
            # 힌트: render_template() 사용, self.model.args.server.page
            return # TODO: 완성하세요

        @route("/api", methods=["POST"])
        def api(self):
            """
            감성분석 API 엔드포인트

            POST /api
            Request Body: JSON 형태의 텍스트

            Returns:
                JSON: 분석 결과 (예측, 확률 등)
            """
            # TODO: JSON 요청을 받아 감성분석 후 결과를 반환하세요
            # 힌트: request.json으로 텍스트 받기, self.model.infer_one() 호출, jsonify() 반환
            response = # TODO: 완성하세요
            return # TODO: 완성하세요


# 학습 루프들은 이전 단계에서 완성되었다고 가정
def train_loop(model, optimizer, dataloader, val_dataloader, checkpoint_saver=None):
    """학습 루프 - 이미 구현됨"""
    pass  # 실제로는 Step 4에서 구현한 내용


@torch.no_grad()
def val_loop(model, dataloader, checkpoint_saver=None):
    """검증 루프 - 이미 구현됨"""
    pass  # 실제로는 Step 4에서 구현한 내용


@torch.no_grad()
def test_loop(model, dataloader, checkpoint_path=None):
    """테스트 루프 - 이미 구현됨"""
    pass  # 실제로는 Step 4에서 구현한 내용


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
    seq_len: int = typer.Option(default=128),
    # hardware
    cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
    train_batch: int = typer.Option(default=50),
    infer_batch: int = typer.Option(default=50),
    accelerator: str = typer.Option(default="cuda"),
    precision: str = typer.Option(default="16-mixed"),
    strategy: str = typer.Option(default="ddp"),
    device: List[int] = typer.Option(default=[0]),
    # printing
    print_rate_on_training: float = typer.Option(default=1 / 20),
    print_rate_on_validate: float = typer.Option(default=1 / 2),
    print_rate_on_evaluate: float = typer.Option(default=1 / 2),
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
    num_saving: int = typer.Option(default=1),
    num_epochs: int = typer.Option(default=1),
    check_rate_on_training: float = typer.Option(default=1 / 5),
    name_format_on_saving: str = typer.Option(
        default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}"
    ),
):
    """
    NSMC 감성분석 모델 학습 명령어
    
    주요 기능:
    - BERT 계열 모델을 NSMC 데이터셋으로 fine-tuning
    - 분산 학습 지원 (Lightning Fabric)
    - 자동 체크포인트 저장 및 검증
    - TensorBoard, CSV 로깅
    """
    # TODO: PyTorch 설정을 초기화하세요
    # 힌트: torch.set_float32_matmul_precision("high"), TOKENIZERS_PARALLELISM 환경변수 설정
    # TODO: 완성하세요

    # TODO: 학습 인수를 구성하세요  
    # 힌트: TrainerArguments 클래스 사용, 모든 CLI 인수들을 적절한 섹션에 매핑
    args = TrainerArguments(
        # TODO: 각 섹션별로 적절한 Option 클래스들로 매핑
        # env=EnvOption(...),
        # data=DataOption(...),
        # model=ModelOption(...),
        # hardware=HardwareOption(...),
        # printing=PrintingOption(...),
        # learning=LearningOption(...)
    )

    # TODO: 출력 디렉토리와 로거 설정
    finetuning_home = Path(f"{finetuning}/{data_name}")
    # TODO: 출력 이름, 버전, 로거들 설정

    # TODO: Fabric 초기화 및 실행
    # 힌트: Fabric(loggers, devices, strategy, precision, accelerator) 설정
    fabric = # TODO: 완성하세요

    # TODO: JobTimer 컨텍스트에서 전체 학습 과정 실행
    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        # ... 기타 설정들
    ):
        # TODO: 모델 생성 및 설정
        model = NSMCModel(args=args)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, optimizer)

        # TODO: 데이터로더 생성 및 설정  
        train_dataloader = model.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        
        val_dataloader = model.val_dataloader()
        val_dataloader = fabric.setup_dataloaders(val_dataloader)

        # TODO: 체크포인트 저장소 생성
        checkpoint_saver = CheckpointSaver(
            # ... 필요한 설정들
        )

        # TODO: 학습 실행
        train_loop(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_saver=checkpoint_saver,
        )

        # TODO: 테스트 실행 (테스트 파일이 있는 경우)
        if args.data.files.test:
            test_dataloader = model.test_dataloader()
            test_dataloader = fabric.setup_dataloaders(test_dataloader)
            test_loop(model=model, dataloader=test_dataloader)


@main.command()
def test(
    # TODO: test 명령어의 인수들을 정의하세요
    # 힌트: train과 유사하지만 테스트에 필요한 인수들만 포함
):
    """
    학습된 NSMC 모델 테스트 명령어
    
    기능:
    - 저장된 체크포인트를 로드하여 테스트 데이터 평가
    - 여러 체크포인트 파일에 대해 순차적으로 테스트 가능
    """
    # TODO: test 명령어 구현
    # 힌트: train과 유사하지만 TesterArguments 사용, test_loop만 실행
    pass


@main.command()
def serve(
    # TODO: serve 명령어의 인수들을 정의하세요
    # 힌트: 서버 관련 설정들 (host, port, template 등) 포함
):
    """
    학습된 NSMC 모델 웹 서비스 명령어
    
    기능:
    - Flask 웹 서버로 감성분석 API 제공
    - 웹 UI를 통한 인터랙티브 테스트
    - REST API 엔드포인트 (/api) 제공
    """
    # TODO: serve 명령어 구현
    # 힌트: ServerArguments 사용, 모델 로드 후 run_server 호출
    pass


if __name__ == "__main__":
    main()
