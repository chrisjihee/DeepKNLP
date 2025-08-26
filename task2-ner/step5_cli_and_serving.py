# === Step 5: NER CLI와 서빙 ===
# 수강생 과제: TODO 부분을 완성하여 NER 시스템의 완전한 CLI와 웹 서비스를 구현하세요.

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
        """학습 단계에서 한 배치 처리"""
        inputs.pop("example_ids")
        outputs: TokenClassifierOutput = self.lang_model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """검증 단계에서 한 배치 처리 - 간소화된 버전"""
        example_ids: List[int] = inputs.pop("example_ids").tolist()
        outputs: TokenClassifierOutput = self.lang_model(**inputs)
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)

        # 간소화된 토큰-문자 매핑
        list_of_char_pred_ids: List[int] = []
        list_of_char_label_ids: List[int] = []
        
        for pred_batch, example_id in zip(preds.tolist(), example_ids):
            encoded_example: NEREncodedExample = self._infer_dataset[example_id]
            for pred_id, label_id in zip(pred_batch[:50], encoded_example.label_ids[:50]):
                if label_id != 0:
                    list_of_char_pred_ids.append(pred_id)
                    list_of_char_label_ids.append(label_id)

        return {
            "loss": outputs.loss,
            "preds": list_of_char_pred_ids,
            "labels": list_of_char_label_ids,
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        """테스트 단계에서 한 배치 처리"""
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        """단일 텍스트에 대한 NER 추론"""
        inputs = self.lm_tokenizer(
            tupled(text),
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs: TokenClassifierOutput = self.lang_model(**inputs)
        all_probs: Tensor = outputs.logits[0].softmax(dim=1)
        top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)

        tokens = self.lm_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        top_labels = [self.id_to_label(pred[0].item()) for pred in top_preds]

        result = []
        for token, label, top_prob in zip(tokens, top_labels, top_probs):
            if token in self.lm_tokenizer.all_special_tokens:
                continue
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

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Flask 웹 서버 실행

        Args:
            server: Flask 앱 인스턴스
            *args, **kwargs: Flask 서버 실행 옵션들
        """
        # TODO: WebAPI 클래스를 Flask 앱에 등록하세요
        # 힌트: NERModel.WebAPI.register() 사용
        # TODO: 완성하세요
        
        # TODO: 서버를 실행하세요
        # 힌트: server.run() 사용
        # TODO: 완성하세요

    class WebAPI(FlaskView):
        """
        Flask 기반 웹 API 클래스 - NER 서비스 제공
        """

        def __init__(self, model: "NERModel"):
            """
            WebAPI 초기화

            Args:
                model: 학습된 NERModel 인스턴스
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
            NER API 엔드포인트

            POST /api
            Request Body: JSON 형태의 텍스트

            Returns:
                JSON: 개체명 인식 결과 (토큰별 라벨과 확률)
            """
            # TODO: JSON 요청을 받아 NER 수행 후 결과를 반환하세요
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
    # data - 데이터 설정
    data_home: str = typer.Option(default="data", help="데이터 홈 디렉토리"),
    data_name: str = typer.Option(
        default="klue-ner", help="데이터셋 이름 (klue-ner, kmou-ner)"
    ),
    train_file: str = typer.Option(default="train.jsonl", help="학습 데이터 파일"),
    valid_file: str = typer.Option(default="valid.jsonl", help="검증 데이터 파일"),
    test_file: str = typer.Option(default=None, help="테스트 데이터 파일"),
    num_check: int = typer.Option(default=3, help="데이터 미리보기 개수"),
    # model - 모델 설정
    pretrained: str = typer.Option(default="klue/roberta-base", help="사전학습 모델명"),
    finetuning: str = typer.Option(default="output", help="Fine-tuning 출력 디렉토리"),
    model_name: str = typer.Option(default=None, help="모델 이름 (기본값: 자동 생성)"),
    seq_len: int = typer.Option(
        default=128, help="최대 시퀀스 길이 (64, 128, 256, 512)"
    ),
    # hardware - 하드웨어 설정
    cpu_workers: int = typer.Option(
        default=min(os.cpu_count() / 2, 10), help="CPU 워커 수"
    ),
    train_batch: int = typer.Option(default=50, help="학습 배치 크기"),
    infer_batch: int = typer.Option(default=50, help="추론 배치 크기"),
    accelerator: str = typer.Option(
        default="cuda", help="가속기 타입 (cuda, cpu, mps)"
    ),
    precision: str = typer.Option(
        default="16-mixed", help="정밀도 (32-true, bf16-mixed, 16-mixed)"
    ),
    strategy: str = typer.Option(default="ddp", help="분산 전략"),
    device: List[int] = typer.Option(default=[0], help="사용할 GPU 장치 번호들"),
    # printing - 출력 설정
    print_rate_on_training: float = typer.Option(
        default=1 / 20, help="학습 중 출력 주기 (비율)"
    ),
    print_rate_on_validate: float = typer.Option(
        default=1 / 2, help="검증 중 출력 주기 (비율)"
    ),
    print_rate_on_evaluate: float = typer.Option(
        default=1 / 2, help="평가 중 출력 주기 (비율)"
    ),
    print_step_on_training: int = typer.Option(
        default=-1, help="학습 중 출력 주기 (스텝)"
    ),
    print_step_on_validate: int = typer.Option(
        default=-1, help="검증 중 출력 주기 (스텝)"
    ),
    print_step_on_evaluate: int = typer.Option(
        default=-1, help="평가 중 출력 주기 (스텝)"
    ),
    tag_format_on_training: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, loss={loss:06.4f}, acc={acc:06.4f}",
        help="학습 로그 형식",
    ),
    tag_format_on_validate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}, val_F1c={val_F1c:05.2f}, val_F1e={val_F1e:05.2f}",
        help="검증 로그 형식 (NER F1 포함)",
    ),
    tag_format_on_evaluate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}, test_F1c={test_F1c:05.2f}, test_F1e={test_F1e:05.2f}",
        help="평가 로그 형식 (NER F1 포함)",
    ),
    # learning - 학습 설정
    learning_rate: float = typer.Option(default=5e-5, help="학습률"),
    random_seed: int = typer.Option(default=7, help="랜덤 시드"),
    saving_mode: str = typer.Option(
        default="max val_F1c", help="모델 저장 기준 (NER은 F1c 기준)"
    ),
    num_saving: int = typer.Option(default=1, help="저장할 모델 개수"),
    num_epochs: int = typer.Option(default=1, help="학습 에포크 수"),
    check_rate_on_training: float = typer.Option(
        default=1 / 5, help="학습 중 검증 주기 (비율)"
    ),
    name_format_on_saving: str = typer.Option(
        default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}",
        help="저장 파일명 형식 (NER F1 포함)",
    ),
):
    """
    NER 모델 학습 명령어

    주요 기능:
    - BERT 계열 모델을 NER 데이터셋으로 fine-tuning
    - BIO/BILOU 태깅 스킴 지원
    - 다양한 NER 메트릭 (토큰/문자/개체 레벨) 평가
    - 분산 학습 및 혼합 정밀도 지원
    - 자동 체크포인트 저장 (문자 레벨 F1 기준)
    """
    # TODO: PyTorch 설정을 초기화하세요
    # 힌트: torch.set_float32_matmul_precision("high"), TOKENIZERS_PARALLELISM 환경변수 설정
    # TODO: 완성하세요

    # TODO: 학습 인수를 구성하세요
    # 힌트: TrainerArguments 클래스 사용, 모든 CLI 인수들을 적절한 섹션에 매핑
    args = TrainerArguments(
        # TODO: 각 섹션별로 적절한 Option 클래스들로 매핑
        # env=ProjectEnv(...),
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
        model = NERModel(args=args)
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
    학습된 NER 모델 테스트 명령어
    
    기능:
    - 저장된 체크포인트를 로드하여 테스트 데이터 평가
    - NER 특화 메트릭 (토큰/문자/개체 레벨 F1) 계산
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
    학습된 NER 모델 웹 서비스 명령어
    
    기능:
    - Flask 웹 서버로 개체명 인식 API 제공
    - 웹 UI를 통한 인터랙티브 NER 테스트
    - REST API 엔드포인트 (/api) 제공
    - 토큰별 라벨과 확률 정보 제공
    """
    # TODO: serve 명령어 구현
    # 힌트: ServerArguments 사용, 모델 로드 후 run_server 호출
    pass


if __name__ == "__main__":
    main()
