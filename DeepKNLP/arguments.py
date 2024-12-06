import logging
import os
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import Strategy, DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from chrisbase.data import OptionData, TimeChecker, ResultData, CommonArguments
from chrisbase.io import get_hostname, get_hostaddr, current_file, make_parent_dir, setup_unit_logger, setup_dual_logger, to_table_lines, new_path
from chrisbase.time import now
from chrisbase.util import to_dataframe

logger = logging.getLogger(__name__)


class NewProjectEnv(BaseModel):
    hostname: str = get_hostname()
    hostaddr: str = get_hostaddr()
    global_rank: int = Field(default=-1)
    local_rank: int = Field(default=-1)
    node_rank: int = Field(default=-1)
    world_size: int = Field(default=-1)
    time_stamp: str = Field(default=now('%m%d.%H%M%S'))
    python_path: Path = Path(sys.executable).absolute()
    current_dir: Path = Path().absolute()
    current_file: Path = current_file().absolute()
    command_args: list[str] = sys.argv[1:]
    logging_home: str | Path = Field(default=None)
    logging_file: str | Path = Field(default=None)
    argument_file: str | Path = Field(default=None)
    random_seed: int = Field(default=None)
    max_workers: int = Field(default=1)
    debugging: bool = Field(default=False)
    date_format: str = Field(default="[%m.%d %H:%M:%S]")
    message_level: int = Field(default=logging.INFO)
    message_format: str = Field(default=logging.BASIC_FORMAT)

    @model_validator(mode='after')
    def after(self) -> Self:
        self.logging_home = Path(self.logging_home).absolute() if self.logging_home else None
        return self

    def setup_logger(self):
        if self.logging_home and self.logging_file:
            setup_dual_logger(
                level=self.message_level, fmt=self.message_format, datefmt=self.date_format, stream=sys.stdout,
                filename=self.logging_home / new_path(self.logging_file, post=self.time_stamp),
            )
        else:
            setup_unit_logger(
                level=self.message_level, fmt=self.message_format, datefmt=self.date_format, stream=sys.stdout,
            )
        return self


class NewCommonArguments(BaseModel):
    env: NewProjectEnv = Field(default=None)
    time: TimeChecker = Field(default_factory=TimeChecker)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
        ]).reset_index(drop=True)
        return df

    def info_args(self):
        for line in to_table_lines(self.dataframe()):
            logger.info(line)
        return self

    def save_args(self, to: Path | str = None) -> Path | None:
        if self.env.logging_home and self.env.argument_file:
            args_file = to if to else self.env.logging_home / new_path(self.env.argument_file, post=self.env.time_stamp)
            args_json = self.model_dump_json(indent=2)
            make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
            return args_file
        else:
            return None


class TrainingArguments(NewCommonArguments):
    input: "InputOption" = Field(default=None)
    learn: "LearnOption" = Field(default=None)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input"),
            to_dataframe(columns=columns, raw=self.learn, data_prefix="learn"),
        ]).reset_index(drop=True)
        return df

    class InputOption(BaseModel):
        pretrained: str | Path = Field(default=None)
        train_path: str | Path | None = Field(default=None)
        eval_path: str | Path | None = Field(default=None)
        test_path: str | Path | None = Field(default=None)
        max_train_samples: int = Field(default=-1)
        max_eval_samples: int = Field(default=-1)
        max_test_samples: int = Field(default=-1)
        num_prog_samples: int = Field(default=1)
        max_source_length: int = Field(default=512)
        max_target_length: int = Field(default=512)
        use_cache_data: bool = Field(default=True)

        @model_validator(mode='after')
        def after(self) -> Self:
            self.pretrained = Path(self.pretrained).absolute() if self.pretrained else None
            self.train_path = Path(self.train_path).absolute() if self.train_path else None
            self.eval_path = Path(self.eval_path).absolute() if self.eval_path else None
            self.test_path = Path(self.test_path).absolute() if self.test_path else None
            return self

        @property
        def cache_train_dir(self) -> Optional[Path]:
            if self.train_path:
                return self.train_path.parent / ".cache"

        @property
        def cache_eval_dir(self) -> Optional[Path]:
            if self.eval_path:
                return self.eval_path.parent / ".cache"

        @property
        def cache_test_dir(self) -> Optional[Path]:
            if self.test_path:
                return self.test_path.parent / ".cache"

        def cache_train_path(self, size: int) -> Optional[Path]:
            if self.train_path:
                return self.cache_train_dir / f"{self.train_path.stem}={size}.tmp"

        def cache_eval_path(self, size: int) -> Optional[Path]:
            if self.eval_path:
                return self.cache_eval_dir / f"{self.eval_path.stem}={size}.tmp"

        def cache_test_path(self, size: int) -> Optional[Path]:
            if self.test_path:
                return self.cache_test_dir / f"{self.test_path.stem}={size}.tmp"

    class LearnOption(BaseModel):
        output_home: str | Path | None = Field(default=None)
        num_train_epochs: int = Field(default=1)
        learning_rate: float = Field(default=5e-5)
        weight_decay: float = Field(default=0.0)
        accelerator: str = Field(default="gpu")
        precision: str = Field(default="32")
        gpu_index: int = Field(default=0)
        num_device: int = Field(default=1)
        grad_steps: int = Field(default=1)
        train_batch: int = Field(default=1)
        infer_batch: int = Field(default=1)
        strategy: str = Field(default="ddp")
        ds_stage: int = Field(default=2)
        devices: int | List[int] = Field(default=1)

        @model_validator(mode='after')
        def after(self) -> Self:
            self.output_home = Path(self.output_home).absolute() if self.output_home else None
            self.devices = self.num_device
            if self.strategy == "ddp" and (self.accelerator == "gpu" or self.accelerator == "cuda") and self.gpu_index >= 0:
                self.devices = list(range(self.gpu_index, self.gpu_index + self.num_device))
            return self

        @property
        def strategy_obj(self) -> Strategy | str:
            if self.strategy == "ddp":
                return DDPStrategy()
            elif self.strategy == "deepspeed":
                return DeepSpeedStrategy(stage=self.ds_stage)
            elif self.strategy == "fsdp":
                return FSDPStrategy()
            else:
                return self.strategy


@dataclass
class DataFiles(DataClassJsonMixin):
    train: str | Path | None = field(default=None)
    valid: str | Path | None = field(default=None)
    test: str | Path | None = field(default=None)


@dataclass
class DataOption(OptionData):
    name: str | Path = field()
    home: str | Path | None = field(default=None)
    files: DataFiles | None = field(default=None)
    caching: bool = field(default=False)
    redownload: bool = field(default=False)
    num_check: int = field(default=0)

    def __post_init__(self):
        if self.home:
            self.home = Path(self.home).absolute()


@dataclass
class ModelOption(OptionData):
    pretrained: str | Path = field()
    finetuning: str | Path = field()
    name: str | Path | None = field(default=None)
    seq_len: int = field(default=128)  # maximum total input sequence length after tokenization

    def __post_init__(self):
        self.finetuning = Path(self.finetuning).absolute()


@dataclass
class ServerOption(OptionData):
    port: int = field(default=7000)
    host: str = field(default="localhost")
    temp: str | Path = field(default="templates")
    page: str | Path = field(default=None)

    def __post_init__(self):
        self.temp = Path(self.temp)


@dataclass
class HardwareOption(OptionData):
    cpu_workers: int = field(default=os.cpu_count() / 2)
    train_batch: int = field(default=32)
    infer_batch: int = field(default=32)
    accelerator: str = field(default="auto")  # possible value: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto".
    precision: int | str = field(default="32-true")  # possible value: "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"
    strategy: str = field(default="auto")  # possbile value: "dp", "ddp", "ddp_spawn", "deepspeed", "fsdp".
    devices: List[int] | int | str = field(default="auto")  # devices to use

    def __post_init__(self):
        if not self.strategy:
            if self.devices == 1 or isinstance(self.devices, list) and len(self.devices) == 1:
                self.strategy = "single_device"
            elif isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
                self.strategy = "ddp"


@dataclass
class PrintingOption(OptionData):
    print_rate_on_training: float = field(default=1 / 10)
    print_rate_on_validate: float = field(default=1 / 10)
    print_rate_on_evaluate: float = field(default=1 / 10)
    print_step_on_training: int = field(default=-1)
    print_step_on_validate: int = field(default=-1)
    print_step_on_evaluate: int = field(default=-1)
    tag_format_on_training: str = field(default="")
    tag_format_on_validate: str = field(default="")
    tag_format_on_evaluate: str = field(default="")

    def __post_init__(self):
        self.print_rate_on_training = abs(self.print_rate_on_training)
        self.print_rate_on_validate = abs(self.print_rate_on_validate)
        self.print_rate_on_evaluate = abs(self.print_rate_on_evaluate)


@dataclass
class LearningOption(OptionData):
    random_seed: int | None = field(default=None)
    optimizer_cls: str = field(default="AdamW")
    learning_rate: float = field(default=5e-5)
    saving_mode: str = field(default="min val_loss")
    num_saving: int = field(default=3)
    num_epochs: int = field(default=1)
    log_text: bool = field(default=False)
    check_rate_on_training: float = field(default=1.0)
    name_format_on_saving: str = field(default="")

    def __post_init__(self):
        self.check_rate_on_training = abs(self.check_rate_on_training)


@dataclass
class ProgressChecker(ResultData):
    tb_logger: TensorBoardLogger = field(init=False, default=None)
    csv_logger: CSVLogger = field(init=False, default=None)
    world_size: int = field(init=False, default=1)
    local_rank: int = field(init=False, default=0)
    global_rank: int = field(init=False, default=0)
    global_step: int = field(init=False, default=0)
    global_epoch: float = field(init=False, default=0.0)


@dataclass
class MLArguments(CommonArguments):
    tag = None
    prog: ProgressChecker = field(default_factory=ProgressChecker)
    data: DataOption | None = field(default=None)
    model: ModelOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.prog, data_prefix="prog"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data") if self.data else None,
            to_dataframe(columns=columns, raw=self.model, data_prefix="model") if self.model else None,
        ]).reset_index(drop=True)
        return df


@dataclass
class ServerArguments(MLArguments):
    tag = "serve"
    server: ServerOption | None = field(default=None)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.server, data_prefix="server"),
        ]).reset_index(drop=True)
        return df


@dataclass
class TesterArguments(MLArguments):
    tag = "test"
    hardware: HardwareOption = field(default_factory=HardwareOption)
    printing: PrintingOption = field(default_factory=PrintingOption)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.hardware, data_prefix="hardware"),
            to_dataframe(columns=columns, raw=self.printing, data_prefix="printing"),
        ]).reset_index(drop=True)
        return df


@dataclass
class TrainerArguments(TesterArguments):
    tag = "train"
    learning: LearningOption = field(default_factory=LearningOption)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.learning, data_prefix="learning"),
        ]).reset_index(drop=True)
        return df
