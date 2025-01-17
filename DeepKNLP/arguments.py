import logging
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import Strategy, DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing_extensions import Self

from chrisbase.data import OptionData, ResultData, CommonArguments, NewCommonArguments
from chrisbase.util import to_dataframe
from transformers import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)


class CustomDataArguments(BaseModel):
    pretrained: str | Path = Field(default=None)
    train_file: str | Path | None = Field(default=None)
    study_file: str | Path | None = Field(default=None)
    eval_file: str | Path | None = Field(default=None)
    pred_file: str | Path | None = Field(default=None)
    max_train_samples: int = Field(default=-1)
    max_study_samples: int = Field(default=-1)
    max_eval_samples: int = Field(default=-1)
    max_pred_samples: int = Field(default=-1)
    max_source_length: int = Field(default=512)
    max_target_length: int = Field(default=512)
    use_cache_data: bool = Field(default=True)
    ignore_pad_token_for_loss: bool = Field(default=True)

    @model_validator(mode='after')
    def after(self) -> Self:
        self.pretrained = Path(self.pretrained) if self.pretrained else None
        self.train_file = Path(self.train_file) if self.train_file else None
        self.study_file = Path(self.study_file) if self.study_file else None
        self.eval_file = Path(self.eval_file) if self.eval_file else None
        self.pred_file = Path(self.pred_file) if self.pred_file else None
        return self

    @property
    def cache_train_dir(self) -> Optional[Path]:
        if self.train_file:
            return self.train_file.parent / ".cache"

    @property
    def cache_study_dir(self) -> Optional[Path]:
        if self.study_file:
            return self.study_file.parent / ".cache"

    @property
    def cache_eval_dir(self) -> Optional[Path]:
        if self.eval_file:
            return self.eval_file.parent / ".cache"

    @property
    def cache_test_dir(self) -> Optional[Path]:
        if self.test_file:
            return self.test_file.parent / ".cache"

    def cache_train_path(self, size: int) -> Optional[Path]:
        if self.train_file:
            return self.cache_train_dir / f"{self.train_file.stem}={size}.tmp"

    def cache_study_path(self, size: int) -> Optional[Path]:
        if self.study_file:
            return self.cache_study_dir / f"{self.study_file.stem}={size}.tmp"

    def cache_eval_path(self, size: int) -> Optional[Path]:
        if self.eval_file:
            return self.cache_eval_dir / f"{self.eval_file.stem}={size}.tmp"

    def cache_test_path(self, size: int) -> Optional[Path]:
        if self.test_file:
            return self.cache_test_dir / f"{self.test_file.stem}={size}.tmp"


class TrainingArgumentsForAccelerator(NewCommonArguments):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: CustomDataArguments = Field(default=None)
    train: Seq2SeqTrainingArguments = Field(default=None)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.train, data_prefix="train"),
        ]).reset_index(drop=True)
        return df


class TrainingArgumentsForFabric(NewCommonArguments):
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
        train_file: str | Path | None = Field(default=None)
        study_file: str | Path | None = Field(default=None)
        eval_file: str | Path | None = Field(default=None)
        test_file: str | Path | None = Field(default=None)
        max_train_samples: int = Field(default=-1)
        max_study_samples: int = Field(default=-1)
        max_eval_samples: int = Field(default=-1)
        max_test_samples: int = Field(default=-1)
        max_source_length: int = Field(default=512)
        max_target_length: int = Field(default=512)
        max_generation_length: int = Field(default=1024)
        use_cache_data: bool = Field(default=True)

        @model_validator(mode='after')
        def after(self) -> Self:
            self.pretrained = Path(self.pretrained) if self.pretrained else None
            self.train_file = Path(self.train_file) if self.train_file else None
            self.study_file = Path(self.study_file) if self.study_file else None
            self.eval_file = Path(self.eval_file) if self.eval_file else None
            self.test_file = Path(self.test_file) if self.test_file else None
            return self

        @property
        def cache_train_dir(self) -> Optional[Path]:
            if self.train_file:
                return self.train_file.parent / ".cache"

        @property
        def cache_study_dir(self) -> Optional[Path]:
            if self.study_file:
                return self.study_file.parent / ".cache"

        @property
        def cache_eval_dir(self) -> Optional[Path]:
            if self.eval_file:
                return self.eval_file.parent / ".cache"

        @property
        def cache_test_dir(self) -> Optional[Path]:
            if self.test_file:
                return self.test_file.parent / ".cache"

        def cache_train_path(self, size: int) -> Optional[Path]:
            if self.train_file:
                return self.cache_train_dir / f"{self.train_file.stem}={size}.tmp"

        def cache_study_path(self, size: int) -> Optional[Path]:
            if self.study_file:
                return self.cache_study_dir / f"{self.study_file.stem}={size}.tmp"

        def cache_eval_path(self, size: int) -> Optional[Path]:
            if self.eval_file:
                return self.cache_eval_dir / f"{self.eval_file.stem}={size}.tmp"

        def cache_test_path(self, size: int) -> Optional[Path]:
            if self.test_file:
                return self.cache_test_dir / f"{self.test_file.stem}={size}.tmp"

    class LearnOption(BaseModel):
        output_home: str | Path | None = Field(default=None)
        output_name: str = Field(default="run")
        run_version: str | int | None = Field(default=None)
        num_train_epochs: int = Field(default=1)
        learning_rate: float = Field(default=5e-5)
        weight_decay: float = Field(default=0.0)
        train_batch: int = Field(default=1)
        infer_batch: int = Field(default=1)
        grad_steps: int = Field(default=1)
        eval_steps: int = Field(default=1)
        num_device: int = Field(default=1)
        device_idx: int = Field(default=0)
        device_type: str = Field(default="gpu")
        precision: str = Field(default="32")
        strategy: str = Field(default="ddp")
        ds_stage: int = Field(default=2)
        ds_offload: int = Field(default=0)
        fsdp_shard: Literal["FULL_SHARD", "SHARD_GRAD_OP"] = Field(default="FULL_SHARD")
        fsdp_offload: bool = Field(default=False)
        devices: int | List[int] = Field(default=1)

        @model_validator(mode='after')
        def after(self) -> Self:
            self.output_home = Path(self.output_home) if self.output_home else None
            self.devices = self.num_device
            if self.strategy == "ddp" and (self.device_type == "gpu" or self.device_type == "cuda") and self.device_idx >= 0:
                self.devices = list(range(self.device_idx, self.device_idx + self.num_device))
            self.grad_steps = max(1, self.grad_steps)
            return self

        @property
        def strategy_inst(self) -> Strategy | str:
            if self.strategy == "ddp":
                return DDPStrategy()
            elif self.strategy == "fsdp":
                fsdp_policy = {
                    nn.TransformerEncoderLayer,
                    nn.TransformerDecoderLayer,
                }
                return FSDPStrategy(
                    activation_checkpointing_policy=fsdp_policy,
                    auto_wrap_policy=fsdp_policy,
                    state_dict_type="full",
                    sharding_strategy=self.fsdp_shard,
                    cpu_offload=self.fsdp_offload,
                )
            elif self.strategy == "deepspeed":
                if self.ds_offload >= 3:
                    strategy = DeepSpeedStrategy(stage=self.ds_stage, offload_optimizer=True, offload_parameters=True)
                elif self.ds_offload >= 2:
                    strategy = DeepSpeedStrategy(stage=self.ds_stage, offload_optimizer=False, offload_parameters=True)
                elif self.ds_offload >= 1:
                    strategy = DeepSpeedStrategy(stage=self.ds_stage, offload_optimizer=True, offload_parameters=False)
                else:
                    strategy = DeepSpeedStrategy(stage=self.ds_stage, offload_optimizer=False, offload_parameters=False)
                strategy.config["zero_force_ds_cpu_optimizer"] = False
                return strategy

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
