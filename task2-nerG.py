import datasets
from datasets import load_dataset, Dataset
from pydantic_core import ArgsKwargs
from typing_extensions import Self

from pydantic import model_validator
from pydantic.dataclasses import dataclass
import sys
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Tuple, Dict, Mapping, Any, ClassVar, Union
import pandas as pd

import torch
import typer
from pydantic import BaseModel, Field

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, TypedData, TimeChecker
from chrisbase.io import LoggingFormat, make_dir, files, hr, to_table_lines
from chrisbase.io import get_hostname, get_hostaddr, current_file, first_or, cwd, hr, flush_or, make_parent_dir, setup_unit_logger, setup_dual_logger, open_file, file_lines, to_table_lines, new_path, get_http_clients
from chrisbase.time import now
from chrisbase.util import mute_tqdm_cls, tupled, to_dataframe
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, CharSpan, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier
from DeepKNLP.metrics import accuracy, NER_Char_MacroF1, NER_Entity_MacroF1
from DeepKNLP.ner import NERCorpus, NERDataset, NEREncodedExample

main = AppTyper()
logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


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
    env: NewProjectEnv = Field(default_factory=NewProjectEnv)
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


class NewDataOption(BaseModel):
    train_path: str | Path = Field(default=None)
    eval_path: str | Path = Field(default=None)
    test_path: str | Path = Field(default=None)
    load_cache: bool = Field(default=True)


class NewModelOption(BaseModel):
    pretrained: str | Path = Field(default=None)


class NewLearningOption(BaseModel):
    random_seed: int = Field(default=None)


class NewHardwareOption(BaseModel):
    grad_acc_steps: int = Field(default=1)
    train_batch: int = Field(default=1)
    infer_batch: int = Field(default=1)
    accelerator: str = Field(default="cuda")
    precision: str = Field(default="32")
    strategy: str = Field(default="ddp")
    devices: List[int] = Field(default=[0])


class NewTrainerArguments(NewCommonArguments):
    data: NewDataOption = Field(default=None)
    model: NewModelOption = Field(default=None)
    learning: NewLearningOption = Field(default=None)
    hardware: NewHardwareOption = Field(default=None)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.model, data_prefix="model"),
            to_dataframe(columns=columns, raw=self.learning, data_prefix="learning"),
            to_dataframe(columns=columns, raw=self.hardware, data_prefix="hardware"),
        ]).reset_index(drop=True)
        return df


@main.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        logging_home: str = typer.Option(default="output/task2-nerG"),
        logging_file: str = typer.Option(default="train-messages.out"),
        argument_file: str = typer.Option(default="train-arguments.json"),
        max_workers: int = typer.Option(default=4),
        debugging: bool = typer.Option(default=True),
        # data
        train_data: str = typer.Option(default="data/gner/zero-shot-train.jsonl"),
        eval_data: str = typer.Option(default="data/gner/zero-shot-dev.jsonl"),
        test_data: str = typer.Option(default="data/gner/zero-shot-test.jsonl"),
        load_cache: bool = typer.Option(default=True),
        # model
        # pretrained: str = typer.Option(default="google/flan-t5-small"),
        # pretrained: str = typer.Option(default="meta-llama/Llama-2-7b-hf"),
        pretrained: str = typer.Option(default="meta-llama/Llama-3.2-1B"),
        # pretrained: str = typer.Option(default="etri-lirs/kebyt5-small-preview"),
        # pretrained: str = typer.Option(default="etri-lirs/egpt-1.3b-preview"),
        # learning
        random_seed: int = typer.Option(default=7),
        # hardware
        grad_acc_steps: int = typer.Option(default=4),
        train_batch: int = typer.Option(default=8),
        infer_batch: int = typer.Option(default=32),
        accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
        precision: str = typer.Option(default="bf16-mixed"),  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: str = typer.Option(default="ddp"),  # TODO: -> deepspeed
        device: List[int] = typer.Option(default=[0, 1]),  # TODO: -> [0], [0,1], [0,1,2,3]
):
    torch.set_float32_matmul_precision('high')
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.WARNING)
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    pretrained = Path(pretrained)
    args = NewTrainerArguments(
        env=NewProjectEnv(
            logging_home=logging_home,
            logging_file=logging_file,
            argument_file=argument_file,
            max_workers=1 if debugging else max(max_workers, 1),
            debugging=debugging,
            message_level=logging.INFO,
            message_format=LoggingFormat.CHECK_32,
        ),
        data=NewDataOption(
            train_path=train_data,
            eval_path=eval_data,
            test_path=test_data,
            load_cache=load_cache,
        ),
        model=NewModelOption(
            pretrained=pretrained,
        ),
        learning=NewLearningOption(
            random_seed=random_seed,
        ),
        hardware=NewHardwareOption(
            grad_acc_steps=grad_acc_steps,
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
    )
    fabric = Fabric(
        accelerator=args.hardware.accelerator,
        precision=args.hardware.precision,
        strategy=args.hardware.strategy,
        devices=args.hardware.devices,
    )

    def info_or_debug(m, *a, **k):
        if fabric.is_global_zero:  # or debugging:
            logger.info(m, *a, **k)
        else:
            logger.debug(m, *a, **k)

    fabric.print = info_or_debug
    fabric.launch()
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)
    args.env.setup_logger()
    # for checking
    # logger.info(f"[local_rank={fabric.local_rank}] args.env={args.env}({type(args.env)})")

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args if debugging and fabric.is_global_zero else None, verbose=fabric.is_global_zero,
            # mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.learning.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(pretrained)
        if config.is_encoder_decoder:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        else:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained)
        fabric.print(f"type(model)={type(model)} - {isinstance(model, PreTrainedModel)}")
        fabric.print(f"type(config)={type(config)} - {isinstance(config, PretrainedConfig)}")
        fabric.print(f"type(tokenizer)={type(tokenizer)} - {isinstance(tokenizer, PreTrainedTokenizerFast)}")
        fabric.print("-" * 100)
        fabric.barrier()

        # Load dataset
        train_dataset: Dataset | None = None
        eval_dataset: Dataset | None = None
        test_dataset: Dataset | None = None
        if args.data.train_path:
            with fabric.rank_zero_first():
                train_dataset: Dataset = load_dataset("json", data_files=args.data.train_path, split=datasets.Split.TRAIN)
                fabric.print(f"Use {args.data.train_path} as train dataset: {len(train_dataset):,} samples")
        if args.data.eval_path:
            with fabric.rank_zero_first():
                eval_dataset: Dataset = load_dataset("json", data_files=args.data.eval_path, split=datasets.Split.TRAIN)
                fabric.print(f"Use {args.data.eval_path} as eval dataset: {len(eval_dataset):,} samples")
        if args.data.test_path:
            with fabric.rank_zero_first():
                test_dataset: Dataset = load_dataset("json", data_files=args.data.test_path, split=datasets.Split.TRAIN)
                fabric.print(f"Use {args.data.eval_path} as test dataset: {len(test_dataset):,} samples")
        fabric.print("-" * 100)

        # Define preprocess function
        def preprocess_sample(sample: Mapping[str, Any]):
            # fabric.print(f"samples: {sample.keys()}")
            pass

        # Preprocess dataset
        if train_dataset:
            with fabric.rank_zero_first():
                fabric.print(f"train_dataset: type=({type(train_dataset)} - {isinstance(train_dataset, Dataset)})")
                train_dataset.map(
                    preprocess_sample,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.load_cache,
                    # desc="Tokenize train dataset",
                )
        if eval_dataset:
            with fabric.rank_zero_first():
                fabric.print(f"valid_dataset: type=({type(eval_dataset)} - {isinstance(eval_dataset, Dataset)})")
                eval_dataset.map(
                    preprocess_sample,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.load_cache,
                    desc="Tokenize eval dataset",
                )
        if test_dataset:
            with fabric.rank_zero_first():
                fabric.print(f"test_dataset: type=({type(test_dataset)} - {isinstance(test_dataset, Dataset)})")
                test_dataset.map(
                    preprocess_sample,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.load_cache,
                    desc="Tokenize test dataset",
                )
        fabric.print("-" * 100)
        fabric.barrier()


if __name__ == "__main__":
    main()
