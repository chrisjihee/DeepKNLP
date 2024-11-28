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
from typing import List, Tuple, Dict, Mapping, Any, ClassVar
import pandas as pd

import torch
import typer
from pydantic import BaseModel, Field

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, TypedData
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
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, CharSpan
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
    node_rank: int = Field(default=-1)
    local_rank: int = Field(default=-1)
    global_rank: int = Field(default=-1)
    time_stamp: str = Field(default=now('%m%d.%H%M%S'))
    python_path: Path = Path(sys.executable).absolute()
    current_dir: Path = Path().absolute()
    current_file: Path = current_file().absolute()
    command_args: list[str] = sys.argv[1:]
    logging_home: str | Path = Field(default=None)
    logging_file: str | Path = Field(default=None)
    argument_file: str | Path = Field(default=None)
    date_format: str = Field(default="[%m.%d %H:%M:%S]")
    message_level: int = Field(default=logging.INFO)
    message_format: str = Field(default=logging.BASIC_FORMAT)

    @model_validator(mode='after')
    def after(self) -> Self:
        self.logging_home = Path(self.logging_home).absolute() if self.logging_home else None
        self._setup_logger()
        return self

    def _setup_logger(self):
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


class NewLearningOption(BaseModel):
    random_seed: int | None = Field(default=None)


class NewTrainerArguments(BaseModel):
    env: NewProjectEnv = Field(default_factory=NewProjectEnv)
    learning: NewLearningOption = Field(default_factory=NewLearningOption)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.__class__.__name__, "value"]
        df = pd.concat([
            to_dataframe(columns=columns, raw=self.learning, data_prefix="learning"),
        ]).reset_index(drop=True)
        return df

    def log_table(self):
        for line in to_table_lines(self.dataframe()):
            logger.info(line)
        return self


@main.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        logging_home: str = typer.Option(default="output/train"),
        logging_file: str = typer.Option(default="logging.out"),
        argument_file: str = typer.Option(default="arguments.json"),
        # model
        pretrained: str = typer.Option(default="jinmang2/kpfbert"),  # TODO: -> "klue/roberta-base"
        # learning
        random_seed: int = typer.Option(default=7),
):
    torch.set_float32_matmul_precision('high')
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.WARNING)
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)

    fabric = Fabric()
    time_stamp: str = now('%m%d.%H%M%S') if fabric.is_global_zero else None
    fabric.launch()
    fabric.barrier()

    pretrained = Path(pretrained)
    args = NewTrainerArguments(
        env=NewProjectEnv(
            time_stamp=fabric.broadcast(time_stamp, src=0),
            node_rank=fabric.node_rank,
            local_rank=fabric.local_rank,
            global_rank=fabric.global_rank,
            logging_home=logging_home,
            logging_file=logging_file,
            argument_file=argument_file,
            message_level=logging.INFO,
            message_format=LoggingFormat.CHECK_40,
        ),
        learning=NewLearningOption(
            random_seed=random_seed,
        ),
    )
    fabric.barrier()
    logger.info(f"[local_rank={fabric.local_rank}] args.env={args.env}({type(args.env)})")
    fabric.barrier()
    fabric.seed_everything(args.learning.random_seed)
    fabric.barrier()
    if fabric.local_rank == 0:
        logger.info("=" * 100)
        args.log_table()
        logger.info("=" * 100)
        logger.info(args)
        logger.info("=" * 100)
    fabric.barrier()


if __name__ == "__main__":
    main()
