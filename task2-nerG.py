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

logger = logging.getLogger(__name__)
main = AppTyper()


class NewLearningOption(BaseModel):
    random_seed: int | None = Field(default=None)


class NewTrainerArguments(BaseModel):
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
        # learning
        random_seed: int = typer.Option(default=7),
):
    torch.set_float32_matmul_precision('high')
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.WARNING)
    # logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)

    args = NewTrainerArguments(
        learning=NewLearningOption(
            random_seed=random_seed,
        ),
    )

    fabric = Fabric()
    fabric.launch()
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
