import random
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from DeepKNLP.gner_trainer import GNERTrainer
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics

from chrisbase.data import AppTyper, JobTimer, Counter, NewProjectEnv
from chrisbase.io import LoggingFormat, set_verbosity_warning, set_verbosity_info, set_verbosity_error, to_table_lines, do_nothing
from typing_extensions import Annotated
import typer
import torch

# Global settings
env = None
app = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer.")
logger = logging.getLogger(__name__)


@app.callback()
def begin(
        # env
        logging_home: Annotated[str, typer.Option("--logging_home")] = "output",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "train-messages.out",
        argument_file: Annotated[str, typer.Option("--argument_file")] = "train-arguments.json",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = 4,
        debugging: Annotated[bool, typer.Option("--debugging")] = False,
):
    global env
    env = NewProjectEnv(
        logging_home=logging_home,
        logging_file=logging_file,
        logging_level="info",
        logging_format=LoggingFormat.TRACE_28,
        argument_file=argument_file,
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
    )
    torch.set_float32_matmul_precision('high')


@app.command()
def train(
        # input
        pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-small",  # (80M)
):
    pass


if __name__ == "__main__":
    app()
