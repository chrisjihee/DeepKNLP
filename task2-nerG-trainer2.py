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

from lightning.fabric.loggers import CSVLogger

from chrisbase.data import AppTyper, JobTimer, Counter, NewProjectEnv
from chrisbase.io import LoggingFormat, set_verbosity_warning, set_verbosity_info, set_verbosity_error, to_table_lines, do_nothing, make_parent_dir
from typing_extensions import Annotated
import typer
import torch

# Global settings
env: Optional[NewProjectEnv] = None
app: AppTyper = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer.")
logger: logging.Logger = logging.getLogger(__name__)


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
        logging_format=LoggingFormat.CHECK_28,
        argument_file=argument_file,
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
    )
    torch.set_float32_matmul_precision('high')


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [4]: https://huggingface.co/docs/accelerate/en/quicktour
# [5]: https://huggingface.co/docs/accelerate/en/basic_tutorials/migration
# [6]: https://huggingface.co/docs/accelerate/en/basic_tutorials/execution
# [7]: https://huggingface.co/docs/transformers/en/main_classes/logging
@app.command()
def train(
        # input
        pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-small",  # (80M)
        # learn
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[str, typer.Option("--run_version")] = None,
):
    # Setup arguments
    basic_logger = CSVLogger(output_home, output_name, run_version, flush_logs_every_n_steps=1)

    # Setup HF arguments
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=basic_logger.log_dir,
        log_level=env.logging_level,
        report_to="tensorboard",
    )

    # Setup logging
    env.setup_logger(
        logging_home=basic_logger.log_dir,
        logging_level=training_args.get_process_log_level()
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f", distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.warning(f"training_args.get_process_log_level()={training_args.get_process_log_level()}")


if __name__ == "__main__":
    app()
