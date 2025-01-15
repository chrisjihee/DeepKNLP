import random
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic import Field, model_validator
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset
from sympy.vector import gradient

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
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry, is_torch_tf32_available, is_torch_bf16_available, is_torch_bf16_gpu_available
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
app: AppTyper = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer.")
env: Optional[NewProjectEnv] = None
csv: Optional[CSVLogger] = None
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
        # out
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[str, typer.Option("--run_version")] = "EAGLE-1B-debug",
):
    global env, csv
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
    csv = CSVLogger(output_home, output_name, run_version, flush_logs_every_n_steps=1)
    torch.set_float32_matmul_precision('high')


@dataclass
class Seq2SeqTrainingArgumentsForGNER(Seq2SeqTrainingArguments):
    model_name_or_path: str = field(default=None)
    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    test_data_path: str = field(default=None)
    max_source_length: int = field(default=640)
    max_target_length: int = field(default=640)


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [4]: https://huggingface.co/docs/accelerate/en/quicktour
# [5]: https://huggingface.co/docs/accelerate/en/basic_tutorials/migration
# [6]: https://huggingface.co/docs/accelerate/en/basic_tutorials/execution
# [7]: https://huggingface.co/docs/transformers/en/main_classes/logging
# [8]: https://huggingface.co/docs/transformers/en/main_classes/trainer
@app.command()
def train(
        # ModelArguments
        model_name_or_path: Annotated[str, typer.Option("--model_name_or_path")] = "etri-lirs/egpt-1.3b-preview",
        # DataTrainingArguments
        overwrite_cache: Annotated[bool, typer.Option("--overwrite_cache/--use_cache")] = False,
        train_data_path: Annotated[str, typer.Option("--train_json_file")] = "data/gner/zero-shot-train.jsonl",
        eval_data_path: Annotated[str, typer.Option("--eval_json_file")] = "data/gner/zero-shot-dev.jsonl",
        test_data_path: Annotated[str, typer.Option("--test_json_file")] = "data/gner/zero-shot-test-min.jsonl",
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        generation_max_length: Annotated[int, typer.Option("--generation_max_length")] = 1280,
        # Seq2SeqTrainingArguments
        deepspeed: Annotated[str, typer.Option("--deepspeed")] = "configs/deepspeed_configs/deepspeed_zero1_llama.json",
):
    # Setup training arguments
    training_args = Seq2SeqTrainingArgumentsForGNER(
        model_name_or_path=model_name_or_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        test_data_path=test_data_path,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        generation_max_length=generation_max_length,
        overwrite_output_dir=True,
        output_dir=csv.log_dir,
        report_to="tensorboard",
        log_level=env.logging_level,
        seed=env.random_seed,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=0.5,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_ratio=0.04,
        weight_decay=0.,
        tf32=is_torch_tf32_available(),
        bf16=is_torch_bf16_gpu_available(),
        bf16_full_eval=is_torch_bf16_gpu_available(),
        deepspeed=deepspeed,
    )

    # Setup logging
    env.setup_logger(
        logging_home=training_args.output_dir,
        logging_level=training_args.get_process_log_level()
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f", distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"training_args.should_log={training_args.should_log}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info(f"training_args.seed={training_args.seed}")


if __name__ == "__main__":
    app()
