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

import contextlib
import itertools
import json
import logging
import math
import re
from typing import Any, Callable, Iterable, Tuple, Optional

import datasets
import lightning
import numpy as np
import pandas as pd
import torch
import transformers
import typer
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, BatchEncoding
from transformers.trainer import get_model_param_count, nested_concat, nested_numpify, denumpify_detensorize
from typing_extensions import Annotated

from DeepKNLP.arguments import NewProjectEnv, TrainingArguments
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from chrisbase.data import AppTyper, JobTimer, Counter
from chrisbase.io import LoggingFormat, set_verbosity_warning, set_verbosity_info, set_verbosity_error, to_table_lines
from chrisbase.util import shuffled, tupled
from chrisdata.ner import GenNERSampleWrapper
from progiter import ProgIter

# Global settings
env = None
app = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer. [trainer]")
logger = logging.getLogger(__name__)


def do_nothing(*args, **kwargs):
    pass


def info_or_debug(fabric, x, *y, **z):
    if fabric.is_global_zero:  # or debugging:
        logger.info(x, *y, **z)
    else:
        logger.debug(x, *y, **z)


def info_or_debug_r(fabric, x, *y, **z):
    x = str(x).rstrip()
    if fabric.is_global_zero:  # or debugging:
        logger.info(x, *y, **z)
    else:
        logger.debug(x, *y, **z)


@app.callback()
def main(
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
        argument_file=argument_file,
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
        message_level=logging.INFO,
        message_format=LoggingFormat.CHECK_20,
    )


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [4]: https://lightning.ai/docs/fabric/2.4.0/guide/
# [5]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_args.html
# [6]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_methods.html
# [7]: https://lightning.ai/docs/fabric/2.4.0/advanced/model_parallel/fsdp.html
# [8]: https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html
@app.command()
def train(
        # input
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/eagle-3b-preview",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-small",  # 80M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-base",  # 250M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-large",  # 780M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xl",  # 3B
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xxl",  # 11B
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "microsoft/Phi-3.5-mini-instruct",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-2-7b-hf",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-3B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.1-8B",
        # train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/pile-ner.jsonl",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        # study_file: Annotated[str, typer.Option("--study_file")] = "data/gner/KG-generation-YAGO3-53220@2.jsonl",
        study_file: Annotated[str, typer.Option("--study_file")] = None,
        # eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev.jsonl",
        eval_file: Annotated[str, typer.Option("--eval_file")] = None,
        # test_file: Annotated[str, typer.Option("--test_file")] = "data/gner/zero-shot-test.jsonl"
        test_file: Annotated[str, typer.Option("--test_file")] = None,
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,  # TODO: 512, 640
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,  # TODO: 512, 640
        max_generation_length: Annotated[int, typer.Option("--max_generation_length")] = 640,  # TODO: 512, 640
        max_train_samples: Annotated[int, typer.Option("--max_train_samples")] = 100,  # TODO: 256, -1
        max_study_samples: Annotated[int, typer.Option("--max_study_samples")] = -1,  # TODO: 256, -1
        max_eval_samples: Annotated[int, typer.Option("--max_eval_samples")] = -1,  # TODO: 256, 1024, -1
        max_test_samples: Annotated[int, typer.Option("--max_test_samples")] = -1,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--use_fresh_data")] = True,
        # learn
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[str, typer.Option("--run_version")] = None,
        num_train_epochs: Annotated[int, typer.Option("--num_train_epochs")] = 1,  # TODO: -> 1, 2, 3, 4, 5, 6
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 5e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,  # TODO: utilize lr_scheduler
        train_batch: Annotated[int, typer.Option("--train_batch")] = 1,  # TODO: -> 1, 2, 4, 8
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 10,  # TODO: -> 10, 20, 40
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 1,  # TODO: -> 2, 4, 8, 10, 20, 40
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = 10,  # TODO: -> 20, 40
        num_device: Annotated[int, typer.Option("--num_device")] = 1,  # TODO: -> 4, 8
        device_idx: Annotated[int, typer.Option("--device_idx")] = 0,  # TODO: -> 0, 4
        device_type: Annotated[str, typer.Option("--device_type")] = "gpu",  # TODO: -> gpu, cpu, mps
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: Annotated[str, typer.Option("--strategy")] = "deepspeed",  # TODO: -> ddp, fsdp, deepspeed
        ds_stage: Annotated[int, typer.Option("--ds_stage")] = 1,  # TODO: -> 1, 2, 3
        ds_offload: Annotated[int, typer.Option("--ds_offload")] = 0,  # TODO: -> 0, 1, 2, 3
        fsdp_shard: Annotated[str, typer.Option("--fsdp_shard")] = "FULL_SHARD",  # TODO: -> FULL_SHARD, SHARD_GRAD_OP
        fsdp_offload: Annotated[bool, typer.Option("--fsdp_offload")] = False,  # TODO: -> True, False
):
    # Setup arguments
    args = TrainingArguments(
        env=env,
        input=TrainingArguments.InputOption(
            pretrained=pretrained,
            train_file=train_file,
            study_file=study_file,
            eval_file=eval_file,
            test_file=test_file,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            max_generation_length=max_generation_length,
            max_train_samples=max_train_samples,
            max_study_samples=max_study_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            use_cache_data=use_cache_data,
        ),
        learn=TrainingArguments.LearnOption(
            output_home=output_home,
            output_name=output_name,
            run_version=run_version,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch=train_batch,
            infer_batch=infer_batch,
            grad_steps=grad_steps,
            eval_steps=eval_steps,
            num_device=num_device,
            device_idx=device_idx,
            device_type=device_type,
            precision=precision,
            strategy=strategy,
            ds_stage=ds_stage,
            ds_offload=ds_offload,
            fsdp_shard=fsdp_shard,
            fsdp_offload=fsdp_offload,
        ),
    )

    # Setup logger
    args.env.setup_logger(logging_home=args.learn.output_home)

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args, verbose=True,
    ):
        # Set random seed
        set_seed(args.env.random_seed)
        logger.warning(f"Set random seed: {args.env.random_seed}")


if __name__ == "__main__":
    app()
