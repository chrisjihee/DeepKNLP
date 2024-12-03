import math
import threading

from progiter import ProgIter
import tqdm
import datasets
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from pydantic_core import ArgsKwargs
from tqdm.asyncio import tqdm_asyncio, trange
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
from typing import List, Tuple, Dict, Mapping, Any, ClassVar, Union, Callable
import pandas as pd

import torch
import typer
from pydantic import BaseModel, Field

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, TypedData, TimeChecker, Counter
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
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, CharSpan, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BatchEncoding
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption, NewTrainerArguments, NewProjectEnv, NewDataOption, NewModelOption, NewLearningOption, NewHardwareOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier
from DeepKNLP.metrics import accuracy, NER_Char_MacroF1, NER_Entity_MacroF1
from DeepKNLP.ner import NERCorpus, NERDataset, NEREncodedExample
from chrisdata.ner import GenNERSampleWrapper

main = AppTyper()
logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


@main.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        logging_home: str = typer.Option(default="output/task2-nerG"),
        logging_file: str = typer.Option(default="train-messages.out"),
        argument_file: str = typer.Option(default="train-arguments.json"),
        max_workers: int = typer.Option(default=10),
        debugging: bool = typer.Option(default=False),
        # data
        # train_data: str = typer.Option(default="data/gner/zero-shot-train.jsonl"),
        train_data: str = typer.Option(default="data/gner/pile-ner.jsonl"),
        eval_data: str = typer.Option(default="data/gner/zero-shot-dev.jsonl"),
        # eval_data: str = typer.Option(default=None),
        test_data: str = typer.Option(default="data/gner/zero-shot-test.jsonl"),
        # test_data: str = typer.Option(default=None),
        max_train_samples: int = typer.Option(default=-1),
        max_eval_samples: int = typer.Option(default=-1),
        max_test_samples: int = typer.Option(default=-1),
        num_prog_samples: int = typer.Option(default=5000),
        max_source_length: int = typer.Option(default=512),
        max_target_length: int = typer.Option(default=512),
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
        device: List[int] = typer.Option(default=[0]),  # TODO: -> [0], [0,1], [0,1,2,3]
):
    torch.set_float32_matmul_precision('high')
    datasets.utils.logging.disable_progress_bar()
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
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            num_prog_samples=num_prog_samples,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
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
            args=args if fabric.is_global_zero else None, verbose=fabric.is_global_zero,
            # mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.learning.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(pretrained)
        using_decoder_only_model = not config.is_encoder_decoder
        if using_decoder_only_model:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
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
                if args.data.max_train_samples > 0:
                    train_dataset = train_dataset.select(range(min(len(train_dataset), args.data.max_train_samples)))
                fabric.print(f"Use {args.data.train_path} as train dataset: {len(train_dataset):,} samples")
        if args.data.eval_path:
            with fabric.rank_zero_first():
                eval_dataset: Dataset = load_dataset("json", data_files=args.data.eval_path, split=datasets.Split.TRAIN)
                if args.data.max_eval_samples > 0:
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.data.max_eval_samples)))
                fabric.print(f"Use {args.data.eval_path} as eval dataset: {len(eval_dataset):,} samples")
        if args.data.test_path:
            with fabric.rank_zero_first():
                test_dataset: Dataset = load_dataset("json", data_files=args.data.test_path, split=datasets.Split.TRAIN)
                if args.data.max_test_samples > 0:
                    test_dataset = test_dataset.select(range(min(len(test_dataset), args.data.max_test_samples)))
                fabric.print(f"Use {args.data.eval_path} as test dataset: {len(test_dataset):,} samples")
        fabric.barrier()
        fabric.print("-" * 100)

        # Define tokenizer function for decoder-only model
        def preprocess_for_decoder_only_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], counter=Counter(step=args.env.max_workers),
                                              update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
            # Fetch input data
            sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
            data_opt: NewDataOption = NewDataOption.model_validate(data_opt)
            prompt_text = f"[INST] {sample.instance.instruction_inputs} [/INST]"
            full_instruction = f"{prompt_text} {sample.instance.prompt_labels}"

            def tokenize_train_sample():
                # Tokenize the full instruction
                model_inputs = tokenizer(
                    text=full_instruction,
                    max_length=data_opt.max_source_length + data_opt.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )

                # Add eos token if it is not the last token
                if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
                    model_inputs["input_ids"].append(tokenizer.eos_token_id)
                    model_inputs["attention_mask"].append(1)

                # Add labels
                model_inputs["labels"] = model_inputs["input_ids"].copy()

                # Find the prompt length
                prompt_tokens = tokenizer(
                    text=prompt_text,
                    max_length=data_opt.max_source_length + data_opt.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )["input_ids"]

                # Remove the last token if it is an eos token
                if prompt_tokens[-1] == tokenizer.eos_token_id:
                    prompt_tokens = prompt_tokens[:-1]

                # Check if the prompt is longer than the input
                if len(prompt_tokens) > len(model_inputs["labels"]):
                    raise ValueError(
                        f"Prompt is longer than the input, something went wrong. Prompt: {prompt_tokens}, input:"
                        f" {model_inputs['input_ids']}"
                    )

                # Mask the prompt tokens
                for i in range(len(prompt_tokens)):
                    model_inputs["labels"][i] = -100

                return model_inputs

            def tokenize_infer_sample():
                # Tokenize the prompt
                model_inputs = tokenizer(
                    text=prompt_text,
                    max_length=max_source_length + max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )

                # Remove the last token if it is an eos token
                if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                    model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                    model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]

                return model_inputs

            # Tokenize the sample
            tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
            if update:
                return update(tokenized_sample, process_rank, counter)
            else:
                return tokenized_sample

        # Define tokenizer function for encoder-decoder model
        def preprocess_for_encoder_decoder_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], cnt=Counter(step=args.env.max_workers),
                                                 update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
            # Fetch input data
            sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
            data_opt: NewDataOption = NewDataOption.model_validate(data_opt)

            def tokenize_train_sample():
                # Tokenize the instruction inputs
                model_inputs = tokenizer(
                    text=sample.instance.instruction_inputs,
                    max_length=data_opt.max_source_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )

                # Tokenize the prompt labels
                model_inputs["labels"] = tokenizer(
                    text=sample.instance.prompt_labels,
                    max_length=data_opt.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )["input_ids"]

                return model_inputs

            def tokenize_infer_sample():
                # Tokenize the instruction inputs
                model_inputs = tokenizer(
                    text=sample.instance.instruction_inputs,
                    max_length=data_opt.max_source_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )
                return model_inputs

            # Tokenize the sample
            tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
            if update:
                return update(tokenized_sample, process_rank, cnt)
            else:
                return tokenized_sample

        # Define progress update function
        def update_progress(res: BatchEncoding, rank: int, counter: Counter, pbar: ProgIter):
            pre, cnt = counter.val(), counter.inc()
            if (cnt >= pbar.total or any(i % num_prog_samples == 0 for i in range(pre + 1, cnt + 1))) and rank == 0:
                pbar.step(inc=min(cnt - pbar._iter_idx, pbar.total - pbar._iter_idx))
                fabric.print(pbar.format_message().rstrip())
            return res

        # Preprocess dataset
        if train_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(train_dataset), desc='Preprocess train samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
                    train_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers,
                        load_from_cache_file=args.data.load_cache,
                    )
            fabric.print("-" * 100)
        if eval_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(eval_dataset), desc='Preprocess eval samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
                    eval_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers,
                        load_from_cache_file=args.data.load_cache,
                    )
            fabric.print("-" * 100)
        if test_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(test_dataset), desc='Preprocess test samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
                    test_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers,
                        load_from_cache_file=args.data.load_cache,
                    )
            fabric.print("-" * 100)

        fabric.print("*" * 100)
        fabric.barrier()


if __name__ == "__main__":
    main()
