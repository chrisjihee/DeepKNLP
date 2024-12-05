import math
import threading

import pydantic
import transformers.utils.logging
from progiter import ProgIter
import tqdm
import datasets
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from pydantic.v1.dataclasses import create_pydantic_model_from_dataclass
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
from typing import List, Tuple, Dict, Mapping, Any, ClassVar, Union, Callable, Type
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
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, CharSpan, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BatchEncoding, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.data.data_collator import *

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption, NewTrainerArguments, NewProjectEnv, NewDataOption, NewModelOption, NewLearningOption, NewHardwareOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier
from DeepKNLP.metrics import accuracy, NER_Char_MacroF1, NER_Entity_MacroF1
from DeepKNLP.ner import NERCorpus, NERDataset, NEREncodedExample
from chrisdata.ner import GenNERSampleWrapper

main = AppTyper()
logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


@dataclass
class DataCollatorForGNER:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        used_features_name = ["input_ids", "attention_mask", "labels"]
        features = [dict([item for item in feature.items() if item[0] in used_features_name]) for feature in features]

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@main.command()
def train(
        # env
        logging_home: str = typer.Option(default="output/task2-nerG"),
        logging_file: str = typer.Option(default="train-messages.out"),
        argument_file: str = typer.Option(default="train-arguments.json"),
        max_workers: int = typer.Option(default=4),
        debugging: bool = typer.Option(default=False),
        # data
        train_data: str = typer.Option(default="data/gner/zero-shot-train.jsonl"),
        # train_data: str = typer.Option(default="data/gner/pile-ner.jsonl"),
        eval_data: str = typer.Option(default="data/gner/zero-shot-dev.jsonl"),
        # eval_data: str = typer.Option(default=None),
        test_data: str = typer.Option(default="data/gner/zero-shot-test.jsonl"),
        # test_data: str = typer.Option(default=None),
        # max_train_samples: int = typer.Option(default=20000),
        max_train_samples: int = typer.Option(default=-1),
        max_eval_samples: int = typer.Option(default=-1),
        max_test_samples: int = typer.Option(default=-1),
        num_prog_samples: int = typer.Option(default=5000),
        max_source_length: int = typer.Option(default=512),
        max_target_length: int = typer.Option(default=512),
        use_cache_data: bool = typer.Option(default=True),
        # model
        # pretrained: str = typer.Option(default="google/flan-t5-small"),
        # pretrained: str = typer.Option(default="meta-llama/Llama-2-7b-hf"),
        pretrained: str = typer.Option(default="meta-llama/Llama-3.2-1B"),
        # pretrained: str = typer.Option(default="etri-lirs/kebyt5-small-preview"),
        # pretrained: str = typer.Option(default="etri-lirs/egpt-1.3b-preview"),
        # learning
        random_seed: int = typer.Option(default=7),
        trainer_args: str = typer.Option(default="configs/args/train_llama3_1b_supervised-base.json"),
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
    datasets.utils.logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("lightning").setLevel(logging.INFO)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.INFO)
    logging.getLogger("datasets.info").setLevel(logging.WARNING)
    logging.getLogger("datasets.builder").setLevel(logging.WARNING)
    logging.getLogger("datasets.arrow_dataset").setLevel(logging.WARNING)
    logging.getLogger("datasets.download.download_manager").setLevel(logging.WARNING)
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
            use_cache_data=use_cache_data,
        ),
        model=NewModelOption(
            pretrained=pretrained,
        ),
        learning=NewLearningOption(
            random_seed=random_seed,
            trainer_args=trainer_args,
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
    trainer_args: Seq2SeqTrainingArguments = HfArgumentParser(Seq2SeqTrainingArguments).parse_json_file(args.learning.trainer_args)[0]

    @dataclass
    class TrainerArguments:
        output_dir: str = Field(default="output/task2-nerG")

    bbb = TrainerArguments()
    exclude_fields = []

    from pydantic import create_model
    from typing import get_type_hints

    hints = get_type_hints(bbb)
    attributes = {}
    for k, v in hints.items():
        if k in exclude_fields:
            attributes[k] = (v, Field(..., exclude=True))
        else:
            attributes[k] = (v, Field(..., include=True))

    new_model = create_model("DynamicBaseModel", **attributes)

    fabric.print(new_model(output_dir="test").model_dump())
    exit(1)

    args.learning.trainer_args = trainer_args.to_dict()

    # for checking
    # logger.info(f"[local_rank={fabric.local_rank}] args.env={args.env}({type(args.env)})")

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args if fabric.is_global_zero else None, verbose=fabric.is_global_zero,
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.learning.random_seed)

        # # Load model
        # config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(pretrained)
        # using_decoder_only_model = not config.is_encoder_decoder
        # if using_decoder_only_model:
        #     model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained)
        # else:
        #     model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        # fabric.barrier()
        # fabric.print(f"type(model)={type(model)} - {isinstance(model, PreTrainedModel)}")
        # fabric.print(f"type(config)={type(config)} - {isinstance(config, PretrainedConfig)}")
        # fabric.print(f"type(tokenizer)={type(tokenizer)} - {isinstance(tokenizer, PreTrainedTokenizerFast)}")
        # fabric.print("-" * 100)
        #
        # # Load dataset
        # train_dataset: Dataset | None = None
        # eval_dataset: Dataset | None = None
        # test_dataset: Dataset | None = None
        # if args.data.train_path:
        #     with fabric.rank_zero_first():
        #         train_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                               data_files=str(args.data.train_path),
        #                                               cache_dir=str(args.data.cache_train_dir))
        #         if args.data.max_train_samples > 0:
        #             train_dataset = train_dataset.select(range(min(len(train_dataset), args.data.max_train_samples)))
        #         fabric.print(f"Use {args.data.train_path} as train dataset: {len(train_dataset):,} samples")
        # if args.data.eval_path:
        #     with fabric.rank_zero_first():
        #         eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                              data_files=str(args.data.eval_path),
        #                                              cache_dir=str(args.data.cache_eval_dir))
        #         if args.data.max_eval_samples > 0:
        #             eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.data.max_eval_samples)))
        #         fabric.print(f"Use {args.data.eval_path} as eval dataset: {len(eval_dataset):,} samples")
        # if args.data.test_path:
        #     with fabric.rank_zero_first():
        #         test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                              data_files=str(args.data.test_path),
        #                                              cache_dir=str(args.data.cache_test_dir))
        #         if args.data.max_test_samples > 0:
        #             test_dataset = test_dataset.select(range(min(len(test_dataset), args.data.max_test_samples)))
        #         fabric.print(f"Use {args.data.eval_path} as test dataset: {len(test_dataset):,} samples")
        # fabric.barrier()
        # fabric.print("-" * 100)
        #
        # # Define tokenizer function for decoder-only model
        # def preprocess_for_decoder_only_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], counter=Counter(step=args.env.max_workers),
        #                                       update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
        #     # Fetch input data
        #     sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
        #     data_opt: NewDataOption = NewDataOption.model_validate(data_opt)
        #     prompt_text = f"[INST] {sample.instance.instruction_inputs} [/INST]"
        #     full_instruction = f"{prompt_text} {sample.instance.prompt_labels}"
        #
        #     def tokenize_train_sample():
        #         # Tokenize the full instruction
        #         model_inputs = tokenizer(
        #             text=full_instruction,
        #             max_length=data_opt.max_source_length + data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Add eos token if it is not the last token
        #         if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
        #             model_inputs["input_ids"].append(tokenizer.eos_token_id)
        #             model_inputs["attention_mask"].append(1)
        #
        #         # Add labels
        #         model_inputs["labels"] = model_inputs["input_ids"].copy()
        #
        #         # Find the prompt length
        #         prompt_tokens = tokenizer(
        #             text=prompt_text,
        #             max_length=data_opt.max_source_length + data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )["input_ids"]
        #
        #         # Remove the last token if it is an eos token
        #         if prompt_tokens[-1] == tokenizer.eos_token_id:
        #             prompt_tokens = prompt_tokens[:-1]
        #
        #         # Check if the prompt is longer than the input
        #         if len(prompt_tokens) > len(model_inputs["labels"]):
        #             raise ValueError(
        #                 f"Prompt is longer than the input, something went wrong. Prompt: {prompt_tokens}, input:"
        #                 f" {model_inputs['input_ids']}"
        #             )
        #
        #         # Mask the prompt tokens
        #         for i in range(len(prompt_tokens)):
        #             model_inputs["labels"][i] = -100
        #
        #         return model_inputs
        #
        #     def tokenize_infer_sample():
        #         # Tokenize the prompt
        #         model_inputs = tokenizer(
        #             text=prompt_text,
        #             max_length=max_source_length + max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Remove the last token if it is an eos token
        #         if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
        #             model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
        #             model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]
        #
        #         return model_inputs
        #
        #     # Tokenize the sample
        #     tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
        #     if update:
        #         return update(tokenized_sample, process_rank, counter)
        #     else:
        #         return tokenized_sample
        #
        # # Define tokenizer function for encoder-decoder model
        # def preprocess_for_encoder_decoder_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], cnt=Counter(step=args.env.max_workers),
        #                                          update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
        #     # Fetch input data
        #     sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
        #     data_opt: NewDataOption = NewDataOption.model_validate(data_opt)
        #
        #     def tokenize_train_sample():
        #         # Tokenize the instruction inputs
        #         model_inputs = tokenizer(
        #             text=sample.instance.instruction_inputs,
        #             max_length=data_opt.max_source_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Tokenize the prompt labels
        #         model_inputs["labels"] = tokenizer(
        #             text=sample.instance.prompt_labels,
        #             max_length=data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )["input_ids"]
        #
        #         return model_inputs
        #
        #     def tokenize_infer_sample():
        #         # Tokenize the instruction inputs
        #         model_inputs = tokenizer(
        #             text=sample.instance.instruction_inputs,
        #             max_length=data_opt.max_source_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #         return model_inputs
        #
        #     # Tokenize the sample
        #     tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
        #     if update:
        #         return update(tokenized_sample, process_rank, cnt)
        #     else:
        #         return tokenized_sample
        #
        # # Define progress update function
        # def update_progress(res: BatchEncoding, rank: int, counter: Counter, pbar: ProgIter):
        #     pre, cnt = counter.val(), counter.inc()
        #     if (cnt >= pbar.total or any(i % num_prog_samples == 0 for i in range(pre + 1, cnt + 1))) and rank == 0:
        #         pbar.step(inc=min(cnt - pbar._iter_idx, pbar.total - pbar._iter_idx))
        #         # fabric.print(pbar.format_message().rstrip())
        #         logger.info(pbar.format_message().rstrip())
        #     return res
        #
        # # Preprocess dataset
        # if train_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(train_dataset), desc='Preprocess train samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             train_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_train_path(len(train_dataset))) if args.data.use_cache_data else None,
        #             )
        # if eval_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(eval_dataset), desc='Preprocess eval samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             eval_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_eval_path(len(eval_dataset))) if args.data.use_cache_data else None,
        #             )
        # if test_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(test_dataset), desc='Preprocess test samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             test_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_test_path(len(test_dataset))) if args.data.use_cache_data else None,
        #             )
        # fabric.barrier()
        # fabric.print("-" * 100)
        #
        # # Construct Data collator
        # label_pad_token_id = -100
        # data_collator: DataCollatorForGNER = DataCollatorForGNER(
        #     tokenizer,
        #     model=model,
        #     padding=True,
        #     label_pad_token_id=label_pad_token_id,
        #     return_tensors="pt",
        # )
        # fabric.barrier()
        # fabric.print("*" * 100)


if __name__ == "__main__":
    main()
