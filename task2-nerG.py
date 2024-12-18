import contextlib
import itertools
import json
import logging
import math
import re
import unittest
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
app = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer.")
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


class TestCases(unittest.TestCase):
    def test_dataframe(self):
        print()
        score_dict = {
            "[avg]": 0.7621,
            "ai": 0.8461,
            "literature": 0.7302,
            "music": 0.7533,
            "politics": 0.7884,
            "science": 0.6673,
            "movie": 0.7744,
            "restaurant": 0.7742
        }
        score_scale = 100.0
        floatfmt = ".2f"
        width = 5

        score_headers = [
            (width - len(k[:width])) * ' ' + k[:width]
            for k in score_dict.keys()
        ]
        score_values = score_dict.values()

        data = pd.DataFrame(score_values) * score_scale
        data.set_index(pd.Index(score_headers), inplace=True)
        data = data.transpose()
        data["epoch"] = 2.44
        data["[mem]"] = 20.0
        data = data.set_index("epoch")
        print(data)
        for x in to_table_lines(data, transposed_df=True, floatfmt=floatfmt):
            print(x)

        data2 = data.copy()
        data2["epoch"] = 4.85
        data2 = data2.set_index("epoch")
        print(data2)

        data3 = pd.concat([data, data2])
        print("=" * 100)
        print(data3)


@app.command()
def check():
    checker = TestCases()
    checker.test_dataframe()


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
    set_verbosity_warning(
        # "root",
        # "DeepSpeed",
        "c10d-NullHandler",
        "c10d-NullHandler-default",
        "lightning.fabric.utilities.seed"
        # "lightning.pytorch.utilities.rank_zero",
        # "lightning.fabric.utilities.distributed",
    )
    torch.set_float32_matmul_precision('high')


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py
# [4]: https://lightning.ai/docs/fabric/2.4.0/guide/
# [5]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_args.html
# [6]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_methods.html
# [7]: https://lightning.ai/docs/fabric/2.4.0/advanced/model_parallel/fsdp.html
# [8]: https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html
@app.command()
def train(
        # input
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/eagle-3b-preview",  # RuntimeError: CUDA error
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",  # RuntimeError: CUDA error
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",  # RuntimeError: CUDA error
        pretrained: Annotated[str, typer.Option("--pretrained")] = "Qwen/Qwen2.5-3B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xl",  # RuntimeError: CUDA error
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "microsoft/Phi-3.5-mini-instruct",  # RuntimeError: CUDA error
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-2-7b-hf",  # RuntimeError: CUDA error
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.1-8B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-3B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/pile-ner.jsonl",
        # train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        # study_file: Annotated[str, typer.Option("--study_file")] = "data/gner/KG-generation-YAGO3-53220@2.jsonl",
        study_file: Annotated[str, typer.Option("--study_file")] = None,
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev.jsonl",
        # eval_file: Annotated[str, typer.Option("--eval_file")] = None,
        # test_file: Annotated[str, typer.Option("--test_file")] = "data/gner/zero-shot-test.jsonl"
        test_file: Annotated[str, typer.Option("--test_file")] = None,
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,  # TODO: 512, 640
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,  # TODO: 512, 640
        max_generation_length: Annotated[int, typer.Option("--max_generation_length")] = 1280,  # TODO: 512, 640
        max_train_samples: Annotated[int, typer.Option("--max_train_samples")] = -1,  # TODO: 256, -1
        max_study_samples: Annotated[int, typer.Option("--max_study_samples")] = -1,  # TODO: 256, -1
        max_eval_samples: Annotated[int, typer.Option("--max_eval_samples")] = -1,  # TODO: 256, 1024, -1
        max_test_samples: Annotated[int, typer.Option("--max_test_samples")] = -1,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--use_fresh_data")] = True,
        # learn
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[str, typer.Option("--run_version")] = None,
        num_train_epochs: Annotated[int, typer.Option("--num_train_epochs")] = 3,  # TODO: -> 1, 2, 3, 4, 5, 6
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 2e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,  # TODO: utilize lr_scheduler
        train_batch: Annotated[int, typer.Option("--train_batch")] = 1,  # TODO: -> 1, 2, 4, 8
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 10,  # TODO: -> 10, 20, 40
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 10,  # TODO: -> 2, 4, 8, 10, 20, 40
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = 40,  # TODO: -> 20, 40
        num_device: Annotated[int, typer.Option("--num_device")] = 8,  # TODO: -> 4, 8
        device_idx: Annotated[int, typer.Option("--device_idx")] = 0,  # TODO: -> 0, 4
        device_type: Annotated[str, typer.Option("--device_type")] = "gpu",  # TODO: -> gpu, cpu, mps
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: Annotated[str, typer.Option("--strategy")] = "deepspeed",  # TODO: -> ddp, fsdp, deepspeed
        ds_stage: Annotated[int, typer.Option("--ds_stage")] = 2,  # TODO: -> 1, 2, 3
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

    # Setup fabric
    lightning.fabric.loggers.csv_logs._ExperimentWriter.NAME_METRICS_FILE = "train-metrics.csv"
    basic_logger = CSVLogger(args.learn.output_home, args.learn.output_name, args.learn.run_version, flush_logs_every_n_steps=1)
    graph_logger = TensorBoardLogger(basic_logger.root_dir, basic_logger.name, basic_logger.version)  # tensorboard --logdir output --bind_all
    fabric = Fabric(
        accelerator=args.learn.device_type,
        precision=args.learn.precision,
        strategy=args.learn.strategy_inst,
        devices=args.learn.devices,
        loggers=[basic_logger, graph_logger],
    )
    fabric.launch()
    fabric.barrier()
    fabric.flush = do_nothing
    fabric.write = lambda x, *y, **z: info_or_debug_r(fabric, x, *y, **z)
    fabric.print = lambda x, *y, **z: info_or_debug(fabric, x, *y, **z)
    fabric.prints = lambda xs: [info_or_debug(fabric, x) for x in xs]
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size

    # Setup logger
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)
    args.env.setup_logger(logging_home=basic_logger.log_dir)
    # transformers.logging.disable_progress_bar()
    datasets.utils.logging.disable_progress_bar()
    if fabric.is_global_zero:
        transformers.logging.set_verbosity_info()
        # datasets.utils.logging.set_verbosity_warning()
        set_verbosity_info(
            "lightning",
        )
        set_verbosity_warning(
            "transformers.configuration_utils",
            "transformers.generation.configuration_utils",
            # "transformers.tokenization_utils_base",
            # "transformers.modeling_utils",
        )
        set_verbosity_error(
            "transformers.generation.utils",
        )
    else:
        transformers.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()
        set_verbosity_error(
            "lightning",
        )

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args if fabric.is_global_zero else None, verbose=fabric.is_global_zero,
            mute_warning="lightning.fabric.loggers.csv_logs"
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.env.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        using_decoder_only_model = not config.is_encoder_decoder

        if using_decoder_only_model:
            # tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained, padding_side="left",
            #                                                                    add_eos_token=True, add_bos_token=True)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        else:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        fabric.print(f"tokenizer.pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        fabric.print(f"tokenizer.eos_token={tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
            # tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
        fabric.print(f"tokenizer.pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})")

        if using_decoder_only_model:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, config=config, trust_remote_code=True)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained, config=config, trust_remote_code=True)
        model_embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > model_embedding_size:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
            model_embedding_size = model.get_input_embeddings().weight.shape[0]
        fabric.barrier()

        # Load dataset
        train_dataset: Dataset | None = None
        study_dataset: Dataset | None = None
        eval_dataset: Dataset | None = None
        test_dataset: Dataset | None = None
        if args.input.train_file:
            with fabric.rank_zero_first():
                train_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                      data_files=str(args.input.train_file),
                                                      cache_dir=str(args.input.cache_train_dir))
                if args.input.max_train_samples > 0:
                    whole_indices = shuffled(range(len(train_dataset)), seed=args.env.random_seed)
                    train_dataset = train_dataset.select(whole_indices[:args.input.max_train_samples])
                train_dataset = train_dataset.add_column("idx", range(len(train_dataset)))
                fabric.print(f"Load train dataset from {args.input.train_file} => {len(train_dataset):,}")
        if args.input.study_file:
            with fabric.rank_zero_first():
                study_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                      data_files=str(args.input.study_file),
                                                      cache_dir=str(args.input.cache_study_dir))
                if args.input.max_study_samples > 0:
                    whole_indices = shuffled(range(len(study_dataset)), seed=args.env.random_seed)
                    study_dataset = study_dataset.select(whole_indices[:args.input.max_study_samples])
                study_dataset = study_dataset.add_column("idx", range(len(study_dataset)))
                fabric.print(f"Load study dataset from {args.input.study_file} => {len(study_dataset):,}")
        if args.input.eval_file:
            with fabric.rank_zero_first():
                eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.eval_file),
                                                     cache_dir=str(args.input.cache_eval_dir))
                if args.input.max_eval_samples > 0:
                    whole_indices = shuffled(range(len(eval_dataset)), seed=args.env.random_seed)
                    eval_dataset = eval_dataset.select(whole_indices[:args.input.max_eval_samples])
                eval_dataset = eval_dataset.add_column("idx", range(len(eval_dataset)))
                fabric.print(f"Load  eval dataset from {args.input.eval_file} => {len(eval_dataset):,}")
        if args.input.test_file:
            with fabric.rank_zero_first():
                test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.test_file),
                                                     cache_dir=str(args.input.cache_test_dir))
                if args.input.max_test_samples > 0:
                    whole_indices = shuffled(range(len(test_dataset)), seed=args.env.random_seed)
                    test_dataset = test_dataset.select(whole_indices[:args.input.max_test_samples])
                test_dataset = test_dataset.add_column("idx", range(len(test_dataset)))
                fabric.print(f"Load  test dataset from {args.input.test_file} => {len(test_dataset):,}")
        fabric.barrier()
        fabric.print("-" * 100)

        # Define tokenizer function for decoder-only model
        def preprocess_for_decoder_only_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], counter=Counter(step=args.env.max_workers),
                                              update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
            # Fetch input data
            sample = GenNERSampleWrapper.model_validate(row)
            data_opt = TrainingArguments.InputOption.model_validate(data_opt)
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

                # model_inputs["num_token_full_instruction"] = len(model_inputs["input_ids"])
                # model_inputs["num_token_prompt_token"] = len(prompt_tokens)
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
            sample = GenNERSampleWrapper.model_validate(row)
            data_opt = TrainingArguments.InputOption.model_validate(data_opt)

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
            if rank > 0:
                return res
            pre, cnt = counter.val(), counter.inc()
            pbar.step(min(cnt - pbar._iter_idx, pbar.total - pbar._iter_idx), force=cnt >= pbar.total)
            return res

        # Preprocess dataset
        if train_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(train_dataset), desc='Preprocess train samples:', stream=fabric, verbose=2) as manual_pbar:
                    train_dataset = train_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                        cache_file_name=str(args.input.cache_train_path(len(train_dataset))) if args.input.use_cache_data else None,
                    )
                # fabric.print(train_dataset["num_token_full_instruction"])
                # fabric.print(train_dataset["num_token_prompt_token"])
        if study_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(study_dataset), desc='Preprocess study samples:', stream=fabric, verbose=2) as manual_pbar:
                    study_dataset = study_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                        cache_file_name=str(args.input.cache_study_path(len(study_dataset))) if args.input.use_cache_data else None,
                    )
                # fabric.print(study_dataset["num_token_full_instruction"])
                # fabric.print(study_dataset["num_token_prompt_token"])
        if eval_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(eval_dataset), desc='Preprocess  eval samples:', stream=fabric, verbose=2) as manual_pbar:
                    eval_dataset = eval_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                        cache_file_name=str(args.input.cache_eval_path(len(eval_dataset))) if args.input.use_cache_data else None,
                    )
        if test_dataset:
            with fabric.rank_zero_first():
                with ProgIter(total=len(test_dataset), desc='Preprocess  test samples:', stream=fabric, verbose=2) as manual_pbar:
                    test_dataset = test_dataset.map(
                        preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                        fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                        with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                        cache_file_name=str(args.input.cache_test_path(len(test_dataset))) if args.input.use_cache_data else None,
                    )
        fabric.barrier()
        fabric.print("-" * 100)

        # Set dataloader
        label_pad_token_id = -100
        train_dataloader: DataLoader | None = None
        study_dataloader: DataLoader | None = None
        eval_dataloader: DataLoader | None = None
        test_dataloader: DataLoader | None = None
        data_collator: DataCollatorForGNER = DataCollatorForGNER(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,  # if training_args.fp16 else None,
            label_pad_token_id=label_pad_token_id,
            return_tensors="pt",
            feature_names=("input_ids", "attention_mask", "labels", "idx"),
        )
        if train_dataset:
            train_dataloader = DataLoader(
                train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                batch_size=args.learn.train_batch,
                collate_fn=data_collator,
                drop_last=False,
            )
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
        if study_dataset:
            study_dataloader = DataLoader(
                study_dataset, sampler=RandomSampler(study_dataset, replacement=False),
                batch_size=args.learn.train_batch,
                collate_fn=data_collator,
                drop_last=False,
            )
            study_dataloader = fabric.setup_dataloaders(study_dataloader)
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset, sampler=SequentialSampler(eval_dataset),
                batch_size=args.learn.infer_batch,
                collate_fn=data_collator,
                drop_last=False,
            )
            eval_dataloader = fabric.setup_dataloaders(eval_dataloader)
        if test_dataset:
            test_dataloader = DataLoader(
                test_dataset, sampler=SequentialSampler(eval_dataset),
                batch_size=args.learn.infer_batch,
                collate_fn=data_collator,
                drop_last=False,
            )
            test_dataloader = fabric.setup_dataloaders(test_dataloader)

        # Define compute metrics function
        def compute_ner_metrics(pred_logits: np.ndarray, pred_indexs: list[int], dataset: Dataset,
                                save_prefix: str | None = None, save_suffix: str | None = None, save_keys: Iterable[str] | None = None) -> dict[str, float]:
            pred_logits = np.where(pred_logits != -100, pred_logits, tokenizer.pad_token_id)
            decoded_preds: list[str] = tokenizer.batch_decode(pred_logits, skip_special_tokens=True)
            if using_decoder_only_model:
                match_pattern = "[/INST]"
                for i, decoded_pred in enumerate(decoded_preds):
                    decoded_preds[i] = decoded_pred[decoded_pred.find(match_pattern) + len(match_pattern):].strip()

            all_examples = [example.copy() for example in dataset]
            for i, decoded_pred in zip(pred_indexs, decoded_preds):
                all_examples[i]["prediction"] = decoded_pred

            results = compute_metrics(all_examples, tokenizer=tokenizer, average_key="[avg]")
            if save_prefix is not None:
                outfile_path = args.env.logging_home / f"{save_prefix}-text_generations{'-' + save_suffix if save_suffix else ''}.jsonl"
                with outfile_path.open("w") as fout:
                    for example in all_examples:
                        if save_keys:
                            example = {key: example[key] for key in save_keys if key in example}
                        fout.write(json.dumps(example) + "\n")
            return results

        def get_performance(origin: dict[str, float], column_name_width=5,
                            score_keys="f1", score_scale: float = 100.0,
                            index_keys="epoch", env_keys="[mem]") -> Tuple[dict[str, float], Optional[pd.DataFrame]]:
            score_keys = tupled(score_keys)
            score_dict = {}
            for key1 in sorted(origin.keys()):
                if any(score_key in key1 for score_key in score_keys):
                    key2 = key1
                    for score_key in score_keys:
                        key2 = key2.removesuffix(f"_{score_key}")
                    key2 = re.sub('[^-_]+[-_]', '', key2)
                    score_dict[key2] = origin[key1]

            performance = None
            if score_dict:
                performance = pd.DataFrame(score_dict.values()) * score_scale
                performance = performance.set_index(pd.Index([
                    (column_name_width - len(k2)) * ' ' + k2
                    for k2 in [k1[:column_name_width] for k1 in score_dict.keys()]
                ]))
                performance = performance.transpose()
                for idx_key in tupled(index_keys):
                    performance[idx_key] = origin[idx_key]
                for env_key in tupled(env_keys):
                    performance[env_key] = origin[env_key]
                performance = performance.set_index(index_keys)
            return performance

        # Train loop
        if train_dataloader:
            epoch_optimization_steps = math.ceil(len(train_dataloader) / args.learn.grad_steps)
            epoch_per_step = 1.0 / epoch_optimization_steps
            train_losses = []
            study_losses = []
            joint_losses = []
            performance_rows = []
            global_step = 0
            global_epoch = 0.0
            total_epochs = args.learn.num_train_epochs
            gen_kwargs = {
                "num_beams": 1,
                "max_length": args.input.max_generation_length,
            }

            fabric.print(f"===== Running training =====")
            fabric.print(f"  # Total Tokens = {len(tokenizer):,}")
            fabric.print(f"  # Model Embedding = {model_embedding_size:,}")
            fabric.print(f"  # Model Parameters = {get_model_param_count(model, trainable_only=True):,}")
            fabric.print(f"  # Total Train Epochs = {total_epochs}")
            fabric.print(f"  # Train Total Samples = {len(train_dataset):,}")
            if study_dataset:
                fabric.print(f"  # Study Total Samples = {len(study_dataset):,}")
            if eval_dataset:
                fabric.print(f"  #  Eval Total Samples = {len(eval_dataset):,}")
            fabric.print(f"  # Train Batch Samples"
                         f" = {args.learn.train_batch} * {args.learn.grad_steps} * {args.env.world_size}"
                         f" = {args.env.world_size * args.learn.train_batch * args.learn.grad_steps}")
            if eval_dataset:
                fabric.print(f"  #  Eval Batch Samples"
                             f" = {args.learn.infer_batch} * {args.env.world_size}"
                             f" = {args.learn.infer_batch * args.env.world_size}")
            fabric.print(f"  # Total Optim Steps = {epoch_optimization_steps * total_epochs:,}")
            fabric.print(f"  #  Eval Cycle Steps = {args.learn.eval_steps}")

            # Set optimizer
            no_decay = ("bias", "layer_norm.weight",)
            optimizer: Optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": args.learn.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ],
                lr=args.learn.learning_rate,
            )
            model_optimizer = fabric.setup(model, optimizer)
            model: lightning.fabric.wrappers._FabricModule = model_optimizer[0]
            optimizer: lightning.fabric.wrappers._FabricOptimizer = model_optimizer[1]
            model.mark_forward_method("generate")

            for epoch in range(args.learn.num_train_epochs):
                fabric.barrier()
                fabric.print("-" * 100)
                model.train()
                torch.cuda.reset_peak_memory_stats()
                with ProgIter(total=epoch_optimization_steps, desc=f'Training [{global_epoch:.2f}/{total_epochs}]:', stream=fabric, verbose=2, time_thresh=3.0) as train_pbar:
                    study_iter = itertools.chain(*[iter(study_dataloader) for _ in range(math.ceil(len(train_dataloader) / len(study_dataloader)))]) if study_dataloader else None
                    for train_loop_i, train_batch in enumerate(train_dataloader, start=1):
                        train_batch: BatchEncoding = train_batch
                        study_batch: BatchEncoding | None = next(study_iter) if study_iter else None

                        # Forward & Backward pass
                        is_accumulating = train_loop_i % args.learn.grad_steps != 0 and train_loop_i != len(train_dataloader)
                        with fabric.no_backward_sync(model, enabled=is_accumulating) if args.learn.strategy != "deepspeed" else contextlib.nullcontext():
                            if study_batch:
                                study_batch.pop("idx")
                                study_outputs = model(**study_batch)
                                study_losses.append(study_outputs.loss.item())
                                # fabric.backward(study_outputs.loss)
                            if train_batch:
                                train_batch.pop("idx")
                                train_outputs = model(**train_batch)
                                train_losses.append(train_outputs.loss.item())
                                # fabric.backward(train_outputs.loss)
                            if train_batch and study_batch:
                                joint_loss = train_outputs.loss + study_outputs.loss
                            else:
                                joint_loss = train_outputs.loss
                            joint_losses.append(joint_loss.item())
                            fabric.backward(joint_loss)
                        if is_accumulating:
                            continue

                        # Optimize parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        fabric.barrier()
                        global_epoch += epoch_per_step
                        global_step += 1

                        # Define metrics and Update train progress
                        metrics: dict[str, int | float] = {
                            "step": round(fabric.all_gather(torch.tensor(global_step * 1.0)).mean().item()),
                            "epoch": round(fabric.all_gather(torch.tensor(global_epoch)).mean().item(), 4),
                            "[mem]": torch.cuda.max_memory_reserved() / math.pow(1024, 3),
                            "joint_loss": torch.cat(fabric.all_gather(joint_losses)).mean().item(),
                            "train_loss": torch.cat(fabric.all_gather(train_losses)).mean().item(),
                        }
                        joint_losses.clear()
                        train_losses.clear()
                        if study_batch:
                            metrics["study_loss"] = torch.cat(fabric.all_gather(study_losses)).mean().item() if study_batch else None
                            study_losses.clear()
                            train_pbar.set_extra(f'| M={torch.cuda.max_memory_reserved() / math.pow(1024, 3):.0f} '
                                                 f'| joint_loss={metrics["joint_loss"]:.4f}, train_loss={metrics["train_loss"]:.4f}, study_loss={metrics["study_loss"]:.4f}')
                        else:
                            train_pbar.set_extra(f'| M={torch.cuda.max_memory_reserved() / math.pow(1024, 3):.0f} '
                                                 f'| train_loss={metrics["train_loss"]:.4f}')
                        train_pbar.set_description(f'Training [{global_epoch:.2f}/{total_epochs}]:', refresh=False)
                        train_pbar.step(force=train_loop_i == 1 or train_loop_i >= len(train_dataloader))

                        # Validation loop
                        if eval_dataloader and (
                                train_loop_i == len(train_dataloader) or
                                (args.learn.eval_steps > 0 and global_step % args.learn.eval_steps) == 0
                        ):
                            train_pbar.refresh()
                            model.eval()
                            accelerator = Accelerator()
                            with ProgIter(total=len(eval_dataloader), desc=f' Testing [{global_epoch:.2f}/{total_epochs}]:', stream=fabric, verbose=2, time_thresh=3.0) as eval_pbar:
                                with torch.no_grad():
                                    eval_logits = None
                                    eval_indexs = None
                                    for eval_loop_i, eval_batch in enumerate(eval_dataloader, start=1):
                                        eval_batch: BatchEncoding = eval_batch
                                        indexs: torch.Tensor = eval_batch.pop("idx")
                                        logits: torch.Tensor = model.generate(**eval_batch, **gen_kwargs)
                                        logits = accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                                        logits = accelerator.gather_for_metrics(logits)
                                        indexs = accelerator.gather_for_metrics(indexs)
                                        eval_logits = logits if eval_logits is None else nested_concat(eval_logits, logits, padding_index=-100)
                                        eval_indexs = indexs if eval_indexs is None else nested_concat(eval_indexs, indexs, padding_index=-100)
                                        eval_pbar.set_extra(f"| M={torch.cuda.max_memory_reserved() / math.pow(1024, 3):.0f}")
                                        eval_pbar.step(force=eval_loop_i == 1 or eval_loop_i >= len(eval_dataloader))
                                    eval_logits = nested_numpify(eval_logits)
                                    eval_indexs = nested_numpify(eval_indexs).tolist()
                                    metrics.update(compute_ner_metrics(eval_logits, eval_indexs, eval_dataset, save_prefix="eval", save_suffix=f"{global_epoch:.2f}",
                                                                       save_keys=["id", "dataset", "split", "prediction", "instance", "label_list", "input_ids", "attention_mask"]))
                            model.train()
                            torch.cuda.reset_peak_memory_stats()

                        # Log metrics
                        metrics = denumpify_detensorize(metrics)
                        fabric.log_dict(metrics=metrics, step=metrics["step"])  # TODO: utilize transformers.trainer_utils.speed_metrics()
                        performance_row = get_performance(metrics)
                        if fabric.is_global_zero and performance_row is not None:
                            performance_rows.append(performance_row)
                            fabric.prints(to_table_lines(performance_row, transposed_df=True, floatfmt=".2f"))
                        fabric.barrier()

                # Save performance
                model.eval()
                if fabric.is_global_zero:
                    performance_table = pd.concat(performance_rows)
                    performance_table.to_excel(args.env.logging_home / f"eval-performances.xlsx")
                    fabric.prints(to_table_lines(performance_table, transposed_df=True, floatfmt=".2f"))
                fabric.barrier()


if __name__ == "__main__":
    app()
