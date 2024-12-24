import json
import logging
import math
from typing import Any, Callable, Iterable

import datasets
import lightning
import numpy as np
import torch
import transformers
import typer
from chrisbase.data import AppTyper, JobTimer, Counter, NewProjectEnv
from chrisbase.io import LoggingFormat, do_nothing
from chrisbase.io import info_r
from chrisbase.util import shuffled
from chrisdata.ner import GenNERSampleWrapper
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, BatchEncoding, Seq2SeqTrainingArguments, set_seed
from typing_extensions import Annotated

from DeepKNLP.arguments import TrainingArguments
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from progiter import ProgIter

# Global settings
env = None
app = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer. [trainer]")
logger = logging.getLogger(__name__)
logger.flush = do_nothing
logger.write = lambda x, *y, **z: info_r(x, *y, **z)


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
        logging_level="info",
        logging_format=LoggingFormat.TRACE_28,
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
# [4]: https://lightning.ai/docs/fabric/2.4.0/guide/
# [5]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_args.html
# [6]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_methods.html
# [7]: https://lightning.ai/docs/fabric/2.4.0/advanced/model_parallel/fsdp.html
# [8]: https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html
@app.command()
def train(
        # input
        pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-small",  # (80M)
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-base",  # (250M)
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-large",  # (780M)
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xl",  # (3B)
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xxl",  # (11B) #torch.cuda.OutOfMemoryError: CUDA out of memory.
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",  # ValueError: Expected input batch_size (175) to match target batch_size (55)
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-3B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.1-8B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-2-7b-hf",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "microsoft/Phi-3.5-mini-instruct",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/eagle-3b-preview",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
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
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 2e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,  # TODO: utilize lr_scheduler
        train_batch: Annotated[int, typer.Option("--train_batch")] = 1,  # TODO: -> 1, 2, 4, 8
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 1,  # TODO: -> 10, 20, 40
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 1,  # TODO: -> 2, 4, 8, 10, 20, 40
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = 1,  # TODO: -> 20, 40
        num_device: Annotated[int, typer.Option("--num_device")] = 8,  # TODO: -> 4, 8
        device_idx: Annotated[int, typer.Option("--device_idx")] = 0,  # TODO: -> 0, 4
        device_type: Annotated[str, typer.Option("--device_type")] = "gpu",  # TODO: -> gpu, cpu, mps
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: Annotated[str, typer.Option("--strategy")] = "deepspeed",  # TODO: -> ddp, fsdp, deepspeed
        ds_stage: Annotated[int, typer.Option("--ds_stage")] = 3,  # TODO: -> 1, 2, 3
        ds_offload: Annotated[int, typer.Option("--ds_offload")] = 3,  # TODO: -> 0, 1, 2, 3
        fsdp_shard: Annotated[str, typer.Option("--fsdp_shard")] = "FULL_SHARD",  # TODO: -> FULL_SHARD, SHARD_GRAD_OP
        fsdp_offload: Annotated[bool, typer.Option("--fsdp_offload")] = False,  # TODO: -> True, False
):
    # Setup arguments
    basic_logger = CSVLogger(output_home, output_name, run_version, flush_logs_every_n_steps=1)
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
    fabric = Fabric(
        accelerator=args.learn.device_type,
        precision=args.learn.precision,
        strategy=args.learn.strategy_inst,
        devices=args.learn.devices,
        loggers=[basic_logger],
    )
    fabric.launch()
    fabric.barrier()
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)

    # Setup HF arguments
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=basic_logger.log_dir,
        log_level=args.env.logging_level,
    )
    training_args.set_training(
        learning_rate=args.learn.learning_rate,
        batch_size=args.learn.train_batch,
        weight_decay=args.learn.weight_decay,
        num_epochs=args.learn.num_train_epochs,
        max_steps=-1,
        gradient_accumulation_steps=args.learn.grad_steps,
        seed=args.env.random_seed,
        gradient_checkpointing=False,
    )
    training_args.set_dataloader(
        train_batch_size=args.learn.train_batch,
        eval_batch_size=args.learn.infer_batch,
        drop_last=False,
        num_workers=args.env.max_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        auto_find_batch_size=False,
        ignore_data_skip=False,
        sampler_seed=args.env.random_seed,
    )
    training_args.set_optimizer(
        name="adamw_torch",
        learning_rate=args.learn.learning_rate,
        weight_decay=args.learn.weight_decay,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    training_args.bf16 = "bf16" in args.learn.precision
    training_args.fp16 = "16" in args.learn.precision

    # Setup logger
    log_level = training_args.get_process_log_level()
    datasets.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    args.env.setup_logger(logging_home=basic_logger.log_dir, level=log_level)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f", distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args, verbose=True,
    ):
        # Set random seed
        set_seed(args.env.random_seed)
        fabric.seed_everything(args.env.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        using_decoder_only_model = not config.is_encoder_decoder

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        logger.info(f"tokenizer.pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        logger.info(f"tokenizer.eos_token={tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
            # tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            logger.info(f"tokenizer.pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})")

        if using_decoder_only_model:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, config=config, trust_remote_code=True)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained, config=config, trust_remote_code=True)
        model_embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > model_embedding_size:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
            model_embedding_size = model.get_input_embeddings().weight.shape[0]

        # Load dataset
        train_dataset: Dataset | None = None
        study_dataset: Dataset | None = None
        eval_dataset: Dataset | None = None
        test_dataset: Dataset | None = None
        if args.input.train_file:
            train_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                  data_files=str(args.input.train_file),
                                                  cache_dir=str(args.input.cache_train_dir))
            if args.input.max_train_samples > 0:
                whole_indices = shuffled(range(len(train_dataset)), seed=args.env.random_seed)
                train_dataset = train_dataset.select(whole_indices[:args.input.max_train_samples])
            train_dataset = train_dataset.add_column("idx", range(len(train_dataset)))
            logger.info(f"Load train dataset from {args.input.train_file} => {len(train_dataset):,}")
        if args.input.study_file:
            study_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                  data_files=str(args.input.study_file),
                                                  cache_dir=str(args.input.cache_study_dir))
            if args.input.max_study_samples > 0:
                whole_indices = shuffled(range(len(study_dataset)), seed=args.env.random_seed)
                study_dataset = study_dataset.select(whole_indices[:args.input.max_study_samples])
            study_dataset = study_dataset.add_column("idx", range(len(study_dataset)))
            logger.info(f"Load study dataset from {args.input.study_file} => {len(study_dataset):,}")
        if args.input.eval_file:
            eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                 data_files=str(args.input.eval_file),
                                                 cache_dir=str(args.input.cache_eval_dir))
            if args.input.max_eval_samples > 0:
                whole_indices = shuffled(range(len(eval_dataset)), seed=args.env.random_seed)
                eval_dataset = eval_dataset.select(whole_indices[:args.input.max_eval_samples])
            eval_dataset = eval_dataset.add_column("idx", range(len(eval_dataset)))
            logger.info(f"Load  eval dataset from {args.input.eval_file} => {len(eval_dataset):,}")
        if args.input.test_file:
            test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                 data_files=str(args.input.test_file),
                                                 cache_dir=str(args.input.cache_test_dir))
            if args.input.max_test_samples > 0:
                whole_indices = shuffled(range(len(test_dataset)), seed=args.env.random_seed)
                test_dataset = test_dataset.select(whole_indices[:args.input.max_test_samples])
            test_dataset = test_dataset.add_column("idx", range(len(test_dataset)))
            logger.info(f"Load  test dataset from {args.input.test_file} => {len(test_dataset):,}")
        logger.info("-" * 100)

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
            with ProgIter(total=len(train_dataset), desc='Preprocess train samples:', stream=logger, verbose=2) as manual_pbar:
                train_dataset = train_dataset.map(
                    preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                    fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                    with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                    cache_file_name=str(args.input.cache_train_path(len(train_dataset))) if args.input.use_cache_data else None,
                )
        if study_dataset:
            with ProgIter(total=len(study_dataset), desc='Preprocess study samples:', stream=logger, verbose=2) as manual_pbar:
                study_dataset = study_dataset.map(
                    preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                    fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                    with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                    cache_file_name=str(args.input.cache_study_path(len(study_dataset))) if args.input.use_cache_data else None,
                )
        if eval_dataset:
            with ProgIter(total=len(eval_dataset), desc='Preprocess  eval samples:', stream=logger, verbose=2) as manual_pbar:
                eval_dataset = eval_dataset.map(
                    preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                    fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                    with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                    cache_file_name=str(args.input.cache_eval_path(len(eval_dataset))) if args.input.use_cache_data else None,
                )
        if test_dataset:
            with ProgIter(total=len(test_dataset), desc='Preprocess  test samples:', stream=logger, verbose=2) as manual_pbar:
                test_dataset = test_dataset.map(
                    preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
                    fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
                    with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
                    cache_file_name=str(args.input.cache_test_path(len(test_dataset))) if args.input.use_cache_data else None,
                )
        logger.info("-" * 100)

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
        # model = model.to(training_args.device)
        model_optimizer = fabric.setup(model, optimizer)
        model: lightning.fabric.wrappers._FabricModule = model_optimizer[0]
        optimizer: lightning.fabric.wrappers._FabricOptimizer = model_optimizer[1]
        fabric.barrier()

        # Train model
        model.train()
        for epoch in range(args.learn.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{args.learn.num_train_epochs}")
            for step, batch in enumerate(train_dataloader, start=1):
                batch: BatchEncoding = batch
                batch.pop("idx")
                # batch = batch.to(training_args.device)
                outputs = model(**batch)
                loss = outputs.loss
                # loss.backward()
                fabric.backward(loss)
                if step % args.learn.grad_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                logger.info(f"Step {step}/{len(train_dataloader)}: loss={loss.item():.4f}, M={torch.cuda.max_memory_reserved() / math.pow(1024, 3):.0f}")


if __name__ == "__main__":
    app()
