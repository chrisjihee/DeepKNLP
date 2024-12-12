import json
import logging
import math
import re
from typing import Any, Callable, Iterable

import datasets
import numpy as np
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
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BatchEncoding, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer import get_model_param_count, nested_concat, nested_numpify, denumpify_detensorize
from typing_extensions import Annotated

from DeepKNLP.arguments import NewProjectEnv, TrainingArguments
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from chrisbase.data import AppTyper, JobTimer, Counter
from chrisbase.io import LoggingFormat, set_verbosity_warning, set_verbosity_info, set_verbosity_error, to_table_lines
from chrisbase.util import shuffled, to_dataframe
from chrisdata.ner import GenNERSampleWrapper
from progiter import ProgIter

# Global settings
env = None
app = AppTyper(name="Generative NER",
               help="Generative Named Entity Recognition (NER) using Hugging Face Transformers.")
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
        random_seed: Annotated[int, typer.Option("--random_seed")] = 1234,
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
        "root",
        "DeepSpeed",
        "c10d-NullHandler-default",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.distributed",
    )
    torch.set_float32_matmul_precision('high')
    # logger.info(f"Start running {app.info.name} with env={env}")


# Reference
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py
# [4]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py
# [5]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_methods.html
# [6]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_args.html
# [7]: https://lightning.ai/docs/fabric/2.4.0/guide/
# [8]: https://lightning.ai/docs/fabric/2.4.0/advanced/model_parallel/fsdp.html
@app.command()
def train(
        # input
        pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",  # TODO: "google/flan-t5-small", "etri-lirs/egpt-1.3b-preview",
        # train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/pile-ner.jsonl",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        # eval_file: Annotated[str, typer.Option("--eval_file")] = None,
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev.jsonl",
        # eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-debug.jsonl",
        test_file: Annotated[str, typer.Option("--test_file")] = None,
        # test_file: Annotated[str, typer.Option("--test_file")] = "data/gner/zero-shot-test.jsonl"
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,  # TODO: 512, 640
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,  # TODO: 512, 640
        max_generation_length: Annotated[int, typer.Option("--max_generation_length")] = 1280,  # TODO: 512, 640
        max_train_samples: Annotated[int, typer.Option("--max_train_samples")] = -1,  # TODO: 256, -1
        max_eval_samples: Annotated[int, typer.Option("--max_eval_samples")] = 128,  # TODO: 128, 256, -1
        max_test_samples: Annotated[int, typer.Option("--max_test_samples")] = -1,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--use_fresh_data")] = False,
        # learn
        run_name: Annotated[str, typer.Option("--run_name")] = "task2-nerG",
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        num_train_epochs: Annotated[int, typer.Option("--num_train_epochs")] = 1,  # TODO: -> 1, 2, 3, 4, 5, 6
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 2e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,
        device_type: Annotated[str, typer.Option("--device_type")] = "gpu",  # TODO: -> gpu, cpu, mps
        device_idx: Annotated[int, typer.Option("--device_idx")] = 0,  # TODO: -> 0, 4
        num_device: Annotated[int, typer.Option("--num_device")] = 4,  # TODO: -> 1, 2, 4, 8
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 8,
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = 24,  # TODO: -> 12, 16, 24, 32
        train_batch: Annotated[int, typer.Option("--train_batch")] = 2,
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 16,
        strategy: Annotated[str, typer.Option("--strategy")] = "fsdp",  # TODO: -> ddp, fsdp, deepspeed
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
            eval_file=eval_file,
            test_file=test_file,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            max_generation_length=max_generation_length,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            use_cache_data=use_cache_data,
        ),
        learn=TrainingArguments.LearnOption(
            run_name=run_name,
            output_home=output_home,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device_type=device_type,
            device_idx=device_idx,
            num_device=num_device,
            precision=precision,
            grad_steps=grad_steps,
            eval_steps=eval_steps,
            train_batch=train_batch,
            infer_batch=infer_batch,
            strategy=strategy,
            ds_stage=ds_stage,
            ds_offload=ds_offload,
            fsdp_shard=fsdp_shard,
            fsdp_offload=fsdp_offload,
        ),
    )

    # Setup fabric and accelerator
    basic_logger = CSVLogger(args.learn.output_home, args.learn.run_name, flush_logs_every_n_steps=1)
    visual_logger = TensorBoardLogger(args.learn.output_home, args.learn.run_name, basic_logger.version)  # tensorboard --logdir output --bind_all
    fabric = Fabric(
        accelerator=args.learn.device_type,
        precision=args.learn.precision,
        strategy=args.learn.strategy_inst,
        devices=args.learn.devices,
        loggers=[basic_logger, visual_logger],
    )
    fabric.launch()
    fabric.barrier()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.learn.grad_steps,
    )
    fabric.flush = do_nothing
    fabric.write = lambda x, *y, **z: info_or_debug_r(fabric, x, *y, **z)
    fabric.print = lambda x, *y, **z: info_or_debug(fabric, x, *y, **z)
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size

    # Setup logger
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)
    args.env.setup_logger(logging_home=basic_logger.log_dir)
    if fabric.is_global_zero:
        transformers.logging.set_verbosity_info()
        datasets.utils.logging.set_verbosity_warning()
        set_verbosity_info(
            "lightning",
        )
        set_verbosity_warning(
            "transformers.generation.configuration_utils",
            "transformers.tokenization_utils_base",
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "lightning.fabric.utilities.seed",
            "DeepSpeed",
        )
        set_verbosity_error(
            "transformers.generation.utils",
        )
    else:
        transformers.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()
        set_verbosity_error(
            "lightning",
            "DeepSpeed",
        )
    transformers.logging.disable_progress_bar()
    datasets.utils.logging.disable_progress_bar()
    # logger.info(f"[local_rank={fabric.local_rank}] args.env={args.env}({type(args.env)})")

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args if fabric.is_global_zero else None, verbose=fabric.is_global_zero,
            mute_warning="lightning.fabric.loggers.csv_logs"
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.env.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        using_decoder_only_model = not config.is_encoder_decoder

        if using_decoder_only_model:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                pretrained,
                # cache_dir=model_args.cache_dir,
                # use_fast=model_args.use_fast_tokenizer,
                # revision=model_args.model_revision,
                # token=model_args.token,
                # trust_remote_code=model_args.trust_remote_code,
                padding_side="left",
                add_eos_token=True,
                add_bos_token=True,
            )
        else:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained)
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token

        if using_decoder_only_model:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, config=config)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained, config=config)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        fabric.barrier()
        fabric.print(f"Num tokens: {len(tokenizer):,}, Size embedding={model.get_input_embeddings().weight.shape[0]:,}")

        # Load dataset
        train_dataset: Dataset | None = None
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
                fabric.print(f"Use {args.input.train_file} as train dataset: {len(train_dataset):,} samples")
        if args.input.eval_file:
            with fabric.rank_zero_first():
                eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.eval_file),
                                                     cache_dir=str(args.input.cache_eval_dir))
                if args.input.max_eval_samples > 0:
                    whole_indices = shuffled(range(len(eval_dataset)), seed=args.env.random_seed)
                    eval_dataset = eval_dataset.select(whole_indices[:args.input.max_eval_samples])
                eval_dataset = eval_dataset.add_column("idx", range(len(eval_dataset)))
                fabric.print(f"Use {args.input.eval_file} as eval dataset: {len(eval_dataset):,} samples")
        if args.input.test_file:
            with fabric.rank_zero_first():
                test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.test_file),
                                                     cache_dir=str(args.input.cache_test_dir))
                if args.input.max_test_samples > 0:
                    whole_indices = shuffled(range(len(test_dataset)), seed=args.env.random_seed)
                    test_dataset = test_dataset.select(whole_indices[:args.input.max_test_samples])
                test_dataset = test_dataset.add_column("idx", range(len(test_dataset)))
                fabric.print(f"Use {args.input.test_file} as test dataset: {len(test_dataset):,} samples")
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

        # Set data collator
        label_pad_token_id = -100
        data_collator: DataCollatorForGNER = DataCollatorForGNER(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,  # if training_args.fp16 else None,
            label_pad_token_id=label_pad_token_id,
            return_tensors="pt",
            feature_names=("input_ids", "attention_mask", "labels", "idx"),
        )

        # Set data loader
        if train_dataset:
            train_dataloader = DataLoader(
                train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                batch_size=args.learn.train_batch,
                collate_fn=data_collator,
                drop_last=False,
            )
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
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

        # Set optimizer
        no_decay = ("bias", "layer_norm.weight",)
        optimizer: Optimizer = torch.optim.AdamW(
            lr=args.learn.learning_rate,
            params=[
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.learn.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ],
        )
        model, optimizer = fabric.setup(model, optimizer)
        # model: lightning.fabric.wrappers._FabricModule = model
        # optimizer: lightning.fabric.wrappers._FabricOptimizer = optimizer
        model.mark_forward_method("generate")

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

            results = compute_metrics(all_examples, tokenizer=tokenizer, average_key="[Average]")
            if save_prefix is not None:
                outfile_path = args.env.logging_home / f"{save_prefix}_text_generations{'_' + save_suffix if save_suffix else ''}.jsonl"
                with outfile_path.open("w") as fout:
                    for example in all_examples:
                        if save_keys:
                            example = {key: example[key] for key in save_keys if key in example}
                        fout.write(json.dumps(example) + "\n")
            return results

        def print_metrics(origin: dict[str, float], filter_key="f1", data_key="dataset", score_key="score", score_factor: float = 100.0, floatfmt=".1f") -> dict[str, float]:
            filtered = {
                re.sub("[-_]", "/", key.removesuffix(f'_{filter_key}')): origin[key]
                for key in sorted(origin.keys())
                if filter_key in key
            }
            if filtered:
                data = to_dataframe(filtered, columns=[data_key, score_key])
                data[score_key] = data[score_key] * score_factor
                for x in to_table_lines(data, floatfmt=floatfmt):
                    fabric.print(x)
            return origin

        # Train loop
        fabric.barrier()
        if train_dataloader:
            epoch_optimization_steps = math.ceil(len(train_dataloader) / args.learn.grad_steps)
            epoch_per_step = 1.0 / epoch_optimization_steps
            train_losses = []
            global_step = 0
            global_epoch = 0.0
            total_epochs = args.learn.num_train_epochs
            gen_kwargs = {
                "num_beams": 1,
                "max_length": args.input.max_generation_length,
            }

            fabric.print(f"===== Running training =====")
            fabric.print(f"  Num Epochs = {args.learn.num_train_epochs:,}")
            fabric.print(f"  Num Examples = {len(train_dataset):,}")
            fabric.print(f"  Num Parameters = {get_model_param_count(model, trainable_only=True):,}")
            fabric.print(f"  World size = {args.env.world_size:,}")
            fabric.print(f"  Grad acc. steps = {args.learn.grad_steps:,}")
            fabric.print(f"  Evaluation steps = {args.learn.eval_steps:,}")
            fabric.print(f"  Train batch size = {args.learn.train_batch:,}")
            fabric.print(f"  Infer batch size = {args.learn.infer_batch:,}")
            fabric.print(f"  Total batch size = {args.env.world_size * args.learn.train_batch * args.learn.grad_steps:,}")
            fabric.print(f"  Epoch optim steps = {epoch_optimization_steps:,}")
            torch.cuda.reset_peak_memory_stats()

            for epoch in range(args.learn.num_train_epochs):
                fabric.print("-" * 100)
                with ProgIter(total=epoch_optimization_steps, desc=f'Training [{global_epoch:.2f}/{total_epochs}]:', stream=fabric, verbose=2, time_thresh=3.0) as train_pbar:
                    for train_loop_i, train_batch in enumerate(train_dataloader, start=1):
                        # Forward pass
                        model.train()
                        outputs = model(**train_batch)
                        fabric.backward(outputs.loss)
                        train_losses.append(outputs.loss.item())
                        if train_loop_i % args.learn.grad_steps != 0 and train_loop_i != len(train_dataloader):
                            continue

                        # Backward pass
                        optimizer.step()
                        optimizer.zero_grad()

                        # Update train progress
                        global_step += 1
                        global_epoch += epoch_per_step
                        fabric.barrier()
                        train_loss = torch.cat(fabric.all_gather(train_losses)).mean().item()
                        train_losses.clear()
                        train_pbar.set_extra(f"| train_loss={train_loss:.4f}")
                        train_pbar.set_description(f'Training [{global_epoch:.2f}/{total_epochs}]:', refresh=False)
                        train_pbar.step(force=train_loop_i == 1 or train_loop_i >= len(train_dataloader))

                        # Define metrics
                        metrics: dict[str, Any] = {
                            "step": round(fabric.all_gather(torch.tensor(global_step * 1.0)).mean().item()),
                            "epoch": round(fabric.all_gather(torch.tensor(global_epoch)).mean().item(), 4),
                            "train_loss": train_loss,
                        }

                        # Validation loop
                        if eval_dataloader and (
                                train_loop_i == len(train_dataloader) or
                                (args.learn.eval_steps > 0 and global_step % args.learn.eval_steps) == 0
                        ):
                            train_pbar.refresh()
                            model.eval()
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
                                        eval_pbar.step(force=eval_loop_i == 1 or eval_loop_i >= len(eval_dataloader))
                                    eval_logits = nested_numpify(eval_logits)
                                    eval_indexs = nested_numpify(eval_indexs).tolist()
                                    metrics.update(compute_ner_metrics(eval_logits, eval_indexs, eval_dataset, save_prefix="eval", save_suffix=f"{global_epoch:.1f}",
                                                                       save_keys=["id", "dataset", "split", "prediction", "instance", "label_list", "input_ids", "attention_mask"]))

                        # Print and save metrics
                        metrics = print_metrics(denumpify_detensorize(metrics))
                        fabric.log_dict(metrics=metrics, step=metrics["step"])

                max_memory = torch.cuda.max_memory_allocated() / math.pow(1024, 3)
                fabric.print(f"{train_pbar.desc} max_memory={max_memory:.1f}GB")
                fabric.barrier()


if __name__ == "__main__":
    app()
