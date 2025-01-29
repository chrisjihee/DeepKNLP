import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Mapping, Any

import datasets
import numpy as np
import pandas as pd
import torch
import typer
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from datasets.utils.logging import set_verbosity as datasets_set_verbosity
from torch.utils.data import DataLoader
from typing_extensions import Annotated

import transformers
import transformers.utils.logging
from DeepKNLP.arguments import TrainingArgumentsForAccelerator, CustomDataArguments
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from DeepKNLP.gner_trainer import GNERTrainer
from accelerate import Accelerator
from accelerate import DeepSpeedPlugin
from accelerate.utils import gather_object
from chrisbase.data import AppTyper, JobTimer, Counter, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, set_verbosity_info, log_table, new_path, convert_all_events_in_dir
from chrisbase.time import from_timestamp, now_stamp
from chrisdata.ner import GenNERSampleWrapper
from progiter import ProgIter
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    BatchEncoding,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
    PrinterCallback,
)
from transformers.trainer_utils import has_length
from transformers.utils import is_torch_tf32_available, is_torch_bf16_gpu_available
from transformers.utils.logging import set_verbosity as transformers_set_verbosity

# Global settings
logger: logging.Logger = logging.getLogger("DeepKNLP")


class CustomProgressCallback(TrainerCallback):

    def __init__(self, output_path: str | Path, logging_ratio: float, eval_ratio: float, save_ratio: float):
        super().__init__()
        self.training_iter: Optional[ProgIter] = None
        self.prediction_iter: Optional[ProgIter] = None
        self.current_step: int = 0
        self.output_path: str | Path = output_path
        self.metrics_table: pd.DataFrame = pd.DataFrame()
        self.train_dataloader: Optional[DataLoader] = None
        self.epoch_optimization_steps: Optional[int] = None
        self.logging_ratio = logging_ratio if 0 < logging_ratio < 1 else -1
        self.eval_ratio = eval_ratio if 0 < eval_ratio < 1 else -1
        self.save_ratio = save_ratio if 0 < save_ratio < 1 else -1
        self.logging_steps = []
        self.save_steps = []
        self.eval_steps = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_dataloader: DataLoader = kwargs["train_dataloader"]
        self.epoch_optimization_steps = len(self.train_dataloader) // args.gradient_accumulation_steps
        if state.is_world_process_zero:
            self.training_iter = ProgIter(
                verbose=2,
                stream=LoggerWriter(logger),
                total=state.max_steps,
                desc="[TRAINING]",
            )
            self.training_iter.begin()
        self.current_step = 0
        if 0 < self.logging_ratio < 1:
            logging_times = math.floor(1 / self.logging_ratio)
            self.logging_steps = set([round(self.epoch_optimization_steps * self.logging_ratio * (i + 1)) for i in range(logging_times)])
            assert len(self.logging_steps) <= logging_times
            assert max(self.logging_steps) <= self.epoch_optimization_steps
        if 0 < self.save_ratio < 1:
            save_times = math.floor(1 / self.save_ratio)
            self.save_steps = set([round(self.epoch_optimization_steps * self.save_ratio * (i + 1)) for i in range(save_times)])
            assert len(self.save_steps) <= save_times
            assert max(self.save_steps) <= self.epoch_optimization_steps
        if 0 < self.eval_ratio < 1:
            eval_times = math.floor(1 / self.eval_ratio)
            self.eval_steps = set([round(self.epoch_optimization_steps * self.eval_ratio * (i + 1)) for i in range(eval_times)])
            assert len(self.eval_steps) <= eval_times
            assert max(self.eval_steps) <= self.epoch_optimization_steps

    def epoch_by_step(self, state: TrainerState):
        state.epoch = state.global_step / self.epoch_optimization_steps
        return state.epoch

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.training_iter is not None:
            self.training_iter.set_extra(f"| (Ep {round(self.epoch_by_step(state), 3):.3f})")
            self.training_iter.step(state.global_step - self.current_step)
        self.current_step = state.global_step
        if self.current_step in self.logging_steps:
            control.should_log = True
        if self.current_step in self.save_steps:
            control.should_save = True
        if self.current_step in self.eval_steps:
            control.should_evaluate = True

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = True

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = True
        if self.training_iter is not None:
            self.training_iter.end()
            self.training_iter = None

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                           eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_iter is None:
                self.prediction_iter = ProgIter(
                    verbose=2,
                    stream=LoggerWriter(logger),
                    total=len(eval_dataloader),
                    desc="[METERING]",
                )
                self.prediction_iter.begin()
            self.prediction_iter.step()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.prediction_iter is not None:
            self.prediction_iter.end()
            self.prediction_iter = None

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.prediction_iter is not None:
            self.prediction_iter.end()
            self.prediction_iter = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: Optional[Mapping[str, Any]] = None, exclude_keys=("epoch", "step"), **kwargs):
        metrics = {
            "step": state.global_step,
            "epoch": round(self.epoch_by_step(state), 3),
        }
        for k, v in logs.items():
            if k not in exclude_keys:
                metrics[k] = v
        new_metrics_row = pd.DataFrame([metrics])
        log_table(logger, new_metrics_row, tablefmt="plain", left=">> ", showindex=False)
        self.metrics_table = pd.concat([self.metrics_table, new_metrics_row], ignore_index=True)
        self.metrics_table.to_csv(self.output_path, index=False)


def update_progress(
        counter: Counter,
        rank: int = -1,
        pbar: Optional[ProgIter] = None,
):
    """
    Updates a global counter and an optional progress bar.
    """
    count = counter.inc()
    if pbar and rank == 0:
        # Ensure the progress bar does not exceed its total.
        pbar.step(min(count - pbar._iter_idx, pbar.total - pbar._iter_idx), force=count >= pbar.total)


def preprocess_row(
        row: LazyRow,
        rank: int,
        is_encoder_decoder: bool,
        max_source_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizerBase,
        counter: Counter,
        update: Optional[Callable[[Counter, int], None]] = None,
) -> BatchEncoding:
    """
    A unified preprocessing function for both encoder-decoder and decoder-only models.

    Args:
        row (LazyRow): A single row from the dataset.
        rank (int): The process rank (for distributed processing).
        is_encoder_decoder (bool): True for encoder-decoder models, False otherwise.
        max_source_length (int): Maximum token length for the input (source).
        max_target_length (int): Maximum token length for the output (target).
        tokenizer (PreTrainedTokenizerBase): Pretrained tokenizer.
        counter (Counter): A counter to track the number of processed samples.
        update (Callable, optional): A callback function used for updating progress.

    Returns:
        BatchEncoding: The tokenized sample.
    """
    sample = GenNERSampleWrapper.model_validate(row)

    # Construct decoder-only prompt and instruction text
    prompt_input = f"[INST] {sample.instance.instruction_inputs} [/INST]"
    full_instruction = f"{prompt_input} {sample.instance.prompt_labels}"

    def tokenize_encoder_decoder_train():
        model_inputs = tokenizer(
            text=sample.instance.instruction_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                text=sample.instance.prompt_labels,
                max_length=max_target_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    def tokenize_encoder_decoder_infer():
        model_inputs = tokenizer(
            text=sample.instance.instruction_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        return model_inputs

    def tokenize_decoder_only_train():
        model_inputs = tokenizer(
            text=full_instruction,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
            model_inputs["input_ids"].append(tokenizer.eos_token_id)
            model_inputs["attention_mask"].append(1)
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        prompt_tokens = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )["input_ids"]

        if prompt_tokens[-1] == tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]

        for i in range(len(prompt_tokens)):
            model_inputs["labels"][i] = -100

        return model_inputs

    def tokenize_decoder_only_infer():
        model_inputs = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
            model_inputs["input_ids"].pop()
            model_inputs["attention_mask"].pop()
        return model_inputs

    # Check if this row belongs to the training split
    is_train = (sample.split == "train")

    if is_encoder_decoder:
        if is_train:
            tokenized_sample = tokenize_encoder_decoder_train()
        else:
            tokenized_sample = tokenize_encoder_decoder_infer()
    else:
        if is_train:
            tokenized_sample = tokenize_decoder_only_train()
        else:
            tokenized_sample = tokenize_decoder_only_infer()

    # Update progress if a callback is provided
    if update:
        update(counter=counter, rank=rank)

    tokenized_sample["time_stamp"] = now_stamp()
    return tokenized_sample


def preprocess_dataset(
        file_path: Optional[str],
        dataset_name: str,
        is_encoder_decoder: bool,
        max_source_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizerBase,
        use_cache_data: bool,
        max_workers: int,
        local_rank: int,
        cache_path_func: Optional[Callable[[int], str]] = None,
) -> Optional[datasets.Dataset]:
    """
    Loads a JSON dataset from `file_path` and applies tokenization via `preprocess_row`.
    This function is intended to handle train, eval, or predict splits in a uniform way.

    Args:
        file_path (Optional[str]): Path to the dataset file. If None, returns None immediately.
        dataset_name (str): A descriptive name for logging purposes (e.g., "train_dataset").
        is_encoder_decoder (bool): Indicates if the model is an encoder-decoder type.
        max_source_length (int): Maximum token length for the input.
        max_target_length (int): Maximum token length for the output.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        use_cache_data (bool): If True, loads from cached results if available.
        max_workers (int): Number of worker processes to use for dataset.map().
        local_rank (int): The local rank for distributed training.
        cache_path_func (Callable[[int], str], optional): A function that, given the dataset length,
            returns the path to the cache file.

    Returns:
        Optional[datasets.Dataset]: The processed dataset, or None if `file_path` was None.
    """
    if file_path is None:
        return None

    # Load the raw dataset
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    logger.info(f"Loaded raw {dataset_name} (#={len(dataset)}): {file_path}")

    # Prepare a progress bar
    with ProgIter(
            verbose=2,
            stream=LoggerWriter(logger),
            total=len(dataset),
            desc=f"Preprocess {dataset_name}:"
    ) as pbar:
        # Disable the default progress bar in datasets
        datasets.disable_progress_bar()

        # Map function over the dataset
        counter = Counter(step=max_workers)
        dataset = dataset.map(
            batched=False,
            function=preprocess_row,
            with_rank=True,
            fn_kwargs={
                "is_encoder_decoder": is_encoder_decoder,
                "max_source_length": max_source_length,
                "max_target_length": max_target_length,
                "tokenizer": tokenizer,
                "counter": counter,
                "update": lambda *vs, **ws: update_progress(
                    *vs, **ws, pbar=pbar if local_rank == 0 else None
                ),
            },
            load_from_cache_file=use_cache_data,
            cache_file_name=cache_path_func(len(dataset)) if cache_path_func else None,
            num_proc=max_workers,
        )
        # Re-enable datasets progress bars
        datasets.enable_progress_bars()

    # Log the last timestamp recorded (if any) in the dataset
    if len(dataset) > 0 and "time_stamp" in dataset.column_names:
        timestamp_str = from_timestamp(max(dataset['time_stamp']))
        logger.info(f"Completed preprocessing for {dataset_name} at {timestamp_str}")

    return dataset


def compute_ner_metrics(dataset, preds, tokenizer, is_encoder_decoder, output_dir=None, save_prefix=None, save_suffix=None):
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if not is_encoder_decoder:
        match_pattern = "[/INST]"
        for i, preds in enumerate(decoded_preds):
            decoded_preds[i] = preds[preds.find(match_pattern) + len(match_pattern):].strip()

    all_examples = [example.copy() for example in dataset]
    for idx, decoded_pred in enumerate(decoded_preds):
        all_examples[idx]["prediction"] = decoded_pred

    results = compute_metrics(all_examples, tokenizer=tokenizer, detailed=False)
    if output_dir is not None and save_prefix is not None:
        suffix = f"_{save_suffix}" if save_suffix else ""
        file_name = f"{save_prefix}-text_generations{suffix}.jsonl"
        with open(os.path.join(output_dir, file_name), "w") as fout:
            for example in all_examples:
                fout.write(json.dumps(example) + "\n")
    return results


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [4]: https://huggingface.co/docs/accelerate/en/quicktour
# [5]: https://huggingface.co/docs/accelerate/en/basic_tutorials/migration
# [6]: https://huggingface.co/docs/accelerate/en/basic_tutorials/execution
# [7]: https://huggingface.co/docs/transformers/en/main_classes/logging
# [8]: https://huggingface.co/docs/transformers/en/main_classes/trainer
def main(
        # for NewProjectEnv
        local_rank: Annotated[int, typer.Option("--local_rank")] = -1,
        world_size: Annotated[int, typer.Option("--world_size")] = -1,
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[Optional[str], typer.Option("--run_version")] = "EAGLE-1B-debug",
        output_file: Annotated[str, typer.Option("--output_file")] = "train-metrics.csv",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "train-loggings.out",
        argument_file: Annotated[str, typer.Option("--argument_file")] = "train-arguments.json",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = 4,
        debugging: Annotated[bool, typer.Option("--debugging/--no-debugging")] = False,
        # for CustomDataArguments
        pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev-100.jsonl",
        pred_file: Annotated[str, typer.Option("--pred_file")] = "data/gner/zero-shot-test-100.jsonl",
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        ignore_pad_token_for_loss: Annotated[bool, typer.Option("--ignore_pad_token_for_loss/--no_ignore_pad_token_for_loss")] = True,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--no_use_cache_data")] = True,
        # for Seq2SeqTrainingArguments
        generation_max_length: Annotated[int, typer.Option("--generation_max_length")] = 640,
        report_to: Annotated[str, typer.Option("--report_to")] = "none",  # "tensorboard",  # tensorboard --bind_all --logdir output/GNER
        gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no_gradient_checkpointing")] = True,
        per_device_train_batch_size: Annotated[int, typer.Option("--per_device_train_batch_size")] = 8,
        gradient_accumulation_steps: Annotated[int, typer.Option("--gradient_accumulation_steps")] = 4,
        per_device_eval_batch_size: Annotated[int, typer.Option("--per_device_eval_batch_size")] = 8,
        eval_accumulation_steps: Annotated[int, typer.Option("--eval_accumulation_steps")] = 4,
        max_steps: Annotated[int, typer.Option("--max_steps")] = -1,
        num_train_epochs: Annotated[float, typer.Option("--num_train_epochs")] = 1.0,
        logging_ratio: Annotated[float, typer.Option("--logging_ratio")] = 1 / 10,
        logging_steps: Annotated[int, typer.Option("--logging_steps")] = -1,
        save_ratio: Annotated[float, typer.Option("--save_ratio")] = -1,
        save_steps: Annotated[int, typer.Option("--save_steps")] = -1,
        eval_ratio: Annotated[float, typer.Option("--eval_ratio")] = 1 / 3,
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = -1,
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 2e-5,
        # for DeepSpeed
        trainer_deepspeed: Annotated[str, typer.Option("--trainer_deepspeed")] = None,  # for deepspeed.launcher.runner
        accelerate_deepspeed: Annotated[bool, typer.Option("--accelerate_deepspeed")] = False,  # for accelerate.commands.launch
):
    # Setup project environment
    if local_rank < 0 and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 0 and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    stamp = now_stamp()
    stamp = sorted(gather_object([stamp]))[0]
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d.%H%M%S'),
        local_rank=local_rank,
        world_size=world_size,
        output_home=output_home,
        output_name=output_name,
        run_version=run_version,
        logging_level=logging.WARNING,
        logging_format=LoggingFormat.CHECK_20,
        output_file=new_path(output_file, post=from_timestamp(stamp, fmt='%m%d.%H%M%S')),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d.%H%M%S')),
        argument_file=new_path(argument_file, post=from_timestamp(stamp, fmt='%m%d.%H%M%S')),
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
    )

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin() if accelerate_deepspeed else None,
    )
    accelerator.wait_for_everyone()

    # Setup training arguments
    level_to_str = {n: s for s, n in transformers.utils.logging.log_levels.items()}
    log_level_str = level_to_str.get(env.logging_level, "passive")
    log_level_rep_str = level_to_str.get(env.logging_level + 10, "passive")
    args = TrainingArgumentsForAccelerator(
        env=env,
        data=CustomDataArguments(
            pretrained=pretrained,
            train_file=train_file,
            eval_file=eval_file,
            pred_file=pred_file,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            use_cache_data=use_cache_data,
        ),
        train=Seq2SeqTrainingArguments(
            disable_tqdm=True,
            predict_with_generate=True,
            generation_max_length=generation_max_length,
            remove_unused_columns=False,
            overwrite_output_dir=True,
            output_dir=str(env.output_dir),
            report_to=report_to,
            log_level=log_level_str,
            log_level_replica=log_level_rep_str,
            seed=env.random_seed,
            do_train=True,
            do_eval=True,
            gradient_checkpointing=gradient_checkpointing,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            logging_strategy="steps" if logging_steps > 0 else "epoch" if logging_steps == 0 else "no",
            save_strategy="steps" if save_steps > 0 else "epoch" if save_steps == 0 else "no",
            eval_strategy="steps" if eval_steps > 0 else "epoch" if eval_steps == 0 else "no",
            logging_steps=logging_ratio if 0 < logging_ratio < 1 else logging_steps if logging_steps >= 1 else sys.maxsize,
            save_steps=save_ratio if 0 < save_ratio < 1 else save_steps if save_steps >= 1 else sys.maxsize,
            eval_steps=eval_ratio if 0 < eval_ratio < 1 else eval_steps if eval_steps >= 1 else sys.maxsize,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.04,
            weight_decay=0.0,
            tf32=is_torch_tf32_available(),
            bf16=is_torch_bf16_gpu_available(),
            bf16_full_eval=is_torch_bf16_gpu_available(),
            local_rank=env.local_rank,
            deepspeed=trainer_deepspeed,
        ),
    )
    args.env.local_rank = args.train.local_rank

    # Setup logging
    process_log_level = args.train.get_process_log_level()
    args.env.setup_logger(process_log_level)
    datasets_set_verbosity(process_log_level)
    transformers_set_verbosity(process_log_level)
    set_verbosity_info("c10d-NullHandler-default")
    if accelerator.is_main_process:
        set_verbosity_info(
            "transformers.trainer",
            "chrisbase.data",
            "DeepKNLP",
        )

    # Log on each process
    logger.info(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {args.train.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {args.train.fp16 or args.train.bf16}"
    )
    accelerator.wait_for_everyone()

    # Set random seed
    set_seed(args.train.seed)
    torch.set_float32_matmul_precision('high')

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}",
            rt=1, rb=1, rc='=', verbose=True, args=args,
    ):
        accelerator.wait_for_everyone()

        # Load config and tokenizer
        config = AutoConfig.from_pretrained(
            args.data.pretrained,
            trust_remote_code=True,
            use_cache=not args.train.gradient_checkpointing,
        )
        is_encoder_decoder = config.is_encoder_decoder

        # Initialize tokenizer
        if is_encoder_decoder:
            tokenizer = AutoTokenizer.from_pretrained(
                args.data.pretrained,
                trust_remote_code=True,
            )
        else:  # https://github.com/Vision-CAIR/MiniGPT-4/issues/129
            tokenizer = AutoTokenizer.from_pretrained(
                args.data.pretrained,
                trust_remote_code=True,
                add_eos_token=True,
                add_bos_token=True,
                padding_side="left",
            )
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.data.pretrained,
                from_tf=bool(".ckpt" in str(args.data.pretrained)),
                config=config,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.data.pretrained,
                from_tf=bool(".ckpt" in str(args.data.pretrained)),
                config=config,
                trust_remote_code=True,
            )
        logger.info(f"type(model)={type(model)}")
        model.generation_config.pad_token_id = tokenizer.pad_token_id  # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        logger.info(f"model.generation_config.pad_token_id={model.generation_config.pad_token_id}")

        # Preprocess training dataset (if do_train)
        train_dataset = preprocess_dataset(
            file_path=args.data.train_file if args.train.do_train else None,
            dataset_name="train_dataset",
            is_encoder_decoder=is_encoder_decoder,
            max_source_length=args.data.max_source_length,
            max_target_length=args.data.max_target_length,
            tokenizer=tokenizer,
            use_cache_data=args.data.use_cache_data,
            max_workers=args.env.max_workers,
            local_rank=args.train.local_rank,
            cache_path_func=args.data.cache_train_path,
        )
        accelerator.wait_for_everyone()

        # Preprocess evaluation dataset (if do_eval)
        eval_dataset = preprocess_dataset(
            file_path=args.data.eval_file if args.train.do_eval else None,
            dataset_name="eval_dataset",
            is_encoder_decoder=is_encoder_decoder,
            max_source_length=args.data.max_source_length,
            max_target_length=args.data.max_target_length,
            tokenizer=tokenizer,
            use_cache_data=args.data.use_cache_data,
            max_workers=args.env.max_workers,
            local_rank=args.train.local_rank,
            cache_path_func=args.data.cache_eval_path,
        )
        accelerator.wait_for_everyone()

        # Preprocess prediction dataset (if do_predict)
        pred_dataset = preprocess_dataset(
            file_path=args.data.pred_file if args.train.do_predict else None,
            dataset_name="pred_dataset",
            is_encoder_decoder=is_encoder_decoder,
            max_source_length=args.data.max_source_length,
            max_target_length=args.data.max_target_length,
            tokenizer=tokenizer,
            use_cache_data=args.data.use_cache_data,
            max_workers=args.env.max_workers,
            local_rank=args.train.local_rank,
            cache_path_func=args.data.cache_pred_path,
        )
        accelerator.wait_for_everyone()

        # Data collator
        label_pad_token_id = -100 if args.data.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForGNER(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8 if args.train.fp16 else None,
            label_pad_token_id=label_pad_token_id,
            return_tensors="pt",
        )

        # Initialize trainer
        progress_callback = CustomProgressCallback(
            output_path=args.env.output_dir / args.env.output_file,
            logging_ratio=logging_ratio,
            eval_ratio=eval_ratio,
            save_ratio=save_ratio,
        )
        trainer = GNERTrainer(
            args=args.train,
            model=model,
            callbacks=[progress_callback],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ner_metrics if args.train.predict_with_generate else None,
            is_encoder_decoder=is_encoder_decoder,
        )
        trainer.remove_callback(PrinterCallback)
        if accelerator.is_main_process:
            if trainer.accelerator.state.deepspeed_plugin:
                logger.info(
                    f"Using deepspeed configuration:\n"
                    f"{json.dumps(trainer.accelerator.state.deepspeed_plugin.deepspeed_config, indent=2)}"
                )

        # Train
        accelerator.wait_for_everyone()
        if args.train.do_train:
            train_result = trainer.train()
            logger.info(f"Train result: {train_result}")
            convert_all_events_in_dir(args.train.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    AppTyper.run(main)
