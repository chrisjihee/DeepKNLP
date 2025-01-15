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
        logging_format=LoggingFormat.CHECK_48,
        # logging_format=LoggingFormat.DEBUG_48,
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
    pred_data_path: str = field(default=None)
    max_source_length: int = field(default=640)
    max_target_length: int = field(default=640)
    ignore_pad_token_for_loss: bool = field(default=True)


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
        overwrite_cache: Annotated[bool, typer.Option("--overwrite_cache/--load_from_cache")] = False,
        train_data_path: Annotated[str, typer.Option("--train_json_file")] = "data/gner/zero-shot-train.jsonl",
        eval_data_path: Annotated[str, typer.Option("--eval_json_file")] = "data/gner/zero-shot-test-min.jsonl",
        # eval_data_path: Annotated[str, typer.Option("--eval_json_file")] = "data/gner/zero-shot-dev.jsonl",
        pred_data_path: Annotated[str, typer.Option("--test_json_file")] = "data/gner/zero-shot-test-min.jsonl",
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        generation_max_length: Annotated[int, typer.Option("--generation_max_length")] = 1280,
        ignore_pad_token_for_loss: Annotated[bool, typer.Option("--ignore_pad_token_for_loss")] = True,
        # Seq2SeqTrainingArguments
        deepspeed: Annotated[str, typer.Option("--deepspeed")] = "configs/deepspeed_configs/deepspeed_zero1_llama.json",
):
    # Setup training arguments
    training_args = Seq2SeqTrainingArgumentsForGNER(
        model_name_or_path=model_name_or_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        pred_data_path=pred_data_path,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        generation_max_length=generation_max_length,
        ignore_pad_token_for_loss=ignore_pad_token_for_loss,
        remove_unused_columns=False,
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
    ran = random.Random(training_args.seed)
    raw_datasets = {}

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        training_args.model_name_or_path,
        trust_remote_code=True,
    )
    is_encoder_decoder = config.is_encoder_decoder
    if is_encoder_decoder:
        tokenizer = AutoTokenizer.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
        )
    else:  # https://github.com/Vision-CAIR/MiniGPT-4/issues/129
        tokenizer = AutoTokenizer.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=True,
            padding_side="left",
        )
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
        # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
    MODEL_CLASS = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    model = MODEL_CLASS.from_pretrained(
        training_args.model_name_or_path,
        from_tf=bool(".ckpt" in training_args.model_name_or_path),
        config=config,
        trust_remote_code=True,
    )
    logger.info(f"type(model)={type(model)}")
    model.generation_config.pad_token_id = tokenizer.pad_token_id  # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
    logger.info(f"model.generation_config.pad_token_id={model.generation_config.pad_token_id}")

    # Preprocess the datasets
    def preprocess_function(example):
        # remove pairs where at least one record is None
        inference = example['split'] != "train"
        if is_encoder_decoder:
            model_inputs = tokenizer(
                text=example['instance']['instruction_inputs'],
                max_length=training_args.max_source_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            if not inference:
                model_inputs["labels"] = tokenizer(
                    text_target=example['instance']['prompt_labels'],
                    max_length=training_args.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )['input_ids']
        else:
            prompt = f"[INST] {example['instance']['instruction_inputs']} [/INST]"
            full_instruction = f"{prompt} {example['instance']['prompt_labels']}"
            max_length = training_args.max_source_length + training_args.max_target_length
            if inference:
                model_inputs = tokenizer(
                    text=prompt,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )
                # Remove the last token if it is an eos token
                if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                    model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                    model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]
            else:
                model_inputs = tokenizer(
                    text=full_instruction,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )

                if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
                    model_inputs["input_ids"].append(tokenizer.eos_token_id)
                    model_inputs["attention_mask"].append(1)

                model_inputs["labels"] = model_inputs["input_ids"].copy()

                # Find the prompt length
                prompt = tokenizer(
                    text=prompt,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )["input_ids"]

                # Remove the last token if it is an eos token
                if prompt[-1] == tokenizer.eos_token_id:
                    prompt = prompt[:-1]

                if len(prompt) > len(model_inputs["labels"]):
                    raise ValueError(
                        f"Prompt is longer than the input, something went wrong. Prompt: {prompt}, input:"
                        f" {model_inputs['input_ids']}"
                    )

                for i in range(len(prompt)):
                    model_inputs["labels"][i] = -100

        return model_inputs

    if training_args.do_train:
        assert training_args.train_data_path is not None, "Need to provide train_data_path"
        train_dataset = load_dataset("json", data_files=training_args.train_data_path, split="train")
        logger.info(f"Use {training_args.train_data_path} as train_dataset(#={len(train_dataset)})")
        with training_args.main_process_first(desc="train_dataset map preprocessing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=env.max_workers,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on train_dataset",
            )

    if training_args.do_eval:
        assert training_args.eval_data_path is not None, "Need to provide eval_data_path"
        eval_dataset = load_dataset("json", data_files=training_args.eval_data_path, split="train")
        logger.info(f"Use {training_args.eval_data_path} as eval_dataset(#={len(eval_dataset)})")
        with training_args.main_process_first(desc="eval_dataset map preprocessing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=env.max_workers,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on eval_dataset",
            )

    if training_args.do_predict:
        assert training_args.pred_data_path is not None, "Need to provide pred_data_path"
        pred_dataset = load_dataset("json", data_files=training_args.pred_data_path, split="train")
        logger.info(f"Use {training_args.pred_data_path} as pred_dataset(#={len(pred_dataset)})")
        with training_args.main_process_first(desc="pred_dataset map preprocessing"):
            pred_dataset = pred_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=env.max_workers,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on pred_dataset",
            )

    # Construct a data collator
    label_pad_token_id = -100 if training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForGNER(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        label_pad_token_id=label_pad_token_id,
        return_tensors="pt",
    )

    def compute_ner_metrics(dataset, preds, save_prefix=None, save_suffix=None):
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if not is_encoder_decoder:
            match_pattern = "[/INST]"
            for i, preds in enumerate(decoded_preds):
                decoded_preds[i] = preds[preds.find(match_pattern) + len(match_pattern):].strip()

        all_examples = [example.copy() for example in dataset]
        for idx, decoded_pred in enumerate(decoded_preds):
            all_examples[idx]["prediction"] = decoded_pred

        results = compute_metrics(all_examples, tokenizer=tokenizer)
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_text_generations{'_' + save_suffix if save_suffix else ''}.jsonl"), "w") as fout:
                for example in all_examples:
                    fout.write(json.dumps(example) + "\n")
        return results

    # Initialize our trainer
    trainer = GNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,  # FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `GNERTrainer.__init__`. Use `processing_class` instead.
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics if training_args.predict_with_generate else None,
    )

    # Do training
    if training_args.do_train:
        train_result = trainer.train()


if __name__ == "__main__":
    app()
