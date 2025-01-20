import random
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import typer

from DeepKNLP.arguments import TrainingArgumentsForAccelerator, CustomDataArguments
from accelerate import Accelerator, DeepSpeedPlugin
from datasets import load_dataset
from typing_extensions import Annotated

from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from DeepKNLP.gner_trainer import GNERTrainer
from accelerate.utils import broadcast, broadcast_object_list, gather_object, gather
from chrisbase.data import AppTyper, NewProjectEnv, JobTimer
from chrisbase.io import LoggingFormat, set_verbosity_warning, set_verbosity_info, new_path
from chrisbase.time import from_timestamp, now, now_stamp
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import is_torch_tf32_available, is_torch_bf16_gpu_available
from transformers.utils.logging import set_verbosity as transformers_set_verbosity
from datasets.utils.logging import set_verbosity as datasets_set_verbosity

# Global settings
app: AppTyper = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer.")
env: Optional[NewProjectEnv] = None
logger: logging.Logger = logging.getLogger("DeepKNLP")


# Class for training arguments
@dataclass
class Seq2SeqTrainingArgumentsForGNER(Seq2SeqTrainingArguments):
    model_name_or_path: str = field(default=None)
    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    pred_data_path: str = field(default=None)
    max_source_length: int = field(default=640)
    max_target_length: int = field(default=640)
    ignore_pad_token_for_loss: bool = field(default=True)


@app.callback()
def init(
        local_rank: Annotated[int, typer.Option("--local_rank")] = -1,
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        run_version: Annotated[Optional[str], typer.Option("--run_version")] = "EAGLE-1B-debug",  # try None
        logging_file: Annotated[str, typer.Option("--logging_file")] = "train-messages.out",
        argument_file: Annotated[str, typer.Option("--argument_file")] = "train-arguments.json",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = 4,
        debugging: Annotated[bool, typer.Option("--debugging/--no-debugging")] = False,
):
    global env
    stamp = now_stamp(delay=random.randint(1, 500) / 200.0)  # TODO: remove delay
    logger.warning(f"stamp={stamp} / {from_timestamp(stamp)}")
    stamp = sorted(gather(torch.tensor(stamp, dtype=torch.float64)).tolist())[0]
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d.%H%M%S'),
        local_rank=local_rank,
        output_home=output_home,
        output_name=output_name,
        run_version=run_version,
        logging_format=LoggingFormat.CHECK_48,
        logging_level="info",
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d.%H%M%S')),
        argument_file=new_path(argument_file, post=from_timestamp(stamp, fmt='%m%d.%H%M%S')),
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
    )


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
        # for CustomDataArguments
        pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-test-min.jsonl",
        # eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev.jsonl",
        pred_file: Annotated[str, typer.Option("--pred_file")] = "data/gner/zero-shot-test-min.jsonl",
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        ignore_pad_token_for_loss: Annotated[bool, typer.Option("--ignore_pad_token_for_loss/--no_ignore_pad_token_for_loss")] = True,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--no_use_cache_data")] = False,
        # for Seq2SeqTrainingArguments
        generation_max_length: Annotated[int, typer.Option("--generation_max_length")] = 1280,
        report_to: Annotated[str, typer.Option("--report_to")] = "tensorboard",
        gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no_gradient_checkpointing")] = True,
        gradient_accumulation_steps: Annotated[int, typer.Option("--gradient_accumulation_steps")] = 4,
        per_device_train_batch_size: Annotated[int, typer.Option("--per_device_train_batch_size")] = 8,
        per_device_eval_batch_size: Annotated[int, typer.Option("--per_device_eval_batch_size")] = 8,
        num_train_epochs: Annotated[float, typer.Option("--num_train_epochs")] = 0.5,
        # for DeepSpeedPlugin
        ds_stage: Annotated[int, typer.Option("--ds_stage")] = 1,  # TODO: -> 1, 2, 3
        ds_config: Annotated[str, typer.Option("--ds_config")] = "configs/deepspeed/deepspeed_zero1_llama.json",
):
    # Setup training arguments
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
            predict_with_generate=True,
            generation_max_length=generation_max_length,
            remove_unused_columns=False,
            overwrite_output_dir=True,
            output_dir=str(env.output_dir),
            report_to=report_to,
            log_level=env.logging_level,
            seed=env.random_seed,
            do_train=True,
            do_eval=True,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
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
            local_rank=env.local_rank,
        ),
    )
    args.env.local_rank = args.train.local_rank

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin(zero_stage=ds_stage, hf_ds_config=ds_config),
        project_dir=args.env.output_dir,
        log_with=args.train.report_to,
    )

    # Setup logging
    process_log_level = args.train.get_process_log_level()
    args.env.setup_logger(process_log_level)
    datasets_set_verbosity(process_log_level + 10)
    transformers_set_verbosity(process_log_level)
    # set_verbosity_info("c10d-NullHandler-default")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}"
        f", distributed training: {args.train.parallel_mode.value == 'distributed'}"
        f", 16-bits training: {args.train.fp16 or args.train.bf16}"
    )
    accelerator.wait_for_everyone()

    # Set seed before initializing model.
    set_seed(args.train.seed)
    torch.set_float32_matmul_precision('high')

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            verbose=True,
            args=args,
    ):
        accelerator.wait_for_everyone()

        # Load pretrained model and tokenizer
        config = AutoConfig.from_pretrained(
            args.data.pretrained,
            trust_remote_code=True,
        )
        is_encoder_decoder = config.is_encoder_decoder
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
        MODEL_CLASS = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
        model = MODEL_CLASS.from_pretrained(
            args.data.pretrained,
            from_tf=bool(".ckpt" in str(args.data.pretrained)),
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
                    max_length=args.data.max_source_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=True,
                )
                if not inference:
                    model_inputs["labels"] = tokenizer(
                        text_target=example['instance']['prompt_labels'],
                        max_length=args.data.max_target_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None,
                        add_special_tokens=True,
                    )['input_ids']
            else:
                prompt = f"[INST] {example['instance']['instruction_inputs']} [/INST]"
                full_instruction = f"{prompt} {example['instance']['prompt_labels']}"
                max_length = args.data.max_source_length + args.data.max_target_length
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

        if args.train.do_train:
            assert args.data.train_file is not None, "Need to provide train_data_path"
            train_dataset = load_dataset("json", data_files=str(args.data.train_file), split="train")
            logger.info(f"Use {args.data.train_file} as train_dataset(#={len(train_dataset)})")
            with args.train.main_process_first(desc="train_dataset map preprocessing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.use_cache_data,
                    desc="Running tokenizer on train_dataset",
                )
            accelerator.wait_for_everyone()

        if args.train.do_eval:
            assert args.data.eval_file is not None, "Need to provide eval_data_path"
            eval_dataset = load_dataset("json", data_files=str(args.data.eval_file), split="train")
            logger.info(f"Use {args.data.eval_file} as eval_dataset(#={len(eval_dataset)})")
            with args.train.main_process_first(desc="eval_dataset map preprocessing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.use_cache_data,
                    desc="Running tokenizer on eval_dataset",
                )
            accelerator.wait_for_everyone()

        if args.train.do_predict:
            assert args.data.pred_file is not None, "Need to provide pred_data_path"
            pred_dataset = load_dataset("json", data_files=str(args.data.pred_file), split="train")
            logger.info(f"Use {args.data.pred_file} as pred_dataset(#={len(pred_dataset)})")
            with args.train.main_process_first(desc="pred_dataset map preprocessing"):
                pred_dataset = pred_dataset.map(
                    preprocess_function,
                    batched=False,
                    num_proc=args.env.max_workers,
                    load_from_cache_file=args.data.use_cache_data,
                    desc="Running tokenizer on pred_dataset",
                )
            accelerator.wait_for_everyone()

    # # Construct a data collator
    # label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForGNER(
    #     tokenizer,
    #     model=model,
    #     padding=True,
    #     pad_to_multiple_of=8 if args.fp16 else None,
    #     label_pad_token_id=label_pad_token_id,
    #     return_tensors="pt",
    # )
    #
    # def compute_ner_metrics(dataset, preds, save_prefix=None, save_suffix=None):
    #     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     if not is_encoder_decoder:
    #         match_pattern = "[/INST]"
    #         for i, preds in enumerate(decoded_preds):
    #             decoded_preds[i] = preds[preds.find(match_pattern) + len(match_pattern):].strip()
    #
    #     all_examples = [example.copy() for example in dataset]
    #     for idx, decoded_pred in enumerate(decoded_preds):
    #         all_examples[idx]["prediction"] = decoded_pred
    #
    #     results = compute_metrics(all_examples, tokenizer=tokenizer)
    #     if save_prefix is not None:
    #         with open(os.path.join(args.output_dir, f"{save_prefix}_text_generations{'_' + save_suffix if save_suffix else ''}.jsonl"), "w") as fout:
    #             for example in all_examples:
    #                 fout.write(json.dumps(example) + "\n")
    #     return results
    #
    # # Initialize our trainer
    # trainer = GNERTrainer(
    #     args=args,
    #     model=model,
    #     train_dataset=train_dataset if args.do_train else None,
    #     eval_dataset=eval_dataset if args.do_eval else None,
    #     processing_class=tokenizer,  # FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `GNERTrainer.__init__`. Use `processing_class` instead.
    #     data_collator=data_collator,
    #     compute_metrics=compute_ner_metrics if args.predict_with_generate else None,
    # )
    #
    # # Do training
    # if args.do_train:
    #     train_result = trainer.train()
    #     logger.info(f"train_result={train_result}")

    accelerator.end_training()


if __name__ == "__main__":
    app()
