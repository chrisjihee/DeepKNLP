import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
import torch
import typer
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from datasets.utils.logging import set_verbosity as datasets_set_verbosity
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
from chrisbase.io import LoggingFormat, set_verbosity_info, make_parent_dir
from chrisbase.io import new_path, files, tb_events_to_csv
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
    Seq2SeqTrainingArguments,
    ProgressCallback,
    set_seed,
)
from transformers.utils import is_torch_tf32_available, is_torch_bf16_gpu_available
from transformers.utils.logging import set_verbosity as transformers_set_verbosity

# Global settings
logger: logging.Logger = logging.getLogger("DeepKNLP")


def convert_all_events_in_dir(log_dir: str | Path):
    """
    logdir 아래의 모든 event 파일을 찾아서 CSV로 변환.
    이벤트 파일마다 따로 CSV를 만듭니다.
    """
    input_files = os.path.join(log_dir, "**/events.out.tfevents.*")
    for input_file in files(input_files):
        if not input_file.name.endswith(".csv"):
            output_file = input_file.with_name(input_file.name + ".csv")
            logger.info(f"Convert {input_file} to csv")
            tb_events_to_csv(input_file, output_file)


def read_state(path: str | Path) -> dict:
    return json.loads(make_parent_dir(path).read_text())


def write_state(path: str | Path, state: dict):
    make_parent_dir(path).write_text(json.dumps(state))


# Define progress update function
def update_progress(
        counter: Counter,
        rank: int = -1,
        pbar: Optional[ProgIter] = None,
):
    count = counter.inc()
    if pbar and rank == 0:
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
    A unified preprocessing function for encoder-decoder models and decoder-only models.

    Args:
        row (LazyRow): A single row from the dataset.
        rank (int): Process rank (for distributed processing).
        is_encoder_decoder (bool): True if the model is an encoder-decoder model, False otherwise.
        max_source_length (int): Maximum token length for the input (source).
        max_target_length (int): Maximum token length for the output (target).
        tokenizer (PreTrainedTokenizerBase): Pretrained tokenizer.
        counter: A counter to track the number of processed samples.
        update (Callable, optional): A callback function used for progress updates.
            If provided, it will be called to update the batch encoding object.

    Returns:
        BatchEncoding: The tokenized sample.
    """
    # Convert the JSON row to a pydantic model object
    sample = GenNERSampleWrapper.model_validate(row)

    # Text components for the decoder-only scenario
    prompt_input = f"[INST] {sample.instance.instruction_inputs} [/INST]"
    full_instruction = f"{prompt_input} {sample.instance.prompt_labels}"

    def tokenize_encoder_decoder_train():
        """
        Preprocessing function for an encoder-decoder model in the train split.
        """
        # Tokenize the instruction_inputs as the source
        model_inputs = tokenizer(
            text=sample.instance.instruction_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        # Tokenize prompt_labels as the labels
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
        """
        Preprocessing function for an encoder-decoder model in the eval/predict split.
        """
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
        """
        Preprocessing function for a decoder-only model in the train split.
        """
        # Tokenize the entire instruction (prompt + labels)
        model_inputs = tokenizer(
            text=full_instruction,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        # If the last token is not the EOS token, append it
        if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
            model_inputs["input_ids"].append(tokenizer.eos_token_id)
            model_inputs["attention_mask"].append(1)

        # Set labels by copying the input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        # Determine the length of the prompt tokens (for masking)
        prompt_tokens = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )["input_ids"]

        # If the last token in prompt_tokens is the EOS token, remove it
        if prompt_tokens[-1] == tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]

        # Replace label tokens corresponding to the prompt with -100
        for i in range(len(prompt_tokens)):
            model_inputs["labels"][i] = -100

        return model_inputs

    def tokenize_decoder_only_infer():
        """
        Preprocessing function for a decoder-only model in the eval/predict split.
        """
        model_inputs = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        # If the last token is the EOS token, remove it
        if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
            model_inputs["input_ids"].pop()
            model_inputs["attention_mask"].pop()

        return model_inputs

    # Determine whether this is a training sample
    is_train = (sample.split == "train")

    # Branch by is_encoder_decoder
    if is_encoder_decoder:
        # Encoder-decoder model
        if is_train:
            tokenized_sample = tokenize_encoder_decoder_train()
        else:
            tokenized_sample = tokenize_encoder_decoder_infer()
    else:
        # Decoder-only model
        if is_train:
            tokenized_sample = tokenize_decoder_only_train()
        else:
            tokenized_sample = tokenize_decoder_only_infer()

    # If the update callback is provided, call it (e.g., to update progress bars, etc.)
    if update:
        update(counter=counter, rank=rank)

    return tokenized_sample


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
        run_version: Annotated[Optional[str], typer.Option("--run_version")] = "EAGLE-1B-debug",  # try None
        logging_file: Annotated[str, typer.Option("--logging_file")] = "train-messages.out",
        argument_file: Annotated[str, typer.Option("--argument_file")] = "train-arguments.json",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = 4,
        debugging: Annotated[bool, typer.Option("--debugging/--no-debugging")] = False,
        # for CustomDataArguments
        pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
        train_file: Annotated[str, typer.Option("--train_file")] = "data/gner/zero-shot-train.jsonl",
        eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-test-min.jsonl",
        # eval_file: Annotated[str, typer.Option("--eval_file")] = "data/gner/zero-shot-dev.jsonl",
        pred_file: Annotated[str, typer.Option("--pred_file")] = "data/gner/zero-shot-test-min.jsonl",
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        ignore_pad_token_for_loss: Annotated[bool, typer.Option("--ignore_pad_token_for_loss/--no_ignore_pad_token_for_loss")] = True,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--no_use_cache_data")] = False,  # TODO: True
        # for Seq2SeqTrainingArguments
        generation_max_length: Annotated[int, typer.Option("--generation_max_length")] = 1280,
        report_to: Annotated[str, typer.Option("--report_to")] = "tensorboard",  # tensorboard --bind_all --logdir output/GNER/EAGLE-1B-debug/runs
        gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no_gradient_checkpointing")] = True,
        gradient_accumulation_steps: Annotated[int, typer.Option("--gradient_accumulation_steps")] = 4,
        per_device_train_batch_size: Annotated[int, typer.Option("--per_device_train_batch_size")] = 8,
        per_device_eval_batch_size: Annotated[int, typer.Option("--per_device_eval_batch_size")] = 8,
        num_train_epochs: Annotated[float, typer.Option("--num_train_epochs")] = 0.5,
        # for DeepSpeed
        trainer_deepspeed: Annotated[str, typer.Option("--trainer_deepspeed")] = None,
        accelerate_deepspeed: Annotated[bool, typer.Option("--accelerate_deepspeed")] = False,
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
        logging_format=LoggingFormat.CHECK_48,
        logging_level=logging.WARNING,
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
        project_dir=env.output_dir,
        log_with=report_to,
    )
    logger.warning(f"accelerator.is_main_process={accelerator.is_main_process}")
    if accelerator.is_main_process:
        for k in os.environ.keys():
            logger.info(f"os.environ[{k}]={os.environ[k]}")
        logger.info(f"accelerator={accelerator} / accelerator.state={accelerator.state}")

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
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            logging_strategy="steps",
            logging_steps=10,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="no",  # TODO: "epoch",
            learning_rate=2e-5,
            warmup_ratio=0.04,
            weight_decay=0.,
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

    # Log on each process the small summary:
    logger.info(
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
            use_cache=not args.train.gradient_checkpointing,
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

        if args.train.do_train:
            assert args.data.train_file is not None, "Need to provide train_data_path"
            train_dataset = load_dataset("json", data_files=str(args.data.train_file), split="train")
            logger.info(f"Load raw train_dataset(#={len(train_dataset)}): {args.data.train_file}")
            with ProgIter(total=len(train_dataset), desc="Preprocess train samples:", stream=logger, verbose=2) as pbar:
                datasets.disable_progress_bar()
                train_dataset = train_dataset.map(
                    function=preprocess_row, batched=False, with_rank=True,
                    fn_kwargs={
                        "is_encoder_decoder": is_encoder_decoder,
                        "max_source_length": args.data.max_source_length,
                        "max_target_length": args.data.max_target_length,
                        "tokenizer": tokenizer,
                        "counter": Counter(step=args.env.max_workers),
                        "update": lambda *vs, **ws: update_progress(
                            *vs, **ws, pbar=pbar if args.train.local_rank == 0 else None
                        ),
                    },
                    load_from_cache_file=args.data.use_cache_data,
                    cache_file_name=args.data.cache_train_path(len(train_dataset)),
                    num_proc=args.env.max_workers,
                )
                datasets.enable_progress_bars()
            # if state_path:
            #     if read_state(state_path)["cnt"] == 0 and args.data.cache_train_files(len(train_dataset)):
            #         logger.info(f"Load processed train_dataset: {args.data.cache_train_files(len(train_dataset))[0]}")
            accelerator.wait_for_everyone()
            exit(0)

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

        # Construct a data collator
        label_pad_token_id = -100 if args.data.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForGNER(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8 if args.train.fp16 else None,
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
                with open(os.path.join(args.train.output_dir, f"{save_prefix}_text_generations{'_' + save_suffix if save_suffix else ''}.jsonl"), "w") as fout:
                    for example in all_examples:
                        fout.write(json.dumps(example) + "\n")
            return results

        # Initialize our trainer
        trainer = GNERTrainer(
            args=args.train,
            model=model,
            callbacks=[ProgressCallback(max_str_len=300)],
            train_dataset=train_dataset if args.train.do_train else None,
            eval_dataset=eval_dataset if args.train.do_eval else None,
            processing_class=tokenizer,  # FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `GNERTrainer.__init__`. Use `processing_class` instead.
            data_collator=data_collator,
            compute_metrics=compute_ner_metrics if args.train.predict_with_generate else None,
        )
        if accelerator.is_main_process:
            logger.warning(f"trainer.accelerator={trainer.accelerator} / trainer.accelerator.state={trainer.accelerator.state}")

        # Do training
        if args.train.do_train:
            train_result = trainer.train()
            logger.info(f"train_result={train_result}")
            convert_all_events_in_dir(env.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    AppTyper.run(main)
