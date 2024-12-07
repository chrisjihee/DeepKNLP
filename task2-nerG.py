import json
import logging
import os
import time
from typing import Any, Callable

import datasets
import lightning
import numpy as np
import torch
import transformers
import typer
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from lightning.fabric import Fabric
from progiter import ProgIter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BatchEncoding, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from typing_extensions import Annotated

from DeepKNLP.arguments import NewProjectEnv, TrainingArguments
from DeepKNLP.gner_collator import DataCollatorForGNER
from DeepKNLP.gner_evaluator import compute_metrics
from chrisbase.data import AppTyper, JobTimer, Counter
from chrisbase.io import LoggingFormat, setup_unit_logger, set_verbosity_warning, set_verbosity_info, set_verbosity_error
from chrisdata.ner import GenNERSampleWrapper

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
        logging_home: Annotated[str, typer.Option("--logging_home")] = "output/task2-nerG",
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
        message_format=LoggingFormat.CHECK_32,
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
# [2]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_methods.html
# [3]: https://lightning.ai/docs/fabric/2.4.0/api/fabric_args.html
# [4]: https://lightning.ai/docs/fabric/2.4.0/guide/
# [5]: https://lightning.ai/docs/fabric/2.4.0/advanced/model_parallel/fsdp.html
@app.command()
def train(
        # input
        pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",  # TODO: "google/flan-t5-small", "etri-lirs/egpt-1.3b-preview",
        train_data: Annotated[str, typer.Option("--train_data")] = "data/gner/zero-shot-train.jsonl",  # TODO: "data/gner/pile-ner.jsonl"
        eval_data: Annotated[str, typer.Option("--eval_data")] = None,  # TODO: "data/gner/zero-shot-dev.jsonl"
        test_data: Annotated[str, typer.Option("--test_data")] = None,  # TODO: "data/gner/zero-shot-test.jsonl"
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 512,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 512,
        max_train_samples: Annotated[int, typer.Option("--max_train_samples")] = 1000,  # TODO: -1
        max_eval_samples: Annotated[int, typer.Option("--max_eval_samples")] = -1,
        max_test_samples: Annotated[int, typer.Option("--max_test_samples")] = -1,
        num_prog_samples: Annotated[int, typer.Option("--num_prog_samples")] = 5000,
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--use_fresh_data")] = False,
        # learn
        output_home: Annotated[str, typer.Option("--output_home")] = "output",
        num_train_epochs: Annotated[int, typer.Option("--num_train_epochs")] = 1,  # TODO: -> 2, 3
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 2e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,
        accelerator: Annotated[str, typer.Option("--accelerator")] = "gpu",  # TODO: -> gpu, cpu, mps
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        gpu_index: Annotated[int, typer.Option("--gpu_index")] = 4,
        num_device: Annotated[int, typer.Option("--num_device")] = 1,  # TODO: -> 1, 2, 4
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 8,
        train_batch: Annotated[int, typer.Option("--train_batch")] = 4,
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 32,
        strategy: Annotated[str, typer.Option("--strategy")] = "ddp",  # TODO: -> ddp, fsdp, deepspeed
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
            train_path=train_data,
            eval_path=eval_data,
            test_path=test_data,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            num_prog_samples=num_prog_samples,
            use_cache_data=use_cache_data,
        ),
        learn=TrainingArguments.LearnOption(
            output_home=output_home,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            accelerator=accelerator,
            precision=precision,
            gpu_index=gpu_index,
            num_device=num_device,
            grad_steps=grad_steps,
            train_batch=train_batch,
            infer_batch=infer_batch,
            strategy=strategy,
            ds_stage=ds_stage,
            ds_offload=ds_offload,
            fsdp_shard=fsdp_shard,
            fsdp_offload=fsdp_offload,
        ),
    )

    # Setup fabric
    fabric = Fabric(
        accelerator=args.learn.accelerator,
        precision=args.learn.precision,
        strategy=args.learn.strategy_inst,
        devices=args.learn.devices,
    )
    fabric.launch()
    fabric.barrier()
    fabric.flush = do_nothing
    fabric.write = lambda x, *y, **z: info_or_debug_r(fabric, x, *y, **z)
    fabric.print = lambda x, *y, **z: info_or_debug(fabric, x, *y, **z)
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size

    # Setup logger
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)
    args.env.setup_logger()
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
            "DeepSpeed",
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
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.env.random_seed)

        # Load model
        config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        using_decoder_only_model = not config.is_encoder_decoder
        fabric.print(f"type(config)={type(config)} - {isinstance(config, PretrainedConfig)}")

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
        fabric.print(f"type(tokenizer)={type(tokenizer)} - {isinstance(tokenizer, PreTrainedTokenizerFast)}, len(tokenizer)={len(tokenizer)}")

        if using_decoder_only_model:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, config=config)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained, config=config)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        fabric.barrier()
        fabric.print(f"type(model)={type(model)} - {isinstance(model, PreTrainedModel)}")
        fabric.print(f"embedding_size={model.get_input_embeddings().weight.shape[0]}")
        fabric.print("-" * 100)

        # Load dataset
        train_dataset: Dataset | None = None
        eval_dataset: Dataset | None = None
        test_dataset: Dataset | None = None
        if args.input.train_path:
            with fabric.rank_zero_first():
                train_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                      data_files=str(args.input.train_path),
                                                      cache_dir=str(args.input.cache_train_dir))
                if args.input.max_train_samples > 0:
                    train_dataset = train_dataset.select(range(min(len(train_dataset), args.input.max_train_samples)))
                fabric.print(f"Use {args.input.train_path} as train dataset: {len(train_dataset):,} samples")

                # TODO: Remove after testing
                prog = ProgIter(train_dataset, total=len(train_dataset), desc='Preprocess train samples:', verbose=2, stream=fabric)
                for _ in prog:
                    prog.format_message_parts()
                    time.sleep(0.01)

        if args.input.eval_path:
            with fabric.rank_zero_first():
                eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.eval_path),
                                                     cache_dir=str(args.input.cache_eval_dir))
                if args.input.max_eval_samples > 0:
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.input.max_eval_samples)))
                fabric.print(f"Use {args.input.eval_path} as eval dataset: {len(eval_dataset):,} samples")
        if args.input.test_path:
            with fabric.rank_zero_first():
                test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
                                                     data_files=str(args.input.test_path),
                                                     cache_dir=str(args.input.cache_test_dir))
                if args.input.max_test_samples > 0:
                    test_dataset = test_dataset.select(range(min(len(test_dataset), args.input.max_test_samples)))
                fabric.print(f"Use {args.input.eval_path} as test dataset: {len(test_dataset):,} samples")
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
            pre, cnt = counter.val(), counter.inc()
            if (cnt >= pbar.total or any(i % num_prog_samples == 0 for i in range(pre + 1, cnt + 1))) and rank == 0:
                pbar.step(inc=min(cnt - pbar._iter_idx, pbar.total - pbar._iter_idx))
                # fabric.print(pbar.format_message().rstrip())
            return res

        # # Preprocess dataset
        # if train_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(train_dataset), desc='Preprocess train samples: ', verbose=3,
        #                       stream=fabric, freq=1, adjust=False, rel_adjust_limit=0.0) as manual_pbar:
        #             train_dataset = train_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
        #                 cache_file_name=str(args.input.cache_train_path(len(train_dataset))) if args.input.use_cache_data else None,
        #             )
        # if eval_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(eval_dataset), desc='Preprocess eval samples', verbose=0) as manual_pbar:
        #             eval_dataset = eval_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
        #                 cache_file_name=str(args.input.cache_eval_path(len(eval_dataset))) if args.input.use_cache_data else None,
        #             )
        # if test_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(test_dataset), desc='Preprocess test samples', verbose=0) as manual_pbar:
        #             test_dataset = test_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.input.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.input.use_cache_data,
        #                 cache_file_name=str(args.input.cache_test_path(len(test_dataset))) if args.input.use_cache_data else None,
        #             )
        # fabric.barrier()
        # fabric.print("-" * 100)
        #
        # # Set data collator
        # label_pad_token_id = -100
        # data_collator: DataCollatorForGNER = DataCollatorForGNER(
        #     tokenizer,
        #     model=model,
        #     padding=True,
        #     pad_to_multiple_of=8,  # if training_args.fp16 else None,
        #     label_pad_token_id=label_pad_token_id,
        #     return_tensors="pt",
        # )
        #
        # # Set data loader
        # if train_dataset:
        #     train_dataloader = DataLoader(
        #         train_dataset,
        #         shuffle=True,
        #         collate_fn=data_collator,
        #         batch_size=args.learn.train_batch,
        #     )
        #     train_dataloader = fabric.setup_dataloaders(train_dataloader)
        # if eval_dataset:
        #     eval_dataloader = DataLoader(
        #         eval_dataset,
        #         collate_fn=data_collator,
        #         batch_size=args.learn.infer_batch,
        #     )
        #     eval_dataloader = fabric.setup_dataloaders(eval_dataloader)
        # if test_dataset:
        #     test_dataloader = DataLoader(
        #         test_dataset,
        #         collate_fn=data_collator,
        #         batch_size=args.learn.infer_batch,
        #     )
        #     test_dataloader = fabric.setup_dataloaders(test_dataloader)
        #
        # # Set optimizer
        # no_decay = ("bias", "layer_norm.weight",)
        # optimizer: Optimizer = torch.optim.AdamW(
        #     lr=args.learn.learning_rate,
        #     params=[
        #         {
        #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #             "weight_decay": args.learn.weight_decay,
        #         },
        #         {
        #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #             "weight_decay": 0.0,
        #         },
        #     ],
        # )
        # fabric.print(f"type(optimizer)={type(optimizer)} - {isinstance(optimizer, Optimizer)}")
        # model, optimizer = fabric.setup(model, optimizer)
        # fabric.print(f"type(optimizer)={type(optimizer)} - {isinstance(optimizer, lightning.fabric.wrappers._FabricOptimizer)}")
        # fabric.print(f"type(model)={type(model)} - {isinstance(model, lightning.fabric.wrappers._FabricModule)}")
        #
        # # Define compute metrics function
        # def compute_ner_metrics(dataset, preds, save_prefix=None):
        #     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #     if using_decoder_only_model:
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
        #         with open(os.path.join(args.env.output_home, f"{save_prefix}_text_generations.jsonl"), "w") as fout:
        #             for example in all_examples:
        #                 fout.write(json.dumps(example) + "\n")
        #     return results
        #
        # # Train loop
        # fabric.barrier()
        # fabric.print("*" * 100)
        # global_step = 0
        # global_epoch = 0.0
        # epoch_per_step = 1.0 / len(train_dataloader)
        # torch.cuda.reset_peak_memory_stats()
        # if train_dataloader:
        #     for epoch in range(args.learn.num_train_epochs):
        #         fabric.print("-" * 100)
        #         with ProgIter(total=len(train_dataloader), desc=f'Training [epoch={epoch}]:',
        #                       verbose=2, file=fabric,  # time_thresh=2.0,
        #
        #                       # file=open(os.devnull, 'w'),
        #                       ) as pbar:
        #             for i, batch in enumerate(train_dataloader, start=1):
        #                 model.train()
        #                 is_accumulating = i % args.learn.grad_steps != 0
        #
        #                 outputs = model(**batch)
        #                 loss = outputs.loss
        #                 fabric.backward(loss)
        #                 pbar.set_extra(f" [loss={loss.item():.4f}] ")
        #                 if not is_accumulating:
        #                     optimizer.step()
        #                     optimizer.zero_grad()
        #                     global_step += 1
        #                 pbar.step()
        #                 global_epoch += epoch_per_step
        #         fabric.print(f"max_memory={torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f}MB, final_loss={loss.item():.6f}")


if __name__ == "__main__":
    app()
