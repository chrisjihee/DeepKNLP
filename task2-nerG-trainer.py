import random
import logging

import datasets
import transformers
import typer
from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import LoggingFormat
from chrisbase.util import shuffled
from datasets import Dataset, load_dataset
from lightning.fabric.loggers import CSVLogger
from transformers import (
    Seq2SeqTrainingArguments,
    set_seed, PretrainedConfig, AutoConfig, PreTrainedTokenizerBase, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
)
from typing_extensions import Annotated

from DeepKNLP.arguments import TrainingArguments

# Global settings
env = None
app = AppTyper(name="Generative NER", help="Generative Named Entity Recognition (NER) using Transformer. [trainer]")
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
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/eagle-3b-preview",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "etri-lirs/egpt-1.3b-preview",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-small",  # 80M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-base",  # 250M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-large",  # 780M
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xl",  # 3B
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "google/flan-t5-xxl",  # 11B
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "microsoft/Phi-3.5-mini-instruct",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-2-7b-hf",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-1B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.2-3B",
        # pretrained: Annotated[str, typer.Option("--pretrained")] = "meta-llama/Llama-3.1-8B",
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
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 5e-5,
        weight_decay: Annotated[float, typer.Option("--weight_decay")] = 0.0,  # TODO: utilize lr_scheduler
        train_batch: Annotated[int, typer.Option("--train_batch")] = 1,  # TODO: -> 1, 2, 4, 8
        infer_batch: Annotated[int, typer.Option("--infer_batch")] = 10,  # TODO: -> 10, 20, 40
        grad_steps: Annotated[int, typer.Option("--grad_steps")] = 1,  # TODO: -> 2, 4, 8, 10, 20, 40
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = 10,  # TODO: -> 20, 40
        num_device: Annotated[int, typer.Option("--num_device")] = 1,  # TODO: -> 4, 8
        device_idx: Annotated[int, typer.Option("--device_idx")] = 0,  # TODO: -> 0, 4
        device_type: Annotated[str, typer.Option("--device_type")] = "gpu",  # TODO: -> gpu, cpu, mps
        precision: Annotated[str, typer.Option("--precision")] = "bf16-mixed",  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: Annotated[str, typer.Option("--strategy")] = "deepspeed",  # TODO: -> ddp, fsdp, deepspeed
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

    # Setup logger
    basic_logger = CSVLogger(args.learn.output_home, args.learn.output_name, args.learn.run_version, flush_logs_every_n_steps=1)
    training_args = Seq2SeqTrainingArguments(
        output_dir=basic_logger.log_dir,
        log_level=args.env.logging_level,
    )
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


if __name__ == "__main__":
    app()
