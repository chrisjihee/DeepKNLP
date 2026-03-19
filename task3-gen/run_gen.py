"""Sentence Generation lab entrypoint.

Students work in this single file cumulatively.

Step 1:
- load the pretrained model, tokenizer, and corpus
- confirm dataset preparation and dry-run generation

Step 2:
- run fine-tuning and compare several generation strategies

Step 3:
- load the fine-tuned checkpoint and serve the generator with Flask

Reference:
- https://ratsgo.github.io/nlpbook/docs/generation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import typer
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from ratsnlp import nlpbook
from ratsnlp.nlpbook.generation import (
    GenerationTask,
    GenerationTrainArguments,
    GenerationDeployArguments,
    NsmcCorpus,
    GenerationDataset,
    get_web_service_app,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
main = typer.Typer()

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "kogpt2": {
        "model_name": "skt/kogpt2-base-v2",
        "output_dir": "output/nsmc-gen/train_gen-by-kogpt2",
        "port": 9001,
    },
    "kogpt-trinity": {
        "model_name": "skt/ko-gpt-trinity-1.2B-v0.5",
        "output_dir": "output/nsmc-gen/train_gen-by-kogpt-trinity",
        "port": 9002,
    },
    "polyglot-ko": {
        "model_name": "EleutherAI/polyglot-ko-1.3b",
        "output_dir": "output/nsmc-gen/train_gen-by-polyglot-ko-1.3b",
        "port": 9003,
    },
}


def get_preset(model_preset: str) -> dict[str, Any]:
    if model_preset not in MODEL_PRESETS:
        raise typer.BadParameter(
            f"Unknown model preset: {model_preset}. Choose from {', '.join(MODEL_PRESETS)}"
        )
    return MODEL_PRESETS[model_preset]


def build_train_args(model_preset: str, max_seq_length: int, batch_size: int, learning_rate: float, epochs: int, seed: int):
    # TODO Step 1-1:
    # Build GenerationTrainArguments directly from the selected preset in this function.
    # preset = get_preset(model_preset)
    # return GenerationTrainArguments(...)
    raise NotImplementedError(
        "TODO Step 1-1: build GenerationTrainArguments from the chosen preset here."
    )


def load_pretrained_components(model_name: str):
    # TODO Step 1-2:
    # Load the tokenizer and pretrained language model that match the chosen preset.
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(...)
    # model = GPT2LMHeadModel.from_pretrained(...)
    raise NotImplementedError(
        "TODO Step 1-2: load the pretrained tokenizer and model here."
    )


def prepare_generation_datasets(args: GenerationTrainArguments, tokenizer: PreTrainedTokenizerFast):
    # TODO Step 1-3:
    # Download NSMC and build the train/validation datasets in this function.
    # Korpora.fetch(...)
    # corpus = NsmcCorpus()
    # train_dataset = GenerationDataset(..., mode="train")
    # val_dataset = GenerationDataset(..., mode="test")
    raise NotImplementedError(
        "TODO Step 1-3: prepare the generation corpus and datasets here."
    )


def build_dataloaders(args: GenerationTrainArguments, train_dataset, val_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    return train_dataloader, val_dataloader


def decode_generation(model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast, prompt: str, **generate_kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(input_ids, **generate_kwargs)
    return tokenizer.decode([token.item() for token in generated_ids[0]])


@main.command()
def step1(
    model_preset: str = typer.Option("kogpt2"),
    prompt: str = typer.Option("안녕하세요"),
    max_seq_length: int = typer.Option(32),
    batch_size: int = typer.Option(4),
    learning_rate: float = typer.Option(5e-5),
    epochs: int = typer.Option(1),
    seed: int = typer.Option(7),
):
    """Load the model, tokenizer, and datasets, then run a dry-run generation."""
    args = build_train_args(model_preset, max_seq_length, batch_size, learning_rate, epochs, seed)
    tokenizer, model = load_pretrained_components(args.pretrained_model_name)
    _, train_dataset, val_dataset = prepare_generation_datasets(args, tokenizer)

    typer.echo(f"model={args.pretrained_model_name}")
    typer.echo(f"train_examples={len(train_dataset)}")
    typer.echo(f"valid_examples={len(val_dataset)}")
    typer.echo(f"sample_prompt={prompt}")
    typer.echo("dry_run_generation=")
    typer.echo(
        decode_generation(
            model,
            tokenizer,
            prompt,
            do_sample=False,
            min_length=10,
            max_length=40,
        )
    )


@main.command()
def step2(
    model_preset: str = typer.Option("kogpt2"),
    prompt: str = typer.Option("안녕하세요"),
    max_seq_length: int = typer.Option(32),
    batch_size: int = typer.Option(4),
    learning_rate: float = typer.Option(5e-5),
    epochs: int = typer.Option(1),
    seed: int = typer.Option(7),
    skip_train: bool = typer.Option(False, help="If true, skip fine-tuning and only run generation experiments."),
):
    """Fine-tune the model and compare several generation strategies."""
    args = build_train_args(model_preset, max_seq_length, batch_size, learning_rate, epochs, seed)
    tokenizer, model = load_pretrained_components(args.pretrained_model_name)
    _, train_dataset, val_dataset = prepare_generation_datasets(args, tokenizer)

    # TODO Step 2:
    # Connect datasets, dataloaders, GenerationTask, and the trainer to run fine-tuning.
    if not skip_train:
        raise NotImplementedError(
            "TODO Step 2-1: connect dataloaders, GenerationTask, trainer, and trainer.fit(...) here."
        )

    # TODO Step 2-2:
    # Define the decoding strategy list in place before iterating.
    # generation_cases = [("greedy", {...}), ...]
    raise NotImplementedError(
        "TODO Step 2-2: define the generation strategy list here."
    )

    for name, generate_kwargs in generation_cases:
        typer.echo(f"[{name}]")
        typer.echo(
            decode_generation(
                model,
                tokenizer,
                prompt,
                **generate_kwargs,
            )
        )


@main.command()
def step3(
    model_preset: str = typer.Option("kogpt2"),
    server_host: str = typer.Option("0.0.0.0"),
    port: int | None = typer.Option(None),
):
    """Load the fine-tuned checkpoint and start the Flask demo."""
    preset = get_preset(model_preset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = GenerationDeployArguments(
        pretrained_model_name=preset["model_name"],
        downstream_model_dir=preset["output_dir"],
    )

    # TODO Step 3:
    # Load the fine-tuned checkpoint and connect generation to the Flask web service.
    # pretrained_model_config = GPT2Config.from_pretrained(...)
    # model = GPT2LMHeadModel(pretrained_model_config)
    # fine_tuned_model_ckpt = torch.load(...)
    # model.load_state_dict(...)
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(...)
    raise NotImplementedError(
        "TODO Step 3-1: load the fine-tuned model checkpoint and tokenizer here."
    )

    # TODO Step 3:
    # Build the Flask inference function in place so the service wiring stays visible.
    # def inference_fn(...):
    #     result = decode_generation(...)
    #     return {"result": result}
    raise NotImplementedError(
        "TODO Step 3-2: build the Flask inference function here."
    )

    app = get_web_service_app(
        inference_fn,
        template_folder=Path("templates").resolve(),
        server_page="serve_gen.html",
    )
    app.run(host=server_host, port=port or preset["port"])


if __name__ == "__main__":
    main()
