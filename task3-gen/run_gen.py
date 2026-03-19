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


def complete_step1_build_train_args(
    model_preset: str,
    max_seq_length: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> GenerationTrainArguments:
    raise NotImplementedError(
        "TODO Step 1-1: build and return GenerationTrainArguments from the selected preset."
    )


def complete_step1_load_pretrained_components(model_name: str):
    raise NotImplementedError(
        "TODO Step 1-2: load and return the tokenizer/model pair for generation."
    )


def complete_step1_prepare_generation_datasets(
    args: GenerationTrainArguments,
    tokenizer: PreTrainedTokenizerFast,
):
    raise NotImplementedError(
        "TODO Step 1-3: download NSMC and return the corpus, train dataset, and validation dataset."
    )


# TODO Step 1:
# Understand how preset selection determines the pretrained model, output directory, and default port.
def build_train_args(model_preset: str, max_seq_length: int, batch_size: int, learning_rate: float, epochs: int, seed: int):
    return complete_step1_build_train_args(
        model_preset,
        max_seq_length,
        batch_size,
        learning_rate,
        epochs,
        seed,
    )


# TODO Step 1:
# Load the tokenizer and language model that match the chosen preset.
def load_pretrained_components(model_name: str):
    return complete_step1_load_pretrained_components(model_name)


# TODO Step 1:
# Load NSMC and prepare train/validation datasets so students can inspect the preprocessing boundary.
def prepare_generation_datasets(args: GenerationTrainArguments, tokenizer: PreTrainedTokenizerFast):
    return complete_step1_prepare_generation_datasets(args, tokenizer)


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


def complete_step2_train_loop(
    args: GenerationTrainArguments,
    model: GPT2LMHeadModel,
    train_dataset,
    val_dataset,
) -> None:
    raise NotImplementedError(
        "TODO Step 2-1: connect datasets, dataloaders, GenerationTask, and trainer.fit(...)."
    )


def complete_step2_generation_cases() -> list[tuple[str, dict[str, Any]]]:
    raise NotImplementedError(
        "TODO Step 2-2: return the decoding strategy list used for comparison."
    )


def complete_step3_load_finetuned_components(args: GenerationDeployArguments, device: torch.device):
    raise NotImplementedError(
        "TODO Step 3-1: load and return the fine-tuned model/tokenizer pair for serving."
    )


def complete_step3_build_inference_fn(model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerFast):
    raise NotImplementedError(
        "TODO Step 3-2: build and return the Flask inference function."
    )


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
        complete_step2_train_loop(args, model, train_dataset, val_dataset)

    for name, generate_kwargs in complete_step2_generation_cases():
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
    model, tokenizer = complete_step3_load_finetuned_components(args, device)
    inference_fn = complete_step3_build_inference_fn(model, tokenizer)

    app = get_web_service_app(
        inference_fn,
        template_folder=Path("templates").resolve(),
        server_page="serve_gen.html",
    )
    app.run(host=server_host, port=port or preset["port"])


if __name__ == "__main__":
    main()
