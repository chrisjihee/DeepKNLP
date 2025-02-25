import os

import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ratsnlp import nlpbook
from ratsnlp.nlpbook.generation import GenerationTask
from ratsnlp.nlpbook.generation import GenerationTrainArguments
from ratsnlp.nlpbook.generation import NsmcCorpus, GenerationDataset
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    args = GenerationTrainArguments(
        pretrained_model_name="skt/ko-gpt-trinity-1.2B-v0.5",
        downstream_model_dir="output/nsmc-gen/train_gen-by-kogpt-trinity",
        downstream_corpus_name="nsmc",
        max_seq_length=32,
        batch_size=32 if torch.cuda.is_available() else 4,
        learning_rate=5e-5,
        epochs=3,
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=7,
    )

    nlpbook.set_seed(args)

    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=args.force_download,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.pretrained_model_name,
        eos_token="</s>",
    )

    corpus = NsmcCorpus()
    train_dataset = GenerationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    val_dataset = GenerationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name
    )

    task = GenerationTask(model, args)
    trainer = nlpbook.get_trainer(args)

    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
