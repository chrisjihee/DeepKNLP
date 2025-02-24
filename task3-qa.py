import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Dict, Mapping, Any

import torch
import typer
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, make_dir, files
from chrisbase.util import mute_tqdm_cls, tupled
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.qa import QADataset, KorQuADCorpus
from DeepKNLP.helper import CheckpointSaver, epsilon, data_collator, fabric_barrier
from DeepKNLP.metrics import accuracy

logger = logging.getLogger(__name__)
main = AppTyper()


class KorQuADModel(LightningModule):
    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        super().__init__()
        self.args: TrainerArguments | TesterArguments | ServerArguments = args
        self.data: KorQuADCorpus = KorQuADCorpus(args)

        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained,
        )
        self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )
        self.lang_model: PreTrainedModel = AutoModelForQuestionAnswering.from_pretrained(
            args.model.pretrained,
            config=self.lm_config,
        )

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "lang_model": self.lang_model.state_dict(),
            "args_prog": self.args.prog,
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        self.lang_model.load_state_dict(ckpt_state["lang_model"])
        self.args.prog = ckpt_state["args_prog"]
        self.eval()

    def load_checkpoint_file(self, checkpoint_file):
        assert Path(checkpoint_file).exists(), f"Model file not found: {checkpoint_file}"
        self.fabric.print(f"Loading model from {checkpoint_file}")
        self.from_checkpoint(self.fabric.load(checkpoint_file))

    def load_last_checkpoint_file(self, checkpoints_glob):
        checkpoint_files = files(checkpoints_glob)
        assert checkpoint_files, f"No model file found: {checkpoints_glob}"
        self.load_checkpoint_file(checkpoint_files[-1])

    def configure_optimizers(self):
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        train_dataset = QADataset("train", args=self.args, data=self.data, tokenizer=self.lm_tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=self.args.hardware.cpu_workers,
                                      batch_size=self.args.hardware.train_batch,
                                      collate_fn=data_collator,
                                      drop_last=False)
        self.fabric.print(f"Created train_dataset providing {len(train_dataset)} examples")
        self.fabric.print(f"Created train_dataloader providing {len(train_dataloader)} batches")
        return train_dataloader

    def val_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        val_dataset = QADataset("valid", args=self.args, data=self.data, tokenizer=self.lm_tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.infer_batch,
                                    collate_fn=data_collator,
                                    drop_last=False)
        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(f"Created val_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def test_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        test_dataset = QADataset("test", args=self.args, data=self.data, tokenizer=self.lm_tokenizer)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                     num_workers=self.args.hardware.cpu_workers,
                                     batch_size=self.args.hardware.infer_batch,
                                     collate_fn=data_collator,
                                     drop_last=False)
        self.fabric.print(f"Created test_dataset providing {len(test_dataset)} examples")
        self.fabric.print(f"Created test_dataloader providing {len(test_dataloader)} batches")
        return test_dataloader

    def training_step(self, inputs, batch_idx):
        outputs: QuestionAnsweringModelOutput = self.lang_model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        outputs: QuestionAnsweringModelOutput = self.lang_model(**inputs)
        labels: List[int] = inputs["labels"].tolist()
        preds: List[int] = outputs.logits.argmax(dim=-1).tolist()
        return {
            "loss": outputs.loss,
            "preds": preds,
            "labels": labels
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        inputs = self.lm_tokenizer(
            tupled(text),
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        outputs: QuestionAnsweringModelOutput = self.lang_model(**inputs)
        prob = outputs.logits.softmax(dim=1)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        return {
            'sentence': text,
            'prediction': pred,
            'positive_data': f"긍정 {positive_prob * 100:.1f}%",
            'negative_data': f"부정 {negative_prob * 100:.1f}%",
            'positive_width': f"{positive_prob * 100:.2f}%",
            'negative_width': f"{negative_prob * 100:.2f}%",
        }

    def run_server(self, server: Flask, *args, **kwargs):
        KorQuADModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    class WebAPI(FlaskView):
        def __init__(self, model: "NSMCModel"):
            self.model = model

        @route('/')
        def index(self):
            return render_template(self.model.args.server.page)

        @route('/api', methods=['POST'])
        def api(self):
            response = self.model.infer_one(text=request.json)
            return jsonify(response)
