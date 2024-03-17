import logging
import os
from pathlib import Path
from time import sleep
from typing import List, Dict

import torch
import typer
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from DeepKNLP.arguments import TrainerArguments, TesterArguments, DataOption, DataFiles, ModelOption, HardwareOption, LearningOption
from DeepKNLP.cls import NsmcCorpus, ClassificationDataset
from DeepKNLP.helper import CheckpointSaver, fabric_barrier, epsilon, data_collator
from DeepKNLP.metrics import accuracy
from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat
from chrisbase.util import mute_tqdm_cls

logger = logging.getLogger(__name__)
main = AppTyper()


class NsmcModel(LightningModule):
    def __init__(self, args: TrainerArguments | TesterArguments):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.data: NsmcCorpus = NsmcCorpus(args)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            args.model.pretrained,
            config=AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=self.data.num_labels
            ),
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,
        )

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        train_dataset = ClassificationDataset("train", data=self.data, tokenizer=self.tokenizer)
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
        val_dataset = ClassificationDataset("valid", data=self.data, tokenizer=self.tokenizer)
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
        val_dataset = ClassificationDataset("test", data=self.data, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.infer_batch,
                                    collate_fn=data_collator,
                                    drop_last=False)
        self.fabric.print(f"Created test_dataset providing {len(val_dataset)} examples")
        self.fabric.print(f"Created test_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def training_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
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


def train_loop(
        fabric: Fabric,
        model: NsmcModel,
        optimizer: OptimizerLRScheduler,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
):
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.learning.print_rate_on_training * num_batch - epsilon if model.args.learning.print_step_on_training < 1 else model.args.learning.print_step_on_training
    check_interval = model.args.learning.check_rate_on_training * num_batch - epsilon
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0
    ckpt_saver = CheckpointSaver(
        fabric=fabric,
        output_home=model.args.env.output_home,
        name_format=model.args.learning.name_format_on_saving,
        saving_mode=model.args.learning.saving_mode,
        num_saving=model.args.learning.num_saving,
    )
    for epoch in range(model.args.learning.num_epochs):
        metrics: Dict[str, float | int] = {}
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="training")
        for i, batch in enumerate(dataloader, start=1):
            model.train()
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch
            optimizer.zero_grad()
            outputs = model.training_step(batch, i)
            fabric.backward(outputs["loss"])
            optimizer.step()
            progress.update()
            model.eval()
            fabric.barrier()
            metrics["step"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item())
            metrics["epoch"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4)
            metrics["loss"] = fabric.all_gather(outputs["loss"]).mean().item()
            metrics["acc"] = fabric.all_gather(outputs["acc"]).mean().item()
            fabric.log_dict(metrics=metrics, step=metrics["step"])
            if i % print_interval < 1:
                fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                             f" | {model.args.learning.tag_format_on_training.format(**metrics)}")
            if model.args.prog.global_step % check_interval < 1:
                val_loop(fabric, model, val_dataloader, ckpt_saver)
        fabric_barrier(fabric, "[after-epoch]", c='=')
    fabric_barrier(fabric, "[after-train]")
    if test_dataloader:
        test_loop(fabric, model, test_dataloader, ckpt_saver)


@torch.no_grad()
def val_loop(
        fabric: Fabric,
        model: NsmcModel,
        dataloader: DataLoader,
        ckpt_saver: CheckpointSaver,
):
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.learning.print_rate_on_validate * num_batch - epsilon if model.args.learning.print_step_on_validate < 1 else model.args.learning.print_step_on_validate
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float | int] = {}
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="checking")
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.validation_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i >= num_batch:
            fabric.barrier()
            metrics["step"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item())
            metrics["epoch"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4)
            metrics["val_loss"] = fabric.all_gather(torch.stack(losses)).mean().item()
            metrics["val_acc"] = accuracy(fabric.all_gather(torch.tensor(preds)).flatten(),
                                          fabric.all_gather(torch.tensor(labels)).flatten()).item()
            fabric.log_dict(metrics=metrics, step=metrics["step"])
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                         f" | {model.args.learning.tag_format_on_validate.format(**metrics)}")
        elif i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric_barrier(fabric, "[after-check]")
    ckpt_saver.save_checkpoint(metrics=metrics, state={'model': model,
                                                       'args': model.args})


@torch.no_grad()
def test_loop(
        fabric: Fabric,
        model: NsmcModel,
        dataloader: DataLoader,
        ckpt_saver: CheckpointSaver | None = None,
):
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.learning.print_rate_on_evaluate * num_batch - epsilon if model.args.learning.print_step_on_evaluate < 1 else model.args.learning.print_step_on_evaluate
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float | int] = {}
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(range(num_batch), unit=f"x{dataloader.batch_size}b", desc="testing")
    for i, batch in enumerate(dataloader, start=1):
        if i == 1 and ckpt_saver is not None:
            ckpt_state = ckpt_saver.load_checkpoint()
            if ckpt_state is not None:
                model.load_state_dict(ckpt_state['model'])
                model.args = ckpt_state['args']
        outputs = model.validation_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i >= num_batch:
            fabric.barrier()
            metrics["step"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0)).mean().item())
            metrics["epoch"] = round(fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(), 4)
            metrics["test_loss"] = fabric.all_gather(torch.stack(losses)).mean().item()
            metrics["test_acc"] = accuracy(fabric.all_gather(torch.tensor(preds)).flatten(),
                                           fabric.all_gather(torch.tensor(labels)).flatten()).item()
            fabric.log_dict(metrics=metrics, step=metrics["step"])
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                         f" | {model.args.learning.tag_format_on_evaluate.format(**metrics)}")
        elif i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric_barrier(fabric, "[after-test]")


@main.command()
def train(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        job_version: int = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        logging_file: str = typer.Option(default="logging.out"),
        argument_file: str = typer.Option(default="arguments.json"),
        # data
        data_home: str = typer.Option(default="data"),
        data_name: str = typer.Option(default="nsmc-mini"),  # TODO: -> nsmc
        train_file: str = typer.Option(default="ratings_train.txt"),
        valid_file: str = typer.Option(default="ratings_test.txt"),
        test_file: str = typer.Option(default=None),
        num_check: int = typer.Option(default=0),  # TODO: -> 2
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        finetuning: str = typer.Option(default="finetuning"),
        seq_len: int = typer.Option(default=64),  # TODO: -> 512
        # hardware
        train_batch: int = typer.Option(default=10),  # TODO: -> 64
        infer_batch: int = typer.Option(default=10),  # TODO: -> 64
        accelerator: str = typer.Option(default="cuda"),
        precision: str = typer.Option(default="32-true"),
        strategy: str = typer.Option(default="ddp"),
        device: List[int] = typer.Option(default=[0, 1]),
        # learning
        learning_rate: float = typer.Option(default=5e-5),
        random_seed: int = typer.Option(default=7),
        saving_mode: str = typer.Option(default="max val_acc"),
        num_saving: int = typer.Option(default=2),
        num_epochs: int = typer.Option(default=1),  # TODO: -> 3
        check_rate_on_training: float = typer.Option(default=1 / 5),
        print_rate_on_training: float = typer.Option(default=1 / 20),
        print_rate_on_validate: float = typer.Option(default=1 / 3),
        print_rate_on_evaluate: float = typer.Option(default=1 / 3),
        print_step_on_training: int = typer.Option(default=-1),
        print_step_on_validate: int = typer.Option(default=-1),
        print_step_on_evaluate: int = typer.Option(default=-1),
        tag_format_on_training: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, loss={loss:06.4f}, acc={acc:06.4f}"),
        tag_format_on_validate: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}"),
        tag_format_on_evaluate: str = typer.Option(default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"),
        name_format_on_saving: str = typer.Option(default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}"),
):
    torch.set_float32_matmul_precision('medium')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # logging.getLogger("c10d-NullHandler").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TrainerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_40 if debugging else LoggingFormat.CHECK_40,
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                train=train_file,
                valid=valid_file,
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        learning=LearningOption(
            learning_rate=learning_rate,
            random_seed=random_seed,
            saving_mode=saving_mode,
            num_saving=num_saving,
            num_epochs=num_epochs,
            check_rate_on_training=check_rate_on_training,
            print_rate_on_training=print_rate_on_training,
            print_rate_on_validate=print_rate_on_validate,
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_training=print_step_on_training,
            print_step_on_validate=print_step_on_validate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_training=tag_format_on_training,
            tag_format_on_validate=tag_format_on_validate,
            tag_format_on_evaluate=tag_format_on_evaluate,
            name_format_on_saving=name_format_on_saving,
        ),
    )
    output_home = f"{finetuning}/{data_name}"
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    output_version = args.env.job_version if args.env.job_version else CSVLogger(output_home, output_name).version
    args.prog.tb_logger = TensorBoardLogger(output_home, output_name, output_version)  # tensorboard --logdir finetuning --bind_all
    args.prog.csv_logger = CSVLogger(output_home, output_name, output_version, flush_logs_every_n_steps=1)
    args.env.job_version = output_version
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    fabric = Fabric(
        loggers=[args.prog.tb_logger, args.prog.csv_logger],
        devices=args.hardware.devices,
        strategy=args.hardware.strategy,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    sleep(fabric.global_rank * 0.1)
    fabric.seed_everything(args.learning.random_seed)
    args.prog.world_size = fabric.world_size
    args.prog.node_rank = fabric.node_rank
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    args.env.set_output_home(args.prog.csv_logger.log_dir)
    fabric.barrier()

    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                  args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
                  verbose=verbose > 0 and fabric.local_rank == 0,
                  mute_warning="lightning.fabric.loggers.csv_logs",
                  rt=1, rb=1, rc='='):
        model = NsmcModel(args=args)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, optimizer)
        fabric_barrier(fabric, "[after-model]", c='=')

        train_dataloader = model.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        fabric_barrier(fabric, "[after-train_dataloader]", c='=')

        val_dataloader = model.val_dataloader()
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
        fabric_barrier(fabric, "[after-val_dataloader]", c='=')

        train_loop(fabric, model, optimizer,
                   dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   test_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
