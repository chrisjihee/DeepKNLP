from pydantic import create_model
from typing import get_type_hints
import dataclasses
import math
import threading

import pydantic
import transformers.utils.logging
from progiter import ProgIter
import tqdm
import datasets
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyRow
from pydantic.v1.dataclasses import create_pydantic_model_from_dataclass
from pydantic_core import ArgsKwargs
from tqdm.asyncio import tqdm_asyncio, trange
from transformers.training_args import OptimizerNames
from typing_extensions import Self

from pydantic import model_validator
from pydantic.dataclasses import dataclass
import sys
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Tuple, Dict, Mapping, Any, ClassVar, Union, Callable, Type
import pandas as pd

import torch
import typer
from pydantic import BaseModel, Field

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, TypedData, TimeChecker, Counter
from chrisbase.io import LoggingFormat, make_dir, files, hr, to_table_lines, load_json
from chrisbase.io import get_hostname, get_hostaddr, current_file, first_or, cwd, hr, flush_or, make_parent_dir, setup_unit_logger, setup_dual_logger, open_file, file_lines, to_table_lines, new_path, get_http_clients
from chrisbase.time import now
from chrisbase.util import mute_tqdm_cls, tupled, to_dataframe
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, CharSpan, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BatchEncoding, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.data.data_collator import *

from DeepKNLP.arguments import DataFiles, DataOption, ModelOption, ServerOption, HardwareOption, PrintingOption, LearningOption, NewTrainerArguments, NewProjectEnv, NewDataOption, NewModelOption, NewLearningOption, NewHardwareOption
from DeepKNLP.arguments import TrainerArguments, TesterArguments, ServerArguments
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier
from DeepKNLP.metrics import accuracy, NER_Char_MacroF1, NER_Entity_MacroF1
from DeepKNLP.ner import NERCorpus, NERDataset, NEREncodedExample
from chrisdata.ner import GenNERSampleWrapper

import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import get_full_repo_name
from packaging import version

from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)
from transformers.utils import (
    ACCELERATE_MIN_VERSION,
    ExplicitEnum,
    cached_property,
    is_accelerate_available,
    is_ipex_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tf32_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    requires_backends,
)
from transformers.utils.generic import strtobool
from transformers.utils.import_utils import is_optimum_neuron_available
import torch
import torch.distributed as dist

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_0
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import DistributedType

from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.utils import logging as hf_logging

log_levels = hf_logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


class TrainingArguments(BaseModel):
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " eval_strategy."
            )
        },
    )

    torch_empty_cache_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`."
                    "This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372)."
                    "If left unset or set to None, cache will not be emptied."
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_safetensors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    restore_callback_states_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to the `Trainer` if they exist in the checkpoint."
        },
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers."},
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )
    use_mps_device: bool = field(
        default=False,
        metadata={
            "help": "This argument is deprecated. `mps` device will be used if available similar to `cuda` device."
                    " It will be removed in version 5.0 of 🤗 Transformers"
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )
    use_ipex: bool = field(
        default=False,
        metadata={
            "help": (
                "Use Intel extension for PyTorch when it is available, installation:"
                " 'https://github.com/intel/intel-extension-for-pytorch'"
            )
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    half_precision_backend: str = field(
        default="auto",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["auto", "apex", "cpu_amp"],
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    ddp_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "The backend to be used for distributed training",
            "choices": ["nccl", "gloo", "mpi", "ccl", "hccl", "cncl", "mccl"],
        },
    )
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics"
            )
        },
    )
    debug: Union[str, List[DebugOption]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None if not is_torch_available() or is_torch_greater_or_equal_than_2_0 else 2,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
                "Default is 2 for PyTorch < 2.0.0 and otherwise None."
            )
        },
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."},
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    fsdp: Optional[Union[List[FSDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    fsdp_min_num_params: int = field(
        default=0,
        metadata={
            "help": (
                "This parameter is deprecated. FSDP's minimum number of parameters for Default Auto Wrapping. (useful"
                " only when `fsdp` field is passed)."
            )
        },
    )
    fsdp_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "This parameter is deprecated. Transformer layer class name (case-sensitive) to wrap, e.g,"
                " `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed)."
            )
        },
    )
    accelerator_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with the internal Accelerator object initializtion. The value is either a "
                "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    deepspeed: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )

    optim: Union[OptimizerNames, str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Union[None, str, List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_broadcast_buffers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether the model repository is private or not."})
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    eval_do_concat_batches: bool = field(
        default=True,
        metadata={
            "help": "Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead store them as lists, with each batch kept separate."
        },
    )
    # Deprecated arguments
    fp16_backend: str = field(
        default="auto",
        metadata={
            "help": "Deprecated. Use half_precision_backend instead",
            "choices": ["auto", "apex", "cpu_amp"],
        },
    )
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default=None,
        metadata={"help": "Deprecated. Use `eval_strategy` instead"},
    )
    push_to_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    push_to_hub_organization: Optional[str] = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    _n_gpu: int = field(init=False, repr=False, default=-1)
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )

    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    torchdynamo: Optional[str] = field(
        default=None,
        metadata={
            "help": "This argument is deprecated, use `--torch_compile_backend` instead.",
        },
    )
    ray_scope: Optional[str] = field(
        default="last",
        metadata={
            "help": (
                'The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray'
                " will then use the last checkpoint of all trials, compare those, and select the best one. However,"
                " other options are also available. See the Ray documentation"
                " (https://docs.ray.io/en/latest/tune/api_docs/analysis.html"
                "#ray.tune.ExperimentAnalysis.get_best_trial)"
                " for more options."
            )
        },
    )
    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )

    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Pass {'dispatch_batches':VALUE} to `accelerator_config`."},
    )

    split_batches: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Pass {'split_batches':True} to `accelerator_config`."},
    )

    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )

    include_num_input_tokens_seen: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to `True`, will track the number of input tokens seen throughout training. (May be slower in distributed training)"
        },
    )

    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve model performances for instrcution fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )

    optim_target_modules: Union[None, str, List[str]] = field(
        default=None,
        metadata={
            "help": "Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore optimizer at the moment."
        },
    )

    batch_eval_metrics: bool = field(
        default=False,
        metadata={"help": "Break eval metrics calculation into batches to save memory."},
    )

    eval_on_start: bool = field(
        default=False,
        metadata={
            "help": "Whether to run through the entire `evaluation` step at the very beginning of training as a sanity check."
        },
    )

    use_liger_kernel: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to enable the Liger Kernel for model training."},
    )

    eval_use_gather_object: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices."
        },
    )


main = AppTyper()
logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


@dataclass
class DataCollatorForGNER:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        used_features_name = ["input_ids", "attention_mask", "labels"]
        features = [dict([item for item in feature.items() if item[0] in used_features_name]) for feature in features]

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@main.command()
def train(
        # env
        logging_home: str = typer.Option(default="output/task2-nerG"),
        logging_file: str = typer.Option(default="train-messages.out"),
        argument_file: str = typer.Option(default="train-arguments.json"),
        max_workers: int = typer.Option(default=4),
        debugging: bool = typer.Option(default=False),
        # data
        train_data: str = typer.Option(default="data/gner/zero-shot-train.jsonl"),
        # train_data: str = typer.Option(default="data/gner/pile-ner.jsonl"),
        eval_data: str = typer.Option(default="data/gner/zero-shot-dev.jsonl"),
        # eval_data: str = typer.Option(default=None),
        test_data: str = typer.Option(default="data/gner/zero-shot-test.jsonl"),
        # test_data: str = typer.Option(default=None),
        # max_train_samples: int = typer.Option(default=20000),
        max_train_samples: int = typer.Option(default=-1),
        max_eval_samples: int = typer.Option(default=-1),
        max_test_samples: int = typer.Option(default=-1),
        num_prog_samples: int = typer.Option(default=5000),
        max_source_length: int = typer.Option(default=512),
        max_target_length: int = typer.Option(default=512),
        use_cache_data: bool = typer.Option(default=True),
        # model
        # pretrained: str = typer.Option(default="google/flan-t5-small"),
        # pretrained: str = typer.Option(default="meta-llama/Llama-2-7b-hf"),
        pretrained: str = typer.Option(default="meta-llama/Llama-3.2-1B"),
        # pretrained: str = typer.Option(default="etri-lirs/kebyt5-small-preview"),
        # pretrained: str = typer.Option(default="etri-lirs/egpt-1.3b-preview"),
        # learning
        random_seed: int = typer.Option(default=7),
        trainer_args: str = typer.Option(default="configs/args/train_llama3_1b_supervised-base.json"),
        # hardware
        grad_acc_steps: int = typer.Option(default=4),
        train_batch: int = typer.Option(default=8),
        infer_batch: int = typer.Option(default=32),
        accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
        precision: str = typer.Option(default="bf16-mixed"),  # TODO: -> 32-true, bf16-mixed, 16-mixed
        strategy: str = typer.Option(default="ddp"),  # TODO: -> deepspeed
        device: List[int] = typer.Option(default=[0]),  # TODO: -> [0], [0,1], [0,1,2,3]
):
    torch.set_float32_matmul_precision('high')
    datasets.utils.logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("lightning").setLevel(logging.INFO)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.INFO)
    logging.getLogger("datasets.info").setLevel(logging.WARNING)
    logging.getLogger("datasets.builder").setLevel(logging.WARNING)
    logging.getLogger("datasets.arrow_dataset").setLevel(logging.WARNING)
    logging.getLogger("datasets.download.download_manager").setLevel(logging.WARNING)
    pretrained = Path(pretrained)
    args = NewTrainerArguments(
        env=NewProjectEnv(
            logging_home=logging_home,
            logging_file=logging_file,
            argument_file=argument_file,
            max_workers=1 if debugging else max(max_workers, 1),
            debugging=debugging,
            message_level=logging.INFO,
            message_format=LoggingFormat.CHECK_32,
        ),
        data=NewDataOption(
            train_path=train_data,
            eval_path=eval_data,
            test_path=test_data,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_test_samples=max_test_samples,
            num_prog_samples=num_prog_samples,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            use_cache_data=use_cache_data,
        ),
        model=NewModelOption(
            pretrained=pretrained,
        ),
        learning=NewLearningOption(
            random_seed=random_seed,
            trainer_args=trainer_args,
        ),
        hardware=NewHardwareOption(
            grad_acc_steps=grad_acc_steps,
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
    )
    fabric = Fabric(
        accelerator=args.hardware.accelerator,
        precision=args.hardware.precision,
        strategy=args.hardware.strategy,
        devices=args.hardware.devices,
    )

    def info_or_debug(m, *a, **k):
        if fabric.is_global_zero:  # or debugging:
            logger.info(m, *a, **k)
        else:
            logger.debug(m, *a, **k)

    fabric.print = info_or_debug
    fabric.launch()
    args.env.global_rank = fabric.global_rank
    args.env.local_rank = fabric.local_rank
    args.env.node_rank = fabric.node_rank
    args.env.world_size = fabric.world_size
    args.env.time_stamp = fabric.broadcast(args.env.time_stamp, src=0)
    args.env.setup_logger()
    trainer_args: Seq2SeqTrainingArguments = HfArgumentParser(Seq2SeqTrainingArguments).parse_json_file(args.learning.trainer_args)[0]
    # trainer_args: Seq2SeqTrainingArguments = HfArgumentParser(Seq2SeqTrainingArguments).parse_dict(load_json(args.learning.trainer_args))[0]

    # @dataclass
    # class TrainerArguments:
    #     output_dir: str = Field(default="output/task2-nerG")
    # bbb = TrainerArguments()

    ccc = TrainingArguments(
        output_dir=trainer_args.output_dir,
    )
    fabric.print(f"training_arguments={ccc}")

    args.learning.trainer_args = trainer_args.to_dict()

    # for checking
    # logger.info(f"[local_rank={fabric.local_rank}] args.env={args.env}({type(args.env)})")

    with JobTimer(
            name=f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=',
            args=args if fabric.is_global_zero else None, verbose=fabric.is_global_zero,
    ):
        # Set random seed
        fabric.barrier()
        fabric.seed_everything(args.learning.random_seed)

        # # Load model
        # config: PretrainedConfig = AutoConfig.from_pretrained(pretrained)
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(pretrained)
        # using_decoder_only_model = not config.is_encoder_decoder
        # if using_decoder_only_model:
        #     model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained)
        # else:
        #     model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        # fabric.barrier()
        # fabric.print(f"type(model)={type(model)} - {isinstance(model, PreTrainedModel)}")
        # fabric.print(f"type(config)={type(config)} - {isinstance(config, PretrainedConfig)}")
        # fabric.print(f"type(tokenizer)={type(tokenizer)} - {isinstance(tokenizer, PreTrainedTokenizerFast)}")
        # fabric.print("-" * 100)
        #
        # # Load dataset
        # train_dataset: Dataset | None = None
        # eval_dataset: Dataset | None = None
        # test_dataset: Dataset | None = None
        # if args.data.train_path:
        #     with fabric.rank_zero_first():
        #         train_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                               data_files=str(args.data.train_path),
        #                                               cache_dir=str(args.data.cache_train_dir))
        #         if args.data.max_train_samples > 0:
        #             train_dataset = train_dataset.select(range(min(len(train_dataset), args.data.max_train_samples)))
        #         fabric.print(f"Use {args.data.train_path} as train dataset: {len(train_dataset):,} samples")
        # if args.data.eval_path:
        #     with fabric.rank_zero_first():
        #         eval_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                              data_files=str(args.data.eval_path),
        #                                              cache_dir=str(args.data.cache_eval_dir))
        #         if args.data.max_eval_samples > 0:
        #             eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.data.max_eval_samples)))
        #         fabric.print(f"Use {args.data.eval_path} as eval dataset: {len(eval_dataset):,} samples")
        # if args.data.test_path:
        #     with fabric.rank_zero_first():
        #         test_dataset: Dataset = load_dataset("json", split=datasets.Split.TRAIN,
        #                                              data_files=str(args.data.test_path),
        #                                              cache_dir=str(args.data.cache_test_dir))
        #         if args.data.max_test_samples > 0:
        #             test_dataset = test_dataset.select(range(min(len(test_dataset), args.data.max_test_samples)))
        #         fabric.print(f"Use {args.data.eval_path} as test dataset: {len(test_dataset):,} samples")
        # fabric.barrier()
        # fabric.print("-" * 100)
        #
        # # Define tokenizer function for decoder-only model
        # def preprocess_for_decoder_only_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], counter=Counter(step=args.env.max_workers),
        #                                       update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
        #     # Fetch input data
        #     sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
        #     data_opt: NewDataOption = NewDataOption.model_validate(data_opt)
        #     prompt_text = f"[INST] {sample.instance.instruction_inputs} [/INST]"
        #     full_instruction = f"{prompt_text} {sample.instance.prompt_labels}"
        #
        #     def tokenize_train_sample():
        #         # Tokenize the full instruction
        #         model_inputs = tokenizer(
        #             text=full_instruction,
        #             max_length=data_opt.max_source_length + data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Add eos token if it is not the last token
        #         if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
        #             model_inputs["input_ids"].append(tokenizer.eos_token_id)
        #             model_inputs["attention_mask"].append(1)
        #
        #         # Add labels
        #         model_inputs["labels"] = model_inputs["input_ids"].copy()
        #
        #         # Find the prompt length
        #         prompt_tokens = tokenizer(
        #             text=prompt_text,
        #             max_length=data_opt.max_source_length + data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )["input_ids"]
        #
        #         # Remove the last token if it is an eos token
        #         if prompt_tokens[-1] == tokenizer.eos_token_id:
        #             prompt_tokens = prompt_tokens[:-1]
        #
        #         # Check if the prompt is longer than the input
        #         if len(prompt_tokens) > len(model_inputs["labels"]):
        #             raise ValueError(
        #                 f"Prompt is longer than the input, something went wrong. Prompt: {prompt_tokens}, input:"
        #                 f" {model_inputs['input_ids']}"
        #             )
        #
        #         # Mask the prompt tokens
        #         for i in range(len(prompt_tokens)):
        #             model_inputs["labels"][i] = -100
        #
        #         return model_inputs
        #
        #     def tokenize_infer_sample():
        #         # Tokenize the prompt
        #         model_inputs = tokenizer(
        #             text=prompt_text,
        #             max_length=max_source_length + max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Remove the last token if it is an eos token
        #         if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
        #             model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
        #             model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]
        #
        #         return model_inputs
        #
        #     # Tokenize the sample
        #     tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
        #     if update:
        #         return update(tokenized_sample, process_rank, counter)
        #     else:
        #         return tokenized_sample
        #
        # # Define tokenizer function for encoder-decoder model
        # def preprocess_for_encoder_decoder_model(row: LazyRow, process_rank: int, data_opt: dict[str, Any], cnt=Counter(step=args.env.max_workers),
        #                                          update: Callable[[BatchEncoding, int, Counter, ProgIter], BatchEncoding] = None) -> BatchEncoding:
        #     # Fetch input data
        #     sample: GenNERSampleWrapper = GenNERSampleWrapper.model_validate(row)
        #     data_opt: NewDataOption = NewDataOption.model_validate(data_opt)
        #
        #     def tokenize_train_sample():
        #         # Tokenize the instruction inputs
        #         model_inputs = tokenizer(
        #             text=sample.instance.instruction_inputs,
        #             max_length=data_opt.max_source_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #
        #         # Tokenize the prompt labels
        #         model_inputs["labels"] = tokenizer(
        #             text=sample.instance.prompt_labels,
        #             max_length=data_opt.max_target_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )["input_ids"]
        #
        #         return model_inputs
        #
        #     def tokenize_infer_sample():
        #         # Tokenize the instruction inputs
        #         model_inputs = tokenizer(
        #             text=sample.instance.instruction_inputs,
        #             max_length=data_opt.max_source_length,
        #             truncation=True,
        #             padding=False,
        #             return_tensors=None,
        #             add_special_tokens=True,
        #         )
        #         return model_inputs
        #
        #     # Tokenize the sample
        #     tokenized_sample: BatchEncoding = tokenize_train_sample() if sample.split == "train" else tokenize_infer_sample()
        #     if update:
        #         return update(tokenized_sample, process_rank, cnt)
        #     else:
        #         return tokenized_sample
        #
        # # Define progress update function
        # def update_progress(res: BatchEncoding, rank: int, counter: Counter, pbar: ProgIter):
        #     pre, cnt = counter.val(), counter.inc()
        #     if (cnt >= pbar.total or any(i % num_prog_samples == 0 for i in range(pre + 1, cnt + 1))) and rank == 0:
        #         pbar.step(inc=min(cnt - pbar._iter_idx, pbar.total - pbar._iter_idx))
        #         # fabric.print(pbar.format_message().rstrip())
        #         logger.info(pbar.format_message().rstrip())
        #     return res
        #
        # # Preprocess dataset
        # if train_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(train_dataset), desc='Preprocess train samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             train_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_train_path(len(train_dataset))) if args.data.use_cache_data else None,
        #             )
        # if eval_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(eval_dataset), desc='Preprocess eval samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             eval_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_eval_path(len(eval_dataset))) if args.data.use_cache_data else None,
        #             )
        # if test_dataset:
        #     with fabric.rank_zero_first():
        #         with ProgIter(total=len(test_dataset), desc='Preprocess test samples', file=open(os.devnull, 'w'), verbose=2) as manual_pbar:
        #             test_dataset.map(
        #                 preprocess_for_decoder_only_model if using_decoder_only_model else preprocess_for_encoder_decoder_model,
        #                 fn_kwargs={"data_opt": args.data.model_dump(), "update": lambda *xs: update_progress(*xs, pbar=manual_pbar)},
        #                 with_rank=True, batched=False, num_proc=args.env.max_workers, load_from_cache_file=args.data.use_cache_data,
        #                 cache_file_name=str(args.data.cache_test_path(len(test_dataset))) if args.data.use_cache_data else None,
        #             )
        # fabric.barrier()
        # fabric.print("-" * 100)
        #
        # # Construct Data collator
        # label_pad_token_id = -100
        # data_collator: DataCollatorForGNER = DataCollatorForGNER(
        #     tokenizer,
        #     model=model,
        #     padding=True,
        #     label_pad_token_id=label_pad_token_id,
        #     return_tensors="pt",
        # )
        # fabric.barrier()
        # fabric.print("*" * 100)


if __name__ == "__main__":
    main()
