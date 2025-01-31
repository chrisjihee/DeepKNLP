import pandas as pd
from pydantic import BaseModel

from chrisbase.io import LoggerWriter, hr
from progiter import ProgIter
from transformers import Trainer
from transformers.trainer import *
from transformers.trainer_seq2seq import Seq2SeqTrainer

logger = logging.get_logger(__name__)


class TrainingValues(BaseModel):
    num_train_epochs: float
    num_update_steps_per_epoch: int
    num_examples: int
    num_train_samples: float
    epoch_based: bool
    len_dataloader: int
    max_steps: int
    total_train_batch_size: int

    @classmethod
    def from_trainer(cls, trainer: Trainer):
        total_train_batch_size = trainer.args.world_size * trainer._train_batch_size * trainer.args.gradient_accumulation_steps
        values = trainer.set_initial_training_values(
            args=trainer.args,
            dataloader=trainer.get_train_dataloader(),
            total_train_batch_size=total_train_batch_size
        )
        return cls(**dict(zip(cls.__annotations__.keys(), list(values) + [total_train_batch_size])))


class CustomProgressCallback(TrainerCallback):

    def __init__(
            self,
            trainer: Trainer,
            output_path: str | Path,
            logging_epochs: float,
            eval_epochs: float,
            save_epochs: float,
            progress_seconds: float = 3.0,
            metric_formats: Mapping[str, str] | None = None,
    ):
        super().__init__()
        self.trainer: Trainer = trainer
        self.output_path: str | Path = output_path
        self.training_values: TrainingValues = TrainingValues.from_trainer(trainer)
        self.progress_seconds: float = progress_seconds

        self.training_pbar: Optional[ProgIter] = None
        self.prediction_pbar: Optional[ProgIter] = None
        self.current_step: int = 0
        self.metrics_table: pd.DataFrame = pd.DataFrame()
        self.metric_formats: Mapping[str, str] | None = metric_formats

        self.logging_step_set = set()
        self.eval_step_set = set()
        self.save_step_set = set()
        if 0 < logging_epochs:
            for i in range(int(math.ceil(self.training_values.num_train_epochs) / logging_epochs)):
                self.logging_step_set.add(round(self.training_values.num_update_steps_per_epoch * logging_epochs * (i + 1)))
        if 0 < eval_epochs:
            for i in range(int(math.ceil(self.training_values.num_train_epochs) / eval_epochs)):
                self.eval_step_set.add(round(self.training_values.num_update_steps_per_epoch * eval_epochs * (i + 1)))
        if 0 < save_epochs:
            for i in range(int(math.ceil(self.training_values.num_train_epochs) / save_epochs)):
                self.save_step_set.add(round(self.training_values.num_update_steps_per_epoch * save_epochs * (i + 1)))

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            logger.info(hr(c='-'))
            logger.info(f"***** Beginning Training *****")
            logger.info(f"  - Train Epochs       = {self.trainer.args.num_train_epochs}")
            logger.info(f"  - Train Examples     = {self.training_values.num_examples:,}")
            logger.info(f"  - Train Batch Size   = {self.training_values.total_train_batch_size:,}"
                        f" = {self.trainer._train_batch_size} * {self.trainer.args.gradient_accumulation_steps} * {self.trainer.args.world_size}")
            logger.info(f"  - Train Optim Steps  = {self.training_values.max_steps:,}"
                        f" = {self.training_values.num_update_steps_per_epoch:,} * {self.trainer.args.num_train_epochs}")
            logger.info(f"  - Train Model Params = {get_model_param_count(self.trainer.model, trainable_only=True):,}")
            logger.info(f"  - Eval Examples      = {len(self.trainer.eval_dataset):,}")
            logger.info(f"  - Eval Batch Size    = {self.trainer.args.per_device_eval_batch_size * self.trainer.args.world_size:,}"
                        f" = {self.trainer.args.per_device_eval_batch_size} * {self.trainer.args.world_size}")
            self.training_pbar = ProgIter(
                time_thresh=self.progress_seconds,
                verbose=3,
                stream=LoggerWriter(logger),
                total=state.max_steps,
                desc="[training]",
            )
            self.training_pbar.begin()
        self.current_step = 0

    def epoch_by_step(self, state: TrainerState):
        state.epoch = state.global_step / self.training_values.num_update_steps_per_epoch
        return state.epoch

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.training_pbar is not None:
            self.training_pbar.step(state.global_step - self.current_step, display=False)
        self.current_step = state.global_step
        control.should_log = control.should_log or self.current_step in self.logging_step_set
        control.should_save = control.should_save or self.current_step in self.save_step_set
        control.should_evaluate = control.should_evaluate or self.current_step in self.eval_step_set

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = True
        if self.training_pbar is not None:
            self.training_pbar.end()
            self.training_pbar = None
            logger.info(hr(c='-'))

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero:
            if eval_dataloader and has_length(eval_dataloader):
                if self.prediction_pbar is None:
                    self.prediction_pbar = ProgIter(
                        time_thresh=self.progress_seconds,
                        verbose=3,
                        stream=LoggerWriter(logger),
                        total=len(eval_dataloader),
                        desc="[checking]",
                    )
                    self.prediction_pbar.begin()
                if self.prediction_pbar is not None:
                    self.prediction_pbar.step()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.prediction_pbar is not None:
            self.prediction_pbar.end()
            self.prediction_pbar = None
            if self.training_pbar is not None:
                self.training_pbar.display_message()

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.prediction_pbar is not None:
            self.prediction_pbar.end()
            self.prediction_pbar = None
            if self.training_pbar is not None:
                self.training_pbar.display_message()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: Optional[Mapping[str, Any]] = None, exclude_keys=("epoch", "step"), **kwargs):
        if state.is_world_process_zero:
            metrics = {
                "step": state.global_step,
                "epoch": round(self.epoch_by_step(state), 3),
            }
            for k, v in logs.items():
                if k not in exclude_keys:
                    metrics[k] = v
                    if k.endswith("_flos"):
                        metrics[k.replace("_flos", "_pflos")] = v / 1e15
            new_metrics_row = pd.DataFrame([metrics])
            self.metrics_table = pd.concat([self.metrics_table, new_metrics_row], ignore_index=True)
            self.metrics_table.to_csv(self.output_path, index=False)
            if self.training_pbar is not None and self.metric_formats:
                formatted_metrics = ', '.join([f'{k}={metrics[k]:{self.metric_formats[k]}}' for k in self.metric_formats if k in metrics])
                if formatted_metrics:
                    self.training_pbar.set_extra(f"| {formatted_metrics}")
                    if self.prediction_pbar is None:
                        self.training_pbar.display_message()


class GNERTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        super().__init__(*args, **kwargs)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "E",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():  # FutureWarning: `is_torch_tpu_available` is deprecated and will be removed in 4.41.0. Please use the `is_torch_xla_available` instead.
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                    args.eval_accumulation_steps is not None
                    and (step + 1) % args.eval_accumulation_steps == 0
                    and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds,
                                           tokenizer=self.processing_class, is_encoder_decoder=self.is_encoder_decoder,
                                           output_dir=args.output_dir, save_prefix=metric_key_prefix, save_suffix=f"{self.state.global_step}")
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
