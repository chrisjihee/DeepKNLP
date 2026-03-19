"""Step 2 answer blocks for task4A-qa-ext/train_qa.py."""

import logging

from trainer_qa import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)


def complete_step2_build_trainer(model, training_args, train_dataset, eval_dataset, eval_examples, tokenizer, data_collator, post_processing_function, compute_metrics):
    return QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )


def complete_step2_run_train(trainer, training_args, last_checkpoint, train_dataset, data_args):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def complete_step2_run_eval(trainer, training_args, eval_dataset, data_args):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def complete_step2_run_predict(trainer, training_args, predict_dataset, predict_examples, data_args):
    logger.info("*** Predict ***")
    results = trainer.predict(predict_dataset, predict_examples)
    metrics = results.metrics
    max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
