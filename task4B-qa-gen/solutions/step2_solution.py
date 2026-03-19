"""Step 2 in-place answer snippets for task4B-qa-gen/train_qa_seq2seq.py.

Paste the following blocks into the matching TODO Step 2 locations.

Trainer initialization block:

    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        post_process_function=post_processing_function,
    )

Training block:

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

Evaluation block:

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

Prediction block:

    logger.info("*** Predict ***")
    results = trainer.predict(predict_dataset, predict_examples)
    metrics = results.metrics
    max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
"""
