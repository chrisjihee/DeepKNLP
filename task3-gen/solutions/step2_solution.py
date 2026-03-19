"""Step 2 answer blocks for task3-gen/run_gen.py."""


def complete_step2_train_loop(args, model, train_dataset, val_dataset):
    train_dataloader, val_dataloader = build_dataloaders(args, train_dataset, val_dataset)
    task = GenerationTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def complete_step2_generation_cases():
    return [
        (
            "greedy",
            {
                "do_sample": False,
                "min_length": 10,
                "max_length": 40,
            },
        ),
        (
            "beam_search",
            {
                "do_sample": False,
                "min_length": 10,
                "max_length": 40,
                "num_beams": 3,
            },
        ),
        (
            "sampling_top_k",
            {
                "do_sample": True,
                "min_length": 10,
                "max_length": 40,
                "top_k": 50,
                "temperature": 0.9,
            },
        ),
        (
            "sampling_top_p",
            {
                "do_sample": True,
                "min_length": 10,
                "max_length": 40,
                "top_p": 0.92,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            },
        ),
    ]
