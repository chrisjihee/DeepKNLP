"""Step 2 in-place answer snippets for task3-gen/run_gen.py.

Paste the following blocks into the matching TODO Step 2 locations.

step2 training block:

    train_dataloader, val_dataloader = build_dataloaders(args, train_dataset, val_dataset)
    task = GenerationTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

step2 generation strategy block:

    generation_cases = [
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
"""
