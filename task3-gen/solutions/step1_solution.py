"""Step 1 in-place answer snippets for task3-gen/run_gen.py.

Paste the following blocks into the matching TODO Step 1 locations.

build_train_args:

    preset = get_preset(model_preset)
    return GenerationTrainArguments(
        pretrained_model_name=preset["model_name"],
        downstream_model_dir=preset["output_dir"],
        downstream_corpus_name="nsmc",
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=seed,
    )

load_pretrained_components:

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, eos_token="</s>")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

prepare_generation_datasets:

    nlpbook.set_seed(args)
    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=args.force_download,
    )
    corpus = NsmcCorpus()
    train_dataset = GenerationDataset(args=args, corpus=corpus, tokenizer=tokenizer, mode="train")
    val_dataset = GenerationDataset(args=args, corpus=corpus, tokenizer=tokenizer, mode="test")
    return corpus, train_dataset, val_dataset
"""
