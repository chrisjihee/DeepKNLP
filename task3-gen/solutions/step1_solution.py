"""Step 1 answer blocks for task3-gen/run_gen.py."""


def complete_step1_build_train_args(model_preset, max_seq_length, batch_size, learning_rate, epochs, seed):
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


def complete_step1_load_pretrained_components(model_name):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, eos_token="</s>")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def complete_step1_prepare_generation_datasets(args, tokenizer):
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
