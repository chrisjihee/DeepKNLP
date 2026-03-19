"""Step 1 in-place answer snippets for task1-cls/run_cls.py.

Paste the following blocks into the matching TODO Step 1 locations.

NSMCModel.__init__:

    self.data: NsmcCorpus = NsmcCorpus(args)
    self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
        args.model.pretrained,
        num_labels=self.data.num_labels,
    )
    self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model.pretrained,
        use_fast=True,
    )
    self.lang_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.model.pretrained,
        config=self.lm_config,
    )

NSMCModel.train_dataloader:

    train_dataset = ClassificationDataset(
        "train",
        data=self.data,
        tokenizer=self.lm_tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, replacement=False),
        num_workers=self.args.hardware.cpu_workers,
        batch_size=self.args.hardware.train_batch,
        collate_fn=data_collator,
        drop_last=False,
    )
    self.fabric.print(f"Created train_dataset providing {len(train_dataset)} examples")
    self.fabric.print(f"Created train_dataloader providing {len(train_dataloader)} batches")
    return train_dataloader
"""
