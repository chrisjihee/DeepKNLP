"""Step 1 in-place answer snippets for task2-ner/run_ner.py.

Paste the following blocks into the matching TODO Step 1 locations.

NERModel.__init__:

    self.data: NERCorpus = NERCorpus(args)
    self.labels: List[str] = self.data.labels
    self._label_to_id: Dict[str, int] = {label: index for index, label in enumerate(self.labels)}
    self._id_to_label: Dict[int, str] = {index: label for index, label in enumerate(self.labels)}
    self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
        args.model.pretrained,
        num_labels=self.data.num_labels,
    )
    self.lm_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        args.model.pretrained,
        use_fast=True,
    )
    assert isinstance(self.lm_tokenizer, PreTrainedTokenizerFast)
    self.lang_model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
        args.model.pretrained,
        config=self.lm_config,
    )

NERModel.train_dataloader:

    train_dataset = NERDataset("train", data=self.data, tokenizer=self.lm_tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, replacement=False),
        num_workers=self.args.hardware.cpu_workers,
        batch_size=self.args.hardware.train_batch,
        collate_fn=self.data.encoded_examples_to_batch,
        drop_last=False,
    )
    self.fabric.print(f"Created train_dataset providing {len(train_dataset)} examples")
    self.fabric.print(f"Created train_dataloader providing {len(train_dataloader)} batches")
    return train_dataloader
"""
