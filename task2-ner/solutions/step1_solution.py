"""Step 1 answer blocks for task2-ner/run_ner.py.

Paste these functions into `NERModel`.
"""


def complete_step1_model_setup(args):
    data = NERCorpus(args)
    labels = data.labels
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for index, label in enumerate(labels)}
    lm_config = AutoConfig.from_pretrained(
        args.model.pretrained,
        num_labels=data.num_labels,
    )
    lm_tokenizer = AutoTokenizer.from_pretrained(
        args.model.pretrained,
        use_fast=True,
    )
    assert isinstance(lm_tokenizer, PreTrainedTokenizerFast)
    lang_model = AutoModelForTokenClassification.from_pretrained(
        args.model.pretrained,
        config=lm_config,
    )
    return data, labels, label_to_id, id_to_label, lm_config, lm_tokenizer, lang_model


def complete_step1_train_dataloader(self):
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
