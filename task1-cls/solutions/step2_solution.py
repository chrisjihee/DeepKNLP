"""Step 2 in-place answer snippets for task1-cls/run_cls.py.

Paste the following blocks into the matching TODO Step 2 locations.

NSMCModel.training_step:

    outputs: SequenceClassifierOutput = self.lang_model(**inputs)
    labels: torch.Tensor = inputs["labels"]
    preds: torch.Tensor = outputs.logits.argmax(dim=-1)
    acc: torch.Tensor = accuracy(preds=preds, labels=labels)

NSMCModel.validation_step:

    outputs: SequenceClassifierOutput = self.lang_model(**inputs)
    labels: List[int] = inputs["labels"].tolist()
    preds: List[int] = outputs.logits.argmax(dim=-1).tolist()
"""
