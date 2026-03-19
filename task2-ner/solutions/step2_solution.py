"""Step 2 in-place answer snippets for task2-ner/run_ner.py.

Paste the following blocks into the matching TODO Step 2 locations.

NERModel.training_step:

    inputs.pop("example_ids")
    outputs: TokenClassifierOutput = self.lang_model(**inputs)
    labels: Tensor = inputs["labels"]
    preds: Tensor = outputs.logits.argmax(dim=-1)
    acc: Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

NERModel.validation_step:

    inputs.pop("example_ids")
    outputs: TokenClassifierOutput = self.lang_model(**inputs)
    labels: Tensor = inputs["labels"]
    preds: Tensor = outputs.logits.argmax(dim=-1)
    valid_mask = labels != 0
    list_of_token_pred_ids = preds[valid_mask].tolist()
    list_of_token_label_ids = labels[valid_mask].tolist()
"""
