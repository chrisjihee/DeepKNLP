"""Step 3 in-place answer snippets for task2-ner/run_ner.py.

Paste the following blocks into the matching TODO Step 3 locations.

NERModel.infer_one tokenization block:

    inputs = self.lm_tokenizer(
        tupled(text),
        max_length=self.args.model.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

NERModel.infer_one formatting block:

    all_probs = outputs.logits[0].softmax(dim=1)
    top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)
    tokens = self.lm_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    top_labels = [self.id_to_label(pred[0].item()) for pred in top_preds]
    result = []
    for token, label, top_prob in zip(tokens, top_labels, top_probs):
        if token in self.lm_tokenizer.all_special_tokens:
            continue
        result.append(
            {
                "token": token,
                "label": label,
                "prob": f"{round(top_prob[0].item(), 4):.4f}",
            }
        )
    return {
        "sentence": text,
        "result": result,
    }
"""
