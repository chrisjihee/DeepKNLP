"""Step 3 answer blocks for task2-ner/run_ner.py.

Paste these functions into `NERModel` after finishing Step 1 and Step 2.
"""


def complete_step3_tokenize_text(self, text):
    return self.lm_tokenizer(
        tupled(text),
        max_length=self.args.model.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def complete_step3_format_inference(self, text, inputs, outputs):
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
