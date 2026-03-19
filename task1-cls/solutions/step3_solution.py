"""Step 3 in-place answer snippets for task1-cls/run_cls.py.

Paste the following blocks into the matching TODO Step 3 locations.

NSMCModel.infer_one tokenization block:

    inputs = self.lm_tokenizer(
        tupled(text),
        max_length=self.args.model.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

NSMCModel.infer_one formatting block:

    pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
    positive_prob = round(prob[0][1].item(), 4)
    negative_prob = round(prob[0][0].item(), 4)
    return {
        "sentence": text,
        "prediction": pred,
        "positive_data": f"긍정 {positive_prob * 100:.1f}%",
        "negative_data": f"부정 {negative_prob * 100:.1f}%",
        "positive_width": f"{positive_prob * 100:.2f}%",
        "negative_width": f"{negative_prob * 100:.2f}%",
    }
"""
