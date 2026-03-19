"""Step 3 in-place answer snippets for task4A-qa-ext/serve_qa.py.

Paste the following blocks into the matching TODO Step 3 locations.

QAModel.__init__:

    logger.info(f"Loading model from {pretrained}")
    self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
    self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained)
    self.model.eval()

QAModel.infer_one:

    inputs = self.tokenizer.encode_plus(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        outputs = self.model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    predict_answer_tokens = inputs["input_ids"][0, start_index : end_index + 1]
    answer = self.tokenizer.decode(predict_answer_tokens)

    if self.normalized:
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        score = (torch.max(start_probs) * torch.max(end_probs)).item()
    else:
        score = float(torch.max(start_logits) + torch.max(end_logits))
"""
