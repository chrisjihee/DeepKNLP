"""Step 3 in-place answer snippets for task4B-qa-gen/serve_qa_seq2seq.py.

Paste the following blocks into the matching TODO Step 3 locations.

QAModel.__init__:

    logger.info(f"Loading model from {pretrained}")
    self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
    self.model.eval()

QAModel.infer_one:

    input_text = f"question: {question} context: {context}"
    inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
            num_beams=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

    answer = self.tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)

    token_probs = []
    for i, token_id in enumerate(output_ids.sequences[0]):
        if i == 0:
            continue
        token_prob = F.softmax(output_ids.scores[i - 1], dim=-1)[0, token_id].item()
        token_probs.append(token_prob)

    score = torch.prod(torch.tensor(token_probs)).item()
"""
