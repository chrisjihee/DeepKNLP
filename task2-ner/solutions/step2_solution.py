"""Step 2 answer blocks for task2-ner/run_ner.py.

Paste these functions into `NERModel` after finishing Step 1.
"""


def complete_step2_training_batch(self, inputs):
    inputs.pop("example_ids")
    outputs = self.lang_model(**inputs)
    labels = inputs["labels"]
    preds = outputs.logits.argmax(dim=-1)
    acc = accuracy(preds=preds, labels=labels, ignore_index=0)
    return outputs, labels, preds, acc


def complete_step2_validation_batch(self, inputs):
    example_ids = inputs.pop("example_ids").tolist()
    outputs = self.lang_model(**inputs)
    preds = outputs.logits.argmax(dim=-1)

    dict_of_char_label_ids = {}
    dict_of_char_pred_ids = {}
    for token_pred_ids, example_id in zip(preds.tolist(), example_ids):
        token_pred_tags = [self.id_to_label(x) for x in token_pred_ids]
        encoded_example = self._infer_dataset[example_id]
        offset_to_label = encoded_example.raw.get_offset_label_dict()
        all_char_pair_tags = [(None, None)] * len(encoded_example.raw.character_list)
        for token_id in range(self.args.model.seq_len):
            token_span = encoded_example.encoded.token_to_chars(token_id)
            if token_span:
                char_pred_tags = NERModel.label_to_char_labels(
                    token_pred_tags[token_id],
                    token_span.end - token_span.start,
                )
                for offset, char_pred_tag in zip(range(token_span.start, token_span.end), char_pred_tags):
                    all_char_pair_tags[offset] = (offset_to_label[offset], char_pred_tag)
        valid_char_pair_tags = [(a, b) for a, b in all_char_pair_tags if a and b]
        dict_of_char_label_ids[example_id] = [self.label_to_id(a) for a, b in valid_char_pair_tags]
        dict_of_char_pred_ids[example_id] = [self.label_to_id(b) for a, b in valid_char_pair_tags]

    list_of_char_pred_ids = []
    list_of_char_label_ids = []
    for encoded_example in [self._infer_dataset[i] for i in example_ids]:
        char_label_ids = dict_of_char_label_ids[encoded_example.idx]
        char_pred_ids = dict_of_char_pred_ids[encoded_example.idx]
        list_of_char_pred_ids.extend(char_pred_ids)
        list_of_char_label_ids.extend(char_label_ids)

    return outputs, list_of_char_pred_ids, list_of_char_label_ids
