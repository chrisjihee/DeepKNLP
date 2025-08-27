OUTPUT_DIR=output/korquad/train_qa_by-pkot5-at-dev0-test
CHECKPOINT_DIR=output/korquad/train_qa_by-pkot5-at-dev0/checkpoint-2500
CUDA_VISIBLE_DEVICES=0 python task4B-qa-gen/train_qa_seq2seq.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $CHECKPOINT_DIR \
  --do_eval \
  --bf16 \
  --per_device_eval_batch_size 8 \
  --max_seq_length 512 \
  --predict_with_generate

OUTPUT_DIR=output/korquad/train_qa_by-pkot5-at-dev0-test
python task4B-qa-gen/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

# pko-t5-base, lr=5e-5: {"exact_match": 71.56217526844475, "f1": 79.15031011581894}
# pko-t5-large, lr=5e-5: {"exact_match": 74.07343262902667, "f1": 80.87612476775051}
