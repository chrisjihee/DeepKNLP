OUTPUT_DIR=output/korquad/train_qa_by-pkot5-at-dev1
CUDA_VISIBLE_DEVICES=1 python task4B-qa-gen/train_qa_seq2seq.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path paust/pko-t5-large \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --bf16 \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 5e-5 \
  --predict_with_generate

python task4B-qa-gen/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

# pko-t5-base, lr=5e-5: {"exact_match": 71.56217526844475, "f1": 79.15031011581894}
# pko-t5-large, lr=5e-5: {"exact_match": 74.07343262902667, "f1": 80.87612476775051}
