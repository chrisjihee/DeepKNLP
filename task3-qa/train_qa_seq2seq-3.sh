OUTPUT_DIR=output/korquad/train_qa_by-pkot5-at-dev2
CUDA_VISIBLE_DEVICES=2 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path paust/pko-t5-base \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --bf16 --tf32 \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 5e-5 \
  --predict_with_generate

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

# 2025.02 (large): {"exact_match": 74.78351229650156, "f1": 81.67965558409172}

# 2025.08.27 (bf16, lr=3e-5, seq=384): {"exact_match": 48.26809837201247, "f1": 55.787263003319154}
# 2025.08.27 (bf16, lr=3e-5): {"exact_match": 70.31520609629374, "f1": 78.15017267878011}
# 2025.08.27 (fp16, lr=3e-5): {"exact_match": 70.15933494977486, "f1": 78.20142143881837}
# 2025.08.27 (bf16, lr=5e-5): {"exact_match": 71.56217526844475, "f1": 79.15031011581894}
