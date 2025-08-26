CUDA_VISIBLE_DEVICES=1 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir output/korquad/train_qa_seq2seq-by-pkot5-base \
  --model_name_or_path paust/pko-t5-base \
  --predict_with_generate \
  --do_train \
  --do_eval \
  --fp16 \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --overwrite_output_dir

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       output/korquad/train_qa_seq2seq-by-pkot5-base/eval_predictions.json

# {"exact_match": 74.78351229650156, "f1": 81.67965558409172}
# 2025.08.27: {"exact_match": 70.15933494977486, "f1": 78.20142143881837}
