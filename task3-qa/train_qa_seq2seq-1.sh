CUDA_VISIBLE_DEVICES=0 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path KETI-AIR/ke-t5-large \
  --output_dir output/korquad/train_qa_seq2seq-by-ket5-large \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --fp16 \
  --num_train_epochs 3 \
  --save_total_limit 2 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --predict_with_generate

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       output/korquad/train_qa_seq2seq-by-ket5-large/eval_predictions.json

# {"exact_match": 78.195358503637, "f1": 88.64622370393245}
