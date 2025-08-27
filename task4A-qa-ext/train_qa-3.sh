OUTPUT_DIR=output/korquad/train_qa-by-koelectra
CUDA_VISIBLE_DEVICES=5 python task4A-qa-ext/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path monologg/koelectra-base-v3-discriminator \
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
  --learning_rate 3e-5 \
  --doc_stride 128

python task4A-qa-ext/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7588675GF
#  train_loss               =     0.8114
#  train_runtime            = 0:04:47.43
#  train_samples            =      31184
#  train_samples_per_second =    108.491
#  train_steps_per_second   =      9.042

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =    84.0665
#  eval_f1                 =    89.5272
#  eval_runtime            = 0:00:17.89
#  eval_samples            =       6178
#  eval_samples_per_second =    345.274
#  eval_steps_per_second   =     43.201

#{"exact_match": 84.20505715275372, "f1": 92.90331398272818}
