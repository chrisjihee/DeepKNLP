CUDA_VISIBLE_DEVICES=5 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation-half.jsonl \
  --output_dir output/korquad/train_qa-by-koelectra \
  --model_name_or_path monologg/koelectra-base-v3-discriminator \
  --do_train \
  --do_eval \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 50 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --learning_rate 3e-5 \
  --overwrite_output_dir

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =    82.4333
#   eval_f1                 =    87.9543
#   eval_runtime            = 0:00:20.95
#   eval_samples            =       3485
#   eval_samples_per_second =     166.29
#   eval_steps_per_second   =     20.804
