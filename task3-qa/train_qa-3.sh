CUDA_VISIBLE_DEVICES=5 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation-half.jsonl \
  --output_dir output/korquad/train_qa-by-koelectra \
  --model_name_or_path monologg/koelectra-base-v3-discriminator \
  --do_train \
  --do_eval \
  --bf16 \
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

# ***** train metrics *****
#   epoch                    =        1.0
#   total_flos               =  6308037GF
#   train_loss               =     1.1661
#   train_runtime            = 0:03:00.62
#   train_samples            =      34562
#   train_samples_per_second =     191.35
#   train_steps_per_second   =      3.831

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =       82.3
#   eval_f1                 =    87.9499
#   eval_runtime            = 0:00:07.40
#   eval_samples            =       3485
#   eval_samples_per_second =    470.772
#   eval_steps_per_second   =     58.897
