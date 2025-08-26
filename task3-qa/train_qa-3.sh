CUDA_VISIBLE_DEVICES=5 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path monologg/koelectra-base-v3-discriminator \
  --output_dir output/korquad/train_qa-by-koelectra \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --fp16 \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --doc_stride 128

# ***** train metrics *****
#   epoch                    =        1.0
#   total_flos               =  6308037GF
#   train_loss               =      1.035
#   train_runtime            = 0:03:04.18
#   train_samples            =      34562
#   train_samples_per_second =    187.644
#   train_steps_per_second   =      5.869

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =    83.4333
#   eval_f1                 =    88.8271
#   eval_runtime            = 0:00:07.38
#   eval_samples            =       3485
#   eval_samples_per_second =    471.764
#   eval_steps_per_second   =     59.021
