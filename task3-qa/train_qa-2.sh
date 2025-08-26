CUDA_VISIBLE_DEVICES=6 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation-half.jsonl \
  --output_dir output/korquad/train_qa-by-kluebert \
  --model_name_or_path klue/bert-base \
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
#   eval_exact_match        =    83.9333
#   eval_f1                 =    89.2887
#   eval_runtime            = 0:00:18.47
#   eval_samples            =       3464
#   eval_samples_per_second =    187.537
#   eval_steps_per_second   =     23.442
