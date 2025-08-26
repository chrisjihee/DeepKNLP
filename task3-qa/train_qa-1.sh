CUDA_VISIBLE_DEVICES=7 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation-half.jsonl \
  --output_dir output/korquad/train_qa-by-kpfbert \
  --model_name_or_path jinmang2/kpfbert \
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
#   total_flos               =  6262956GF
#   train_loss               =     0.6857
#   train_runtime            = 0:02:18.61
#   train_samples            =      34315
#   train_samples_per_second =    247.549
#   train_steps_per_second   =      4.956

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =    87.1333
#   eval_f1                 =    91.5366
#   eval_runtime            = 0:00:05.76
#   eval_samples            =       3466
#   eval_samples_per_second =    601.401
#   eval_steps_per_second   =     75.305
