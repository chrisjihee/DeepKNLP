CUDA_VISIBLE_DEVICES=7 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path jinmang2/kpfbert \
  --output_dir output/korquad/train_qa-by-kpfbert \
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
#   total_flos               =  6262956GF
#   train_loss               =     0.6208
#   train_runtime            = 0:02:21.53
#   train_samples            =      34315
#   train_samples_per_second =    242.444
#   train_steps_per_second   =      7.581

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =    86.5667
#   eval_f1                 =    91.3738
#   eval_runtime            = 0:00:05.74
#   eval_samples            =       3466
#   eval_samples_per_second =    602.954
#   eval_steps_per_second   =       75.5
