CUDA_VISIBLE_DEVICES=6 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path klue/bert-base \
  --output_dir output/korquad/train_qa-by-kluebert \
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
#   total_flos               =  6269527GF
#   train_loss               =     0.7004
#   train_runtime            = 0:02:23.49
#   train_samples            =      34351
#   train_samples_per_second =    239.393
#   train_steps_per_second   =      7.485

# ***** eval metrics *****
#   epoch                   =        1.0
#   eval_exact_match        =       84.4
#   eval_f1                 =    89.7209
#   eval_runtime            = 0:00:05.88
#   eval_samples            =       3464
#   eval_samples_per_second =    588.265
#   eval_steps_per_second   =     73.533
