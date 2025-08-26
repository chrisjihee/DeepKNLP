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

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7588675GF
#  train_loss               =     1.0164
#  train_runtime            = 0:04:15.29
#  train_samples            =      31184
#  train_samples_per_second =     122.15
#  train_steps_per_second   =      3.819

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =    82.4558
#  eval_f1                 =    88.5156
#  eval_runtime            = 0:00:17.78
#  eval_samples            =       6178
#  eval_samples_per_second =    347.297
#  eval_steps_per_second   =     43.454
