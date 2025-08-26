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

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7575290GF
#  train_loss               =     0.6886
#  train_runtime            = 0:03:06.75
#  train_samples            =      31129
#  train_samples_per_second =     166.68
#  train_steps_per_second   =       5.21

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =    85.8504
#  eval_f1                 =     91.178
#  eval_runtime            = 0:00:12.46
#  eval_samples            =       6152
#  eval_samples_per_second =    493.472
#  eval_steps_per_second   =     61.684
