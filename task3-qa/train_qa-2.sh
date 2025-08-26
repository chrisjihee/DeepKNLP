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

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7580401GF
#  train_loss               =     0.7348
#  train_runtime            = 0:03:04.52
#  train_samples            =      31150
#  train_samples_per_second =    168.814
#  train_steps_per_second   =      5.278

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =     83.876
#  eval_f1                 =    89.5267
#  eval_runtime            = 0:00:12.51
#  eval_samples            =       6155
#  eval_samples_per_second =    491.995
#  eval_steps_per_second   =     61.549
