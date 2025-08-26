CUDA_VISIBLE_DEVICES=7 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation-half.jsonl \
  --output_dir output/korquad/train_qa-by-kpfbert \
  --model_name_or_path jinmang2/kpfbert \
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
#   eval_exact_match        =       86.8
#   eval_f1                 =    91.4781
#   eval_runtime            = 0:00:18.46
#   eval_samples            =       3466
#   eval_samples_per_second =    187.685
#   eval_steps_per_second   =     23.501
