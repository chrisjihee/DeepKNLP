OUTPUT_DIR=output/korquad/train_qa-by-kpfbert
CUDA_VISIBLE_DEVICES=7 python task4A-qa-ext/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path jinmang2/kpfbert \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --bf16 \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --doc_stride 128

python task4A-qa-ext/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7575290GF
#  train_loss               =     0.5984
#  train_runtime            = 0:03:31.41
#  train_samples            =      31129
#  train_samples_per_second =    147.238
#  train_steps_per_second   =     12.274

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =    86.5258
#  eval_f1                 =    91.4983
#  eval_runtime            = 0:00:12.53
#  eval_samples            =       6152
#  eval_samples_per_second =    490.797
#  eval_steps_per_second   =      61.35

#{"exact_match": 86.66435746449602, "f1": 94.49885248885852}
