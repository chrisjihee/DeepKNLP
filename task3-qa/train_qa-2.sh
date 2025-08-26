OUTPUT_DIR=output/korquad/train_qa-by-kluebert
CUDA_VISIBLE_DEVICES=6 python task3-qa/train_qa.py \
  --train_file data/korquad/train-half.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --model_name_or_path klue/bert-base \
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

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json

#***** train metrics *****
#  epoch                    =        1.0
#  total_flos               =  7580401GF
#  train_loss               =     0.6435
#  train_runtime            = 0:03:31.49
#  train_samples            =      31150
#  train_samples_per_second =    147.286
#  train_steps_per_second   =     12.275

#***** eval metrics *****
#  epoch                   =        1.0
#  eval_exact_match        =    84.5168
#  eval_f1                 =    90.0583
#  eval_runtime            = 0:00:12.46
#  eval_samples            =       6155
#  eval_samples_per_second =    493.623
#  eval_steps_per_second   =     61.753

#{"exact_match": 84.65535157603048, "f1": 93.21929969013}
