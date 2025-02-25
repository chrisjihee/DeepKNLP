CUDA_VISIBLE_DEVICES=2 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir output/korquad/train_qa_seq2seq-by-kot5 \
  --model_name_or_path wisenut-nlp-team/KoT5-base \
  --predict_with_generate \
  --do_train \
  --do_eval \
  --num_train_epochs 3 \
  --save_total_limit 2 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --per_device_train_batch_size 12 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --learning_rate 3e-5 \
  --overwrite_output_dir

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       output/korquad/train_qa_seq2seq-by-kot5/eval_predictions.json
