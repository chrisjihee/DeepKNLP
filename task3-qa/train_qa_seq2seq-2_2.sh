CUDA_VISIBLE_DEVICES=3 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir output/korquad/train_qa_seq2seq-by-pkot5-flan \
  --model_name_or_path paust/pko-flan-t5-large \
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
       output/korquad/train_qa_seq2seq-by-pkot5-flan/eval_predictions.json

# {"exact_match": 72.7918254243159, "f1": 80.30808517835682}
