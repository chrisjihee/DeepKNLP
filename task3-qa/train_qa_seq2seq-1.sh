CUDA_VISIBLE_DEVICES=0 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir output/korquad/train_qa_seq2seq-by-ket5-large \
  --model_name_or_path KETI-AIR/ke-t5-large \
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
       output/korquad/train_qa_seq2seq-by-ket5-large/eval_predictions.json

# {"exact_match": 78.195358503637, "f1": 88.64622370393245}
