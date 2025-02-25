CUDA_VISIBLE_DEVICES=2 python task3-qa/train_qa_seq2seq.py \
  --train_file data/korquad/train.jsonl \
  --validation_file data/korquad/validation.jsonl \
  --output_dir output/korquad/train_qa_seq2seq-by-kot5 \
  --model_name_or_path output/korquad/train_qa_seq2seq-by-kot5/checkpoint-10068 \
  --predict_with_generate \
  --do_eval \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir

python task3-qa/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       output/korquad/train_qa_seq2seq-by-kot5/eval_predictions.json
