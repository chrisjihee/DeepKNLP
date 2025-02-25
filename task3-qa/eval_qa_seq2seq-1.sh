CUDA_VISIBLE_DEVICES=4 python task3-qa/train_qa_seq2seq.py \
  --model_name_or_path output/korquad/train_qa_seq2seq-by-ket5/checkpoint-10068 \
  --dataset_name squad \
  --predict_with_generate \
  --do_eval \
  --output_dir output/korquad/train_qa_seq2seq-by-ket5-2 \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir
