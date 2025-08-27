OUTPUT_DIR=output/korquad/train_qa-by-koelectra
python task4A-qa-ext/evaluate-KorQuAD-v1.py \
       data/korquad/KorQuAD_v1.0_dev.json \
       $OUTPUT_DIR/eval_predictions.json
