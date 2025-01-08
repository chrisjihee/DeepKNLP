TRAIN_JSON_DIR=data/zero-shot-train.jsonl
VALID_JSON_DIR=data/zero-shot-test.jsonl
MODEL_NAME_OR_PATH=etri-lirs/egpt-1.3b-preview
OUTPUT_DIR=output/train_eagle_1b_supervised-base

python task2-nerG-trainer1.py \
    --do_train --do_eval --predict_with_generate \
    --train_json_dir $TRAIN_JSON_DIR \
    --valid_json_dir $VALID_JSON_DIR \
    --no_load_gner_customized_datasets \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR
