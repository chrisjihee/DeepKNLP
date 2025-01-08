MODEL_NAME_OR_PATH=etri-lirs/egpt-1.3b-preview
OUTPUT_DIR=output/GNER/EGPT-1B

python task2-nerG-trainer1.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR
