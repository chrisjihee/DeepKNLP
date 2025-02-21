set -x
port=$(shuf -i25000-30000 -n1)
TRAIN_JSON_DIR="data/gner/each/crossner_ai-train.jsonl"
VALID_JSON_DIR="data/gner/each-sampled/crossner_ai-dev=100.jsonl"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="output/GNER/Llama-3-1B"
DEEPSPEED_CONFIG="configs/deepspeed/ds2_llama.json"

python -m deepspeed.launcher.runner --include="localhost:0,1" --master_port $port task2-nerG-trainer1.py \
    --do_train --do_eval --predict_with_generate \
    --no_load_gner_customized_datasets \
    --train_json_dir $TRAIN_JSON_DIR \
    --valid_json_dir $VALID_JSON_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --bf16 True --tf32 True \
    --lr_scheduler_type cosine \
    --learning_rate 2e-05 \
    --warmup_ratio 0.04 --weight_decay 0. \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --num_train_epochs 1 \
    --logging_strategy steps --logging_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --overwrite_output_dir --overwrite_cache \
    --seed 1234 --deepspeed $DEEPSPEED_CONFIG
