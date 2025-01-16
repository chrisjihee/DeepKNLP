set -x
port=$(shuf -i25000-30000 -n1)
TRAIN_JSON_DIR=data/gner/zero-shot-train.jsonl
VALID_JSON_DIR=data/gner/zero-shot-test-min.jsonl
MODEL_NAME_OR_PATH=etri-lirs/egpt-1.3b-preview
OUTPUT_DIR=output/GNER/EAGLE-1B-supervised
RUN_NAME=train_eagle_1b_supervised-base
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero1_llama.json

~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include="localhost:0,1,2,3" --master_port $port task2-nerG-trainer1.py \
    --do_train --do_eval --predict_with_generate \
    --train_json_dir $TRAIN_JSON_DIR \
    --valid_json_dir $VALID_JSON_DIR \
    --no_load_gner_customized_datasets \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --bf16 True --tf32 True \
    --lr_scheduler_type cosine \
    --learning_rate 2e-05 \
    --warmup_ratio 0.04 --weight_decay 0. \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --num_train_epochs 0.5 \
    --logging_strategy steps --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir --overwrite_cache \
    --seed 1234 --deepspeed $DEEPSPEED_CONFIG
