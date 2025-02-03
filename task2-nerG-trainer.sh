set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
port=$(shuf -i25000-30000 -n1)

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_ai-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_ai-dev=100.jsonl" \
                       --pretrained "etri-lirs/egpt-1.3b-preview" \
                      --run_version "EAGLE-1B-supervised" \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed "configs/deepspeed/ds1_llama.json"
