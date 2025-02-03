set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
port=$(shuf -i25000-30000 -n1)

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_ai-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_ai-dev=100.jsonl" \
                       --pretrained "google/flan-t5-large" \
                      --run_version "FLAN-T5-Large" \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed "configs/deepspeed/ds2_llama.json"

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_ai-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_ai-dev=100.jsonl" \
                       --pretrained "meta-llama/Llama-3.2-1B" \
                      --run_version "Llama-3-1B" \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed "configs/deepspeed/ds2_llama.json"

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_ai-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_ai-dev=100.jsonl" \
                       --pretrained "etri-lirs/egpt-1.3b-preview" \
                      --run_version "EAGLE-1B-supervised" \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed "configs/deepspeed/ds2_llama.json"
