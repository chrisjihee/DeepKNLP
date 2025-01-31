set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
port=$(shuf -i25000-30000 -n1)
trainer_deepspeed="configs/deepspeed/ds1_llama.json"

pretrained="etri-lirs/egpt-1.3b-preview"
run_version="EAGLE-1B-supervised"

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_ai-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_ai-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_literature-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_literature-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_music-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_music-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_politics-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_politics-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/crossner_science-train.jsonl" \
                        --eval_file "data/gner/each-sampled/crossner_science-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/mit-movie-train.jsonl" \
                        --eval_file "data/gner/each-sampled/mit-movie-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 8 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each/mit-restaurant-train.jsonl" \
                        --eval_file "data/gner/each-sampled/mit-restaurant-dev=100.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 8 \
                --trainer_deepspeed $trainer_deepspeed
