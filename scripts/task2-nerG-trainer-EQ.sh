set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
port=$(shuf -i25000-30000 -n1)
trainer_deepspeed="configs/deepspeed/ds1_llama.json"

pretrained="etri-lirs/egpt-1.3b-preview"
run_version="EAGLE-1B-supervised-EQ"

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/crossner_ai-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/crossner_ai-dev=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/crossner_literature-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/crossner_literature-train=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/crossner_music-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/crossner_music-dev=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/crossner_politics-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/crossner_politics-dev=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/crossner_science-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/crossner_science-dev=100-EQ.jsonll" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 1 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/mit-movie-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/mit-movie-dev=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 8 \
                --trainer_deepspeed $trainer_deepspeed

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
              task2-nerG-trainer.py "data/gner/each-EQ/mit-restaurant-train-EQ.jsonl" \
                        --eval_file "data/gner/each-sampled-EQ/mit-restaurant-dev=100-EQ.jsonl" \
                       --pretrained $pretrained \
                      --run_version $run_version \
      --per_device_train_batch_size 8 \
                --trainer_deepspeed $trainer_deepspeed
