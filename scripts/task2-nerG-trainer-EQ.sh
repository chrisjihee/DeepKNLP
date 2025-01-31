set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
port=$(shuf -i25000-30000 -n1)

python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
    task2-nerG-trainer.py "data/gner/each-EQ/crossner_ai-train-EQ.jsonl" \
              --eval_file "data/gner/each-sampled-EQ/crossner_ai-dev=100-EQ.jsonl" \
             --pretrained "etri-lirs/egpt-1.3b-preview" \
            --run_version "EAGLE-1B-supervised-EQ" \
      --trainer_deepspeed "configs/deepspeed/ds0_llama.json"
