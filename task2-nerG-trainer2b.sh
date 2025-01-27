set -x
~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:4,5,6,7 --master_port 30000 task2-nerG-trainer2.py --trainer_deepspeed configs/deepspeed/ds0_llama.json
~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:4,5,6,7 --master_port 30000 task2-nerG-trainer2.py --trainer_deepspeed configs/deepspeed/ds1_llama.json
~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:4,5,6,7 --master_port 30000 task2-nerG-trainer2.py --trainer_deepspeed configs/deepspeed/ds2_llama.json
~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:4,5,6,7 --master_port 30000 task2-nerG-trainer2.py --trainer_deepspeed configs/deepspeed/ds3_llama.json
~/miniforge3/envs/DeepKNLP/bin/python -m accelerate.commands.launch --config_file configs/accelerate/gpu4b-ds0.yaml task2-nerG-trainer2.py --accelerate_deepspeed
~/miniforge3/envs/DeepKNLP/bin/python -m accelerate.commands.launch --config_file configs/accelerate/gpu4b-ds1.yaml task2-nerG-trainer2.py --accelerate_deepspeed
~/miniforge3/envs/DeepKNLP/bin/python -m accelerate.commands.launch --config_file configs/accelerate/gpu4b-ds2.yaml task2-nerG-trainer2.py --accelerate_deepspeed
~/miniforge3/envs/DeepKNLP/bin/python -m accelerate.commands.launch --config_file configs/accelerate/gpu4b-ds3.yaml task2-nerG-trainer2.py --accelerate_deepspeed
