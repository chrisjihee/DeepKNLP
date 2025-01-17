set -x
#~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:0,1,2,3 --master_port 30000 task2-nerG-trainer2.py train
~/miniforge3/envs/DeepKNLP/bin/python -m accelerate.commands.launch --config_file configs/accelerate/multi-gpu-4.yaml task2-nerG-trainer2.py train
