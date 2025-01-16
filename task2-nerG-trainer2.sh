set -x
~/miniforge3/envs/DeepKNLP/bin/python -m deepspeed.launcher.runner --include=localhost:0,1,2,3 --master_port 30000 task2-nerG-trainer2.py train
