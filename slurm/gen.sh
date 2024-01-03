#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --job-name=encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
#SBATCH --constraint="gpu_40g+"
cd ..
nvidia-smi
source ~/.bashrc
CONFIG_NAME='rag' python3 run.py run_name='test'

#python3 run.py +run_name='encode'

 

