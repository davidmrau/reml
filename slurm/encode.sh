#!/bin/bash

#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --job-name=encode
#SBATCH --output=/beegfs/scratch/user/drau/research/reml/slurm/logs/%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="gpu_32g+"
#SBATCH --mem=1
source ~/.bashrc
port=$(shuf -i 29500-29599 -n 1)
nvidia-smi
cd ..
python3 run.py +run_name='encode'

 

