#!/bin/bash

#SBATCH --job-name=invGAN_scoring_FID
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=peilianglin@sjtu.edu.cn
#SBATCH --gres=gpu:1

module load miniconda3/4.6.14-gcc-4.8.5

source activate linpeiliang

python < scoring_FID_MSE.py
