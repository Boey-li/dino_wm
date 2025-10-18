#!/bin/bash
#SBATCH --job-name=dinowm_demo150
#SBATCH --output=sbatch_logs/dinowm_demo150.out
#SBATCH --error=sbatch_logs/dinowm_demo150.err
#SBATCH --partition="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/miniconda3/etc/profile.d/conda.sh
conda activate robomimic

python train.py \
    --config-name train.yaml \
    env=robomimic \
    frameskip=5 \
    num_hist=3 \
    num_exp_trajs=150 \
    num_exp_val_trajs=5 \
    wandb_exp_name='demo150'