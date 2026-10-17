#!/bin/bash

#SBATCH --job-name=jit_training
#SBATCH --partition=qdefault
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate jit

ulimit -t unlimited

echo "================================"
echo "JOB: $SLURM_JOB_ID"
echo "HOST: $(hostname)"
echo "GPU:"
nvidia-smi
echo "================================"

python main.py \
    --data_path ../datahome/RSCD_dataset_1million_reorganized/ \
    --num_epochs 200 \
    --resume ../datahome/jit_data/output/jitrans_20260615_155632_1gpu/last.pt \
    --model_save_dir ../datahome/jit_data/output/jitrans_20260615_155632_1gpu/