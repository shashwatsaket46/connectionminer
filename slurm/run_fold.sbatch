#!/bin/bash
# One fold (5 seeds x train/val/test) per array task:
#   sbatch --array=0-9 slurm/run_fold.sbatch visual
# Edit the partition/GPU lines for your cluster. fp64 matters: A100/H100/H200 are
# fast at it, L40S-class cards are not.
#SBATCH --job-name=cm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
DATASET=${1:?usage: sbatch --array=0-9 slurm/run_fold.sbatch <visual|motor|cx>}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
python -m connectionminer run "$DATASET" --fold "$SLURM_ARRAY_TASK_ID"
