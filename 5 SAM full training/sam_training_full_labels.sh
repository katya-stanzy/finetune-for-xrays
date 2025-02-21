#!/bin/bash
#
#SBATCH --job-name=huggingface
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT
#SBATCH --partition=basic,gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --gres=shard:l40s:1
#SBATCH --time=2-12:00:00

module load conda
conda activate  /lisc/user/stansfield/.conda/envs/medsam
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Exit the slurm script if a command fails
set -e

# Run the assembly
python main_training_full_labels.py

# If we reached this point, the assembly succeeded. We clean up resources.
rm -rf $TMPDIR