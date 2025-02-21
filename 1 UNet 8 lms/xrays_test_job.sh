#!/bin/bash
#
#SBATCH --job-name=xrays
#SBATCH --partition=basic,gpu
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:4
#SBATCH --mem=20G
#SBATCH --time=2-12:00:00


module load conda
conda activate /lisc/user/stansfield/.conda/envs/xrays

set -e

python main.py --cfg configs/my_dataset_config_inference.yaml

rm -fr $TMPDIR
