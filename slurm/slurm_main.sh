#!/bin/bash
#SBATCH --job-name=hydra_origin_detr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=log.log
#SBATCH --error=log.log

module load python
source $WORK/workspace/env/bin/activate

python -u main.py --coco_path $WORK/workspace/data/coco --hidden_dim 128 --dim_feedforward 1024 --nheads 4 --epochs 1 --batch_size 2 --device cuda --output_dir $WORK/workspace/output/detr/07012026 >> log.log