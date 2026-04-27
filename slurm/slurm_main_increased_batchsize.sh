#!/bin/bash
#SBATCH --job-name=hydra_increased_batchsize_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=log.log
#SBATCH --error=log.log

module load python
source $WORK/workspace/env/bin/activate

python -u main.py --coco_path $WORK/workspace/data/coco --epochs 1 --batch_size 8 --device cuda --output_dir $WORK/workspace/output/detr/v3 &> timm.${SLURM_JOBID}.log
