#!/bin/bash
#SBATCH --job-name=hydra_3head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=timm.3heads.log
#SBATCH --error=timm.3heads.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.3heads.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.3heads.${SLURM_JOBID}.log 
echo "3 Heads; hidden_dim = 96 "  >> timm.3heads.${SLURM_JOBID}.log 
torchrun --master_port 29408 --nproc_per_node=2 main.py --coco_path $WORK/workspace/data/coco --world_size 2 --hidden_dim 96 --dim_feedforward 768 --nheads 3 --epochs 300 --batch_size 8 --world_size 2 --device cuda --output_dir $WORK/workspace/output/detr/dist/3heads/${SLURM_JOBID} >> timm.3heads.${SLURM_JOBID}.log