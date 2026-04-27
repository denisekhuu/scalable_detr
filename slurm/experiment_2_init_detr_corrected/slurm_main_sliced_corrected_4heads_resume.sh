#!/bin/bash
#SBATCH --job-name=hydra_600_corrected_sliced_4head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=timm.4heads.corrected_sliced.resumed.log
#SBATCH --error=timm.4heads.corrected_sliced.resumed.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 
echo "Experiment 1: Naive Sliced DETR"  >> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 
echo "4 Heads; hidden_dim = 128 "  >> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 
echo "600 epochs Scalable DETR"  >> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 
torchrun --master_port 29403 --nproc_per_node=2 sliced_main.py --coco_path $WORK/workspace/data/coco --resume /work/ti5/cdk9132/workspace/output/detr/dist/4heads/sliced/corrected/5085080/checkpoint.pth --hidden_dim 128 --dim_feedforward 1024 --nheads 4 --epochs 600 --world_size 2 --batch_size 8 --device cuda --output_dir $WORK/workspace/output/detr/dist/4heads/sliced/corrected/resumed/${SLURM_JOBID} >> timm.4heads.corrected_sliced.resumed.${SLURM_JOBID}.log 2>&1