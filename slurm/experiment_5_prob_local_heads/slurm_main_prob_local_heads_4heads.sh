#!/bin/bash
#SBATCH --job-name=hydra_prob_local_heads_4head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=timm.4heads.prob_local_heads.log
#SBATCH --error=timm.4heads.prob_local_heads.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 
echo "Experiment 6: Prob Local Heads DETR"  >> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 
echo "4 Heads; hidden_dim = 192 "  >> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 
echo "Scalable Prob Local Heads DETR"  >> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 
torchrun --master_port 29409 --nproc_per_node=2 prob_sliced_local_head_main.py --coco_path $WORK/workspace/data/coco --hidden_dim 192 --dim_feedforward 1536 --nheads 6 --epochs 300  --world_size 2 --batch_size 8 --device cuda --output_dir $WORK/workspace/output/detr/dist/4heads/prob_local_heads/${SLURM_JOBID} >> timm.4heads.prob_local_heads.${SLURM_JOBID}.log 2>&1