#!/bin/bash
#SBATCH --job-name=hydra_2head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=timm.2heads.log
#SBATCH --error=timm.2heads.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.2heads.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.2heads.${SLURM_JOBID}.log 
echo "2 Heads; hidden_dim = 64"  >> timm.2heads.${SLURM_JOBID}.log 
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path $WORK/workspace/data/coco --hidden_dim 64 --dim_feedforward 512 --nheads 2 --epochs 300 --batch_size 8 --world_size 2 --dist_url env:// --device cuda --output_dir $WORK/workspace/output/detr/dist/2heads/${SLURM_JOBID} >> timm.2heads.${SLURM_JOBID}.log
