#!/bin/bash
#SBATCH --job-name=hyd_2_a100_ex1_distributed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=timm.log.log
#SBATCH --error=timm.log.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.${SLURM_JOBID}.log 
echo "Distributed training: gpu:a100:2"  >> timm.${SLURM_JOBID}.log 
echo "Heads: 1"  >> timm.${SLURM_JOBID}.log 
python -u -m torch.distributed.launch --he--nproc_per_node=2 --use_env main.py --coco_path $WORK/workspace/data/coco --epochs 1 --batch_size 8 --world_size 2 --dist_url env:// --device cuda --output_dir $WORK/workspace/output/detr/dist/${SLURM_JOBID} >> timm.${SLURM_JOBID}.log