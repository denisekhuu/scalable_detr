#!/bin/bash
#SBATCH --job-name=hyd_2_h100_ex1_distributed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=timm.log.log
#SBATCH --error=timm.log.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.${SLURM_JOBID}.log 
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path $WORK/workspace/data/coco --epochs 1 --batch_size 8 --world_size 2 --dist_url tcp://localhost:29500 --device cuda --output_dir $WORK/workspace/output/detr/dist/${SLURM_JOBID} >> timm.${SLURM_JOBID}.log