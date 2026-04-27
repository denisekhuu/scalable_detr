#!/bin/bash
#SBATCH --job-name=hydra_1head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=timm.1head.log
#SBATCH --error=timm.1head.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.1head.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.1head.${SLURM_JOBID}.log 
echo "1 Head; hidden_dim = 32"  >> timm.1head.${SLURM_JOBID}.log 
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path $WORK/workspace/data/coco --hidden_dim 32 --dim_feedforward 256 --nheads 1 --epochs 300 --batch_size 8 --world_size 2 --dist_url env:// --device cuda --output_dir $WORK/workspace/output/detr/dist/1head/${SLURM_JOBID} >> timm.1head.${SLURM_JOBID}.log