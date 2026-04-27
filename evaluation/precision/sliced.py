import  sys
sys.path.append('C:\\workspace\\ml\\workspace\\master\\original')
sys.path.append('C:\\workspace\\ml\\workspace\\master\\original\\detr')
import torch
import json
from pathlib import Path

import detr.util.misc as utils
from torch.utils.data import DataLoader

from detr.datasets import build_dataset, get_coco_api_from_dataset
from detr.sliced_models import build_model
from detr.sliced_models.training.engine import evaluate as evaluate_sliced

model_name = "4heads_sliced_checkpoint0119"

def make_serializable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj

from args import HeadArgs

import logging
logging.basicConfig(level=logging.INFO)
args = HeadArgs(number_of_heads=4)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

dataset_val = build_dataset(image_set='val', args=args)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
base_ds = get_coco_api_from_dataset(dataset_val)


model_path= r"C:\workspace\ml\workspace\master\original\results\4heads\sliced\checkpoint0119.pth"
device = torch.device(args.device)
model, criterion, postprocessors = build_model(args)
model.to(device)
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'])

heads = [1, 2, 3, 4]

sliced_test_stats, sliced_coco_evaluator = evaluate_sliced(model, criterion, postprocessors,
                                        data_loader_val, base_ds, device, args.output_dir, heads=heads)


output_path = Path("sliced_test_stats_heads{}.json".format(model_name))
with open(output_path, "w") as f:
    json.dump(make_serializable(sliced_test_stats), f, indent=2)

print(f"Saved to {output_path.resolve()}")