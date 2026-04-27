import torch
import json
from pathlib import Path

import detr.util.misc as utils
from torch.utils.data import DataLoader

from detr.datasets import build_dataset, get_coco_api_from_dataset
from detr.models import build_model
from detr.engine import evaluate 

import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from args import HeadArgs

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

from working.utils.args.one_head_args import HeadArgs

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    models_to_eval = [
        (1, r"C:\workspace\ml\workspace\master\original\results\1head\4920974\checkpoint0299.pth"),
        (2, r"C:\workspace\ml\workspace\master\original\results\2heads\4954265\checkpoint0299.pth"),
        (3, r"C:\workspace\ml\workspace\master\original\results\3heads\checkpoint0299.pth"),
        (4, r"C:\workspace\ml\workspace\master\original\results\4heads\checkpoint0259.pth")
    ]

    for n_heads, model_path_str in models_to_eval:
        print(f"\n============================\nEvaluating {n_heads} heads model\n============================")
        args = HeadArgs(number_of_heads=n_heads)

        dataset_val = build_dataset(image_set='val', args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        base_ds = get_coco_api_from_dataset(dataset_val)

        model_path = Path(model_path_str)
        device = torch.device(args.device)
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                data_loader_val, base_ds, device, args.output_dir)

        output_path = Path(str("baseline_test_stats_heads{}.json").format(args.nheads))
        with open(output_path, "w") as f:
            json.dump(make_serializable(test_stats), f, indent=2)

        print(f"Saved to {output_path.resolve()}")