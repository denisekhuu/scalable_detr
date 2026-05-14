# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
from pathlib import Path

import json
import torch
import time

import numpy as np
import tqdm

from sliced_models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list

from flop_count import flop_count
from models.backbone import build_backbone
from sliced_models.transformer.transformer import build_transformer
from evaluation.args import HeadArgs
from torch import nn
from sliced_models.layers.conv import SlicedConv2d

class BackboneWrapper(nn.Module):
    """Wraps the backbone so it accepts plain tensors instead of NestedTensor,
    allowing torch.jit tracing used by flop_count to work."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tensors: torch.Tensor, mask: torch.Tensor):
        nt = nested_tensor_from_tensor_list([tensors[i] for i in range(tensors.shape[0])])
        # override the mask with the pre-computed one so shapes stay static
        nt.mask = mask
        features, pos = self.backbone(nt)
        # return the last feature map and its position encoding as plain tensors
        src, src_mask = features[-1].decompose()
        return src, src_mask, pos[-1]
                
def get_dataset(coco_path):
    """
    Gets the COCO dataset used for computing the flops on
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset


def warmup(model, inputs, N=10):
    for i in range(N):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            out = model(*inputs)
        elif isinstance(inputs, dict):
            out = model(**inputs)
        else:
            out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            out = model(*inputs)
        elif isinstance(inputs, dict):
            out = model(**inputs)
        else:
            out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t

def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()

if __name__ == '__main__':
    args = HeadArgs(number_of_heads=8)

    # get the first 100 images of COCO val2017
    PATH_TO_COCO = args.coco_path
    dataset = get_dataset(PATH_TO_COCO)
    images = []
    for idx in range(100):
        img, t = dataset[idx]
        images.append(img)

    device = torch.device('cuda')
    results = {}
    for model_name in ['sliced_detr_resnet50']:
        results[model_name] = []
        with torch.no_grad():
            for head in range(1, args.nheads + 1):
                detr_tmp = []
                backbone_temp = []
                transformer_temp = []
                tmp2 = []
                # Update the number of heads in the args
                args = HeadArgs(number_of_heads=head)
                model = build_model(args)[0].to(device)
                # Rebuild the backbone and transformer with the updated args
                backbone = build_backbone(args).to(device)
                transformer = build_transformer(args).to(device)

                # Compute FLOPS and time for each image
                for img in tqdm.tqdm(images):
                    detr_res = flop_count(model, ([img.to(device)], head))
                    detr_t = measure_time(model, ([img.to(device)], head))
                    detr_tmp.append(sum(detr_res.values()))
                    tmp2.append(detr_t)

                    ## Backbone breakdown
                    backbone_nt = nested_tensor_from_tensor_list([img.to(device)])
                    backbone_wrapper = BackboneWrapper(backbone)
                    backbone_res = flop_count(backbone_wrapper, (backbone_nt.tensors, backbone_nt.mask))
                    backbone_temp.append(sum(backbone_res.values()))

                    ## Transformer breakdown
                    input_proj = SlicedConv2d(backbone.num_channels, args.hidden_dim, kernel_size=1).to(device)
                    input = nested_tensor_from_tensor_list([img.to(device)])
                    query_embed = nn.Embedding(args.num_queries, args.hidden_dim).to(device)
                    features, pos = backbone(input)
                    src, mask = features[-1].decompose()
                    head_dim = args.hidden_dim // args.nheads
                    d_active = head * head_dim 
                    pos_active = pos[-1][:, :d_active, :, :]
                    transformer_res = flop_count(transformer, (input_proj(src, None, d_out=d_active), mask, query_embed.weight, pos_active, head))
                    transformer_temp.append(sum(transformer_res.values()))
            
                results[model_name].append({
                    'heads': head,
                    'detr_flops': fmt_res(np.array(detr_tmp)), 
                    'backbone_flops': fmt_res(np.array(backbone_temp)), 
                    'transformer_flops': fmt_res(np.array(transformer_temp)), 
                    'time': fmt_res(np.array(tmp2))
                })
                print(results[model_name][-1])


    json_results = {
        model: [
            {k: [float(x) for x in v] if isinstance(v, tuple) else v
            for k, v in entry.items()}
            for entry in entries
        ]
        for model, entries in results.items()
    }
    output_path = Path('flops_breakdown_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'Results saved to {output_path.resolve()}')