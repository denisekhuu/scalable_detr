# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from typing import List, Optional
   
from .layers.linear import SlicedLinear
from .layers.mlp import SlicedMLP
from .layers.conv import SlicedConv2d


class SlicedDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = SlicedLinear(hidden_dim, num_classes + 1)
        self.bbox_embed = SlicedMLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = SlicedConv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss    

    def forward(self, samples: NestedTensor, effective_heads: Optional[int] = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        
    
        head_dim = self.transformer.d_model // self.transformer.nhead
        d_active = effective_heads * head_dim if effective_heads is not None else self.transformer.d_model
        pos_active = pos[-1][:, :d_active, :, :] if effective_heads is not None else pos[-1]
        hs = self.transformer(self.input_proj(src, d_out=d_active), mask, self.query_embed.weight, pos_active, effective_heads=effective_heads)[0]

        if effective_heads is not None:
            outputs_class = self.class_embed(hs, in_feature=hs.size(-1))
            # TODO confusing naming. effective_heads does not change the input
            outputs_coord = self.bbox_embed(hs, effective_heads=effective_heads).sigmoid()
        else:
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def forward_multi_head(self, samples: NestedTensor, heads: List[int]):
        """Single forward pass, then slice outputs for each head config.

        With block-diagonal FFN (HeadLocalFFN), head i's output is independent of
        heads i+1..N, so slicing the full-model output at k*head_dim gives a valid
        training signal for a k-head submodel — no separate forward passes needed.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        # Full forward pass with all heads
        hs = self.transformer(
            self.input_proj(src), mask,
            self.query_embed.weight, pos[-1],
        )[0]

        head_dim = self.transformer.d_model // self.transformer.nhead
        all_outputs = {}

        for k in heads:
            d_active = k * head_dim
            hs_k = hs[..., :d_active]

            outputs_class = self.class_embed(hs_k, in_feature=d_active)
            outputs_coord = self.bbox_embed(hs_k, effective_heads=k).sigmoid()

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            all_outputs[k] = out

        return all_outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # Use unbind(0) to convert tensors to a Python list before slicing,
        # avoiding the TracerWarning about iterating over a tensor.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class.unbind(0)[:-1], outputs_coord.unbind(0)[:-1])]


from models.detr import DETR, SetCriterion, PostProcess
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
import torch
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from typing import Optional

from .transformer.transformer import build_transformer


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = SlicedDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
