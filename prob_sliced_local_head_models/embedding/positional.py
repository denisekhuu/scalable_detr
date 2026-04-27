# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor



import torch
from torch import nn
import math

import torch
from torch import nn
import math
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Mathematical explanation of 2D sinusoidal positional encoding:

    Goal:
        Given a feature map x ∈ R^{B×C×H×W}, produce a positional embedding
        pos ∈ R^{B×(2·num_pos_feats)×H×W} using sine/cosine functions, 
        extending the 1D transformer positional encoding to 2D images.

    Steps:
    -------------------------------------------------------------------------
    1. Extract the mask (B×H×W):
        mask[b,h,w] = True  → padded pixel
        mask[b,h,w] = False → valid pixel
        not_mask = 1 − mask

    2. Compute coordinates using cumulative sums:
        y_embed[b,h,w] = Σ_{i=0..h} not_mask[b,i,w]
        x_embed[b,h,w] = Σ_{j=0..w} not_mask[b,h,j]

        If no padding:
            y_embed = h+1   (row index)
            x_embed = w+1   (column index)

    3. Optional normalization to [0, scale]:
        y_embed = ((y_embed - 0.5) / (y_max - ε)) * scale
        x_embed = ((x_embed - 0.5) / (x_max - ε)) * scale
        Default: scale = 2π

    4. Frequency denominators (Transformer-style):
        For k = 0 ... num_pos_feats-1:
            dim_t[k] = temperature^(2·floor(k/2) / num_pos_feats)

    5. Produce sine/cosine features for each pixel (h, w):

        Define:
            φ_x(h,w,k) = x_embed[h,w] / dim_t[k]
            φ_y(h,w,k) = y_embed[h,w] / dim_t[k]

        Then:
            PE_x(h,w,2k)   = sin(φ_x(h,w,k))
            PE_x(h,w,2k+1) = cos(φ_x(h,w,k))
            PE_y(h,w,2k)   = sin(φ_y(h,w,k))
            PE_y(h,w,2k+1) = cos(φ_y(h,w,k))

    6. Concatenate:
        pos(h,w) = [PE_y(h,w), PE_x(h,w)]
        Final shape: (B, 2·num_pos_feats, H, W)

    Summary:
        This creates smooth sinusoidal patterns encoding both X and Y
        coordinates, giving each pixel a unique position signature that
        attention layers can use to reason about spatial relationships.
    """


    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x): # B, C, H, W
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask

        # cumulative sum to get coordinates
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # cumsum over height
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # cumsum over width

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # frequency denominators that scale the positions to encode relative distances
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 1/dim_t is the frequency 
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # Even 
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, patch_size=50):
        super().__init__()
        self.row_embed = nn.Embedding(patch_size, num_pos_feats)
        self.col_embed = nn.Embedding(patch_size, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos



def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
