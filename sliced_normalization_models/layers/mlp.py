# ------------------------------------------------------------------------
# Copyright 2026 Denise-Phi Khuu. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import nn
from torch.nn import functional as F
from .linear import SlicedLinear

class SlicedMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(SlicedLinear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x, effective_heads=None):
        if effective_heads is not None:
            for i, layer in enumerate(self.layers):
                if i < self.num_layers - 1:
                    x = F.relu(layer(x, in_feature=x.size(-1)))
                else:
                    x = layer(x, in_feature=x.size(-1))
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
