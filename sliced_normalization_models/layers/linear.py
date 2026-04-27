# ------------------------------------------------------------------------
# Copyright 2026 Denise-Phi Khuu. All Rights Reserved
# ------------------------------------------------------------------------
# Inspired from Kiel University (https://github.com/ds-kiel/HydraViT)
# Copyright 2024 Kiel University
# ------------------------------------------------------------------------
# Extended from PyTorch (https://github.com/pytorch/pytorch)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.linear import Linear

class SlicedLinear(Linear):

    def forward(self, input: Tensor, in_feature: int = None, out_feature: int = None) -> Tensor:

        """Applies a linear transformation to the incoming data: y = xA^T + b with optional slicing of input and output features.
        Args:
            input (Tensor): Input tensor of shape (..., in_features).
            in_feature (int, optional): If provided, slice the input tensor to this number of input features before applying the linear transformation.
            out_feature (int, optional): If provided, slice the output tensor to this number of output features after applying the linear transformation.
        Returns:
            Tensor: The transformed tensor."""

        # Skip validation and the no-op fast-path during tracing to avoid tensor-to-bool TracerWarnings.
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if in_feature is not None and (in_feature < 0 or in_feature > self.weight.shape[1]):
                raise ValueError(f"in_feature {in_feature} is out of bounds for weight with shape {self.weight.shape}")
            if out_feature is not None and (out_feature < 0 or out_feature > self.weight.shape[0]):
                raise ValueError(f"out_feature {out_feature} is out of bounds for weight with shape {self.weight.shape}")
            if (in_feature is None or in_feature == self.weight.shape[1]) and (out_feature is None or out_feature == self.weight.shape[0]):
                return F.linear(input, self.weight, self.bias)
        
    
        if out_feature is None:
            out_feature = self.weight.shape[0]

        if in_feature is None:
            in_feature = self.weight.shape[1]

        if len(input.shape) == 2:
            sliced_input = input[:, :in_feature]
            sliced_weight = self.weight[:out_feature, :in_feature]
            sliced_bias = self.bias[:out_feature] if self.bias is not None else None
            return F.linear(sliced_input, sliced_weight, sliced_bias)
        elif len(input.shape) == 3:
            sliced_input = input[:, :, :in_feature]
            sliced_weight = self.weight[:out_feature, :in_feature]
            sliced_bias = self.bias[:out_feature] if self.bias is not None else None
            return F.linear(sliced_input, sliced_weight, sliced_bias)
        else:
            sliced_input = input[..., :in_feature]
            sliced_weight = self.weight[:out_feature, :in_feature]
            sliced_bias = self.bias[:out_feature] if self.bias is not None else None
            return F.linear(sliced_input, sliced_weight, sliced_bias)