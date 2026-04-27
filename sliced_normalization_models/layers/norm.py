import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm

class SlicedGroupNorm(LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, number_slice=4):
        super().__init__(normalized_shape, eps, elementwise_affine)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.max_embed_dim = normalized_shape[0]
        self.number_slice = number_slice
        self.slice_dim = self.max_embed_dim // self.number_slice
        if self.slice_dim * self.number_slice != self.max_embed_dim:
            raise ValueError(f"max_embed_dim {self.max_embed_dim} must be divisible by number_slice {self.number_slice} to ensure equal slice dimensions.")
    
    def forward(self, input: Tensor, effective_embed_dim: int = None) -> Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs with optional slicing of the embedding dimension.
        For each slice, normalization is applied individually.
        Args:
            input (Tensor): Input tensor of shape (..., embedding_dim).
            effective_embed_dim (int, optional): If provided, slice the input tensor to this embedding dimension before applying LayerNorm.
        Returns:
            Tensor: The normalized tensor.
        """
        k = self.max_embed_dim if effective_embed_dim is None else effective_embed_dim
        if k % self.slice_dim != 0 or k > self.max_embed_dim:
            raise ValueError(f"effective_embed_dim {k} must be a multiple of slice_dim {self.slice_dim} and <= max_embed_dim {self.max_embed_dim}")
        num_active_slices = k // self.slice_dim
        input_sliced = input[..., :k]
        out_slices = []
        for i in range(num_active_slices):
            start = i * self.slice_dim
            end = start + self.slice_dim
            slice_input = input_sliced[..., start:end]
            slice_weight = self.weight[start:end] if self.weight is not None else None
            slice_bias = self.bias[start:end] if self.bias is not None else None
            normed = F.layer_norm(slice_input, (self.slice_dim,), slice_weight, slice_bias, self.eps)
            out_slices.append(normed)
        normed_concat = torch.cat(out_slices, dim=-1)
        return normed_concat
    
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Ensures backward compatibility when loading standard LayerNorm checkpoints.
        Intercepts LayerNorm weight/bias and avoids "Unexpected key" runtime errors 
        if this layer is instantiated without elementwise_affine.
        """
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        
        # If the checkpoint contains weights/bias but this layer was created 
        # with elementwise_affine=False, we remove them from the state_dict 
        # to prevent "Unexpected key" errors.
        if not self.elementwise_affine:
            state_dict.pop(weight_key, None)
            state_dict.pop(bias_key, None)
            
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
