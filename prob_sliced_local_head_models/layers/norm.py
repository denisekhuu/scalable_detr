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
        Applies per-group LayerNorm using a single F.group_norm kernel instead of a
        Python loop of F.layer_norm calls. One CUDA kernel launch regardless of slice count.
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
        x = input[..., :k]
        w = self.weight[:k] if self.weight is not None else None
        b = self.bias[:k]   if self.bias   is not None else None
        # F.group_norm expects [N, C, *] — fold all leading dims into N
        shape = x.shape
        x_2d = x.reshape(-1, k)                               # [N*seq, k]
        out = F.group_norm(x_2d.unsqueeze(-1), num_active_slices, w, b, self.eps).squeeze(-1)
        return out.reshape(shape)
    
    
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
