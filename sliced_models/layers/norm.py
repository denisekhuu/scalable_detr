import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm

class SlicedLayerNorm(LayerNorm):

    def forward(self, input: Tensor, effective_embed_dim: int = None) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs with optional slicing of the embedding dimension.
        Args:
            input (Tensor): Input tensor of shape (..., embedding_dim).
            effective_embed_dim (int, optional): If provided, slice the input tensor to this embedding dimension before applying LayerNorm. 
        Returns:
            Tensor: The normalized tensor.
        """
        if effective_embed_dim is not None:  # use explicit None check to avoid tensor-to-bool (TracerWarning)
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                if effective_embed_dim < 0 or effective_embed_dim > self.normalized_shape[0]:
                    raise ValueError(f"effective_embed_dim {effective_embed_dim} is out of bounds for normalized_shape {self.normalized_shape}")
            
            input = input[..., :effective_embed_dim]
            return F.layer_norm(input, (effective_embed_dim,),
                            self.weight[:effective_embed_dim],
                            self.bias[:effective_embed_dim],
                            self.eps)
        
        return F.layer_norm(input,
                            self.normalized_shape,
                            self.weight,
                            self.bias,
                            self.eps)
    