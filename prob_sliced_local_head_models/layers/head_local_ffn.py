import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class HeadLocalFFN(nn.Module):
    """Block-diagonal FFN where each attention head has its own independent FFN subspace.
    
    Instead of a single Linear(d_model, dim_feedforward), each head gets an independent
    Linear(head_dim, ffn_per_head). Uses einsum for BLAS-efficient batched matmul —
    equivalent to block-diagonal weights without explicit loops.
    
    This eliminates gradient interference across head configurations during elastic
    width training: when slicing to k heads, heads 0..k-1 produce identical outputs
    regardless of whether heads k..nhead-1 exist.
    """

    def __init__(self, d_model: int, dim_feedforward: int, nhead: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert dim_feedforward % nhead == 0, "dim_feedforward must be divisible by nhead"

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.ffn_per_head = dim_feedforward // nhead

        # Block-diagonal weights stored pre-transposed — avoids .permute() at every forward pass.
        # w1: [nhead, head_dim, ffn_per_head]  (up-projection)
        # w2: [nhead, ffn_per_head, head_dim]  (down-projection)
        self.w1 = nn.Parameter(torch.empty(nhead, self.head_dim, self.ffn_per_head))
        self.b1 = nn.Parameter(torch.zeros(nhead, self.ffn_per_head))
        self.w2 = nn.Parameter(torch.empty(nhead, self.ffn_per_head, self.head_dim))
        self.b2 = nn.Parameter(torch.zeros(nhead, self.head_dim))

        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.w1, self.w2]:
            nn.init.xavier_uniform_(w)

    def forward(self, x: Tensor, effective_heads: Optional[int] = None) -> Tensor:
        """
        Args:
            x: [..., k * head_dim] where k = effective_heads or nhead
            effective_heads: number of active heads (None = all heads)
        Returns:
            Tensor of same shape as input: [..., k * head_dim]
        """
        k = effective_heads if effective_heads is not None else self.nhead

        # [..., k * head_dim] -> [..., k, head_dim]
        x_heads = x.unflatten(-1, (k, self.head_dim))

        # Up-project: w1 stored as [nhead, head_dim, ffn_per_head] — no permute needed
        h = torch.einsum('...kd,kdf->...kf', x_heads, self.w1[:k]) + self.b1[:k]

        # Activate and dropout (no intermediate norm — standard FFN has none)
        h = self.activation(h.flatten(-2)).unflatten(-1, (k, self.ffn_per_head))
        h = self.dropout(h)

        # Down-project: w2 stored as [nhead, ffn_per_head, head_dim] — no permute needed
        out = torch.einsum('...kf,kfd->...kd', h, self.w2[:k]) + self.b2[:k]

        return out.flatten(-2)                                 # [..., k * head_dim]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Compatibility: reshape standard FFN weights (linear1/linear2) into head-local format."""
        # Check if loading from a standard SlicedLinear checkpoint
        l1_weight_key = prefix.replace('ffn.', 'linear1.') + 'weight'
        l1_bias_key = prefix.replace('ffn.', 'linear1.') + 'bias'
        l2_weight_key = prefix.replace('ffn.', 'linear2.') + 'weight'
        l2_bias_key = prefix.replace('ffn.', 'linear2.') + 'bias'

        # If standard linear1/linear2 keys exist, reshape them
        if l1_weight_key in state_dict:
            # linear1.weight: [dim_feedforward, d_model] -> [nhead, ffn_per_head, head_dim]
            w1_flat = state_dict.pop(l1_weight_key)
            w1_reshaped = w1_flat.view(self.nhead, self.ffn_per_head, self.nhead, self.head_dim)
            # Take block-diagonal: each head's ffn_per_head rows, each head's head_dim cols
            w1_diag = torch.stack([w1_reshaped[i, :, i, :] for i in range(self.nhead)])
            state_dict[prefix + 'w1'] = w1_diag.permute(0, 2, 1).contiguous()  # -> [nhead, head_dim, ffn_per_head]

        if l1_bias_key in state_dict:
            b1_flat = state_dict.pop(l1_bias_key)
            state_dict[prefix + 'b1'] = b1_flat.view(self.nhead, self.ffn_per_head)

        if l2_weight_key in state_dict:
            # linear2.weight: [d_model, dim_feedforward] -> [nhead, head_dim, ffn_per_head]
            w2_flat = state_dict.pop(l2_weight_key)
            w2_reshaped = w2_flat.view(self.nhead, self.head_dim, self.nhead, self.ffn_per_head)
            w2_diag = torch.stack([w2_reshaped[i, :, i, :] for i in range(self.nhead)])
            state_dict[prefix + 'w2'] = w2_diag.permute(0, 2, 1).contiguous()  # -> [nhead, ffn_per_head, head_dim]

        if l2_bias_key in state_dict:
            b2_flat = state_dict.pop(l2_bias_key)
            state_dict[prefix + 'b2'] = b2_flat.view(self.nhead, self.head_dim)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
