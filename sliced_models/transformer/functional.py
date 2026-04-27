"""Functional interface."""

from typing import Optional, TYPE_CHECKING

from torch import  Tensor
import torch
from torch.overrides import (
    has_torch_function,
)



if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    DType = int

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from torch.nn.functional import pad, softmax, dropout
from torch.nn.functional import _mha_shape_check, _canonical_mask, _check_key_padding_mask
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.functional import handle_torch_function, _none_or_dtype

import math

linear = torch._C._nn.linear



def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), (
        f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    )
    assert b_k is None or b_k.shape == (Eq,), (
        f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    )
    assert b_v is None or b_v.shape == (Eq,), (
        f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    )
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> list[Tensor]:
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            proj = linear(q, w, b)
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _select_heads_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
    effective_heads: Optional[int] = None,
    num_heads: int = 8,
    head_dim: int = 32,
    embed_dim: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    
    """
    Performs in-projection of Q, K, V using only a subset of attention heads and filters Q, K, V before projection.
    This makes sure that the future computations only involve the selected heads, i.e. concat(heads).dim = filtered(q).dim = filtered(k).dim filtered(v).dim = effective_heads  * head_dim

    Args:
        q, k, v: query, key, and value tensors.
        w_q, w_k, w_v: weight tensors for q, k, and v respectively.
        b_q, b_k, b_v: optional bias tensors for q, k, and v respectively.
        effective_heads: number of effective attention heads to use.
        num_heads: total number of attention heads.
        head_dim: dimension of each attention head.
        embed_dim: total embedding dimension.
    Returns:    
        A tuple containing the projected Q, K, and V tensors.
    """

    if effective_heads is None:
        return _in_projection(q, k, v, w_q, w_k, w_v, b_q, b_k, b_v)  
    
    if effective_heads <= 0:
        raise ValueError(f"effective_heads must be a positive integer, but got {effective_heads}")  
    actual_embed_dim = effective_heads * head_dim
    
    q = q[:, :, :actual_embed_dim]
    k = k[:, :, :actual_embed_dim]
    v = v[:, :, :actual_embed_dim]
    

    w_q = w_q.view(num_heads, head_dim, embed_dim)[:effective_heads, :, :actual_embed_dim].reshape(actual_embed_dim, actual_embed_dim)
    w_k = w_k.view(num_heads, head_dim, embed_dim)[:effective_heads, :, :actual_embed_dim].reshape(actual_embed_dim, actual_embed_dim)
    w_v = w_v.view(num_heads, head_dim, embed_dim)[:effective_heads, :, :actual_embed_dim   ].reshape(actual_embed_dim, actual_embed_dim)
    if b_q is not None:
        b_q = b_q[:actual_embed_dim].reshape(-1)
    if b_k is not None: 
        b_k = b_k[:actual_embed_dim].reshape(-1)
    if b_v is not None:
        b_v = b_v[:actual_embed_dim].reshape(-1)
        
    return _in_projection(q, k, v, w_q, w_k, w_v, b_q, b_k, b_v)  

 
def _select_heads_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    effective_heads: Optional[int] = None,
    num_heads: int = 8,
    head_dim: int = 32,
    embed_dim: int = 256,
    ) -> list[Tensor]:

    """
    Performs in-projection of Q, K, V using only a subset of attention heads and filters Q, K, V before projection.
    This makes sure that the future computations only involve the selected heads, i.e. concat(heads).dim = filtered(q).dim = filtered(k).dim filtered(v).dim = filtered(v).dim = effective_heads  * head_dim
    Args:
        q, k, v: query, key, and value tensors.
        w: weight tensor of shape (3 * embed_dim, embed_dim).
        b: optional bias tensor of shape (3 * embed_dim).
        effective_heads: number of effective attention heads to use.
        num_heads: total number of attention heads.
        head_dim: dimension of each attention head.
        embed_dim: total embedding dimension.
        
        Returns:
        A list containing the projected Q, K, and V tensors.
    """

    if effective_heads is None:
        return _in_projection_packed(q, k, v, w, b)  
    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        if effective_heads is not None and effective_heads <= 0:
            raise ValueError(f"effective_heads must be a positive integer, but got {effective_heads}")      
        
    actual_embed_dim = effective_heads * head_dim
    
    q = q[:, :, :actual_embed_dim]
    k = k[:, :, :actual_embed_dim]
    v = v[:, :, :actual_embed_dim]
    
    w = w.view(3, num_heads, head_dim, embed_dim)
    w = w[:, :effective_heads, :, :actual_embed_dim].reshape(3 * actual_embed_dim, actual_embed_dim)
    
    if b is not None:
        b = b.view(3, num_heads, head_dim)
        b = b[:, :effective_heads, :].reshape(3 * actual_embed_dim)
        
    return _in_projection_packed(q, k, v, w, b)

def _select_heads_output_projection(
    attn_output: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor] = None,
    actual_embed_dim: int = 256,
) -> Tensor:
    """
    Performs output projection using only a subset of attention heads.
    This makes sure that the future computations only involve the selected heads, i.e. concat(heads).dim = effective_heads  * head_dim
    Args:
        attn_output: attention output tensor.
        out_proj_weight: output projection weight tensor of shape (embed_dim, embed_dim).
        out_proj_bias: optional output projection bias tensor of shape (embed_dim).
        effective_heads: number of effective attention heads to use.
        num_heads: total number of attention heads.
        head_dim: dimension of each attention head.
        embed_dim: total embedding dimension.
        
    Returns:
        The projected attention output tensor.
    """

    out_proj_weight = out_proj_weight[:actual_embed_dim, :actual_embed_dim]
    
    if out_proj_bias is not None:
        out_proj_bias = out_proj_bias[:actual_embed_dim]
        
    return linear(attn_output, out_proj_weight, out_proj_bias)


# TODO: check if I want to use embed_dim_to_check and original_embed_dim
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    effective_heads: Optional[int] = None,
    original_embed_dim: Optional[int] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    r"""Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
        effective_heads: If provided, uses only a subset of attention heads for the computation.
        original_embed_dim: If provided, uses this value as the original embedding dimension. This is useful when q has been sliced before passing it to this function. 
            This ensures that the in-projection and out-projection weights are correctly shaped. 


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """

    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    
    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, q_embed_dim = query.shape

    # Set default to embed_dim from query if not provided
    embed_dim = original_embed_dim if original_embed_dim is not None else q_embed_dim
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            is_causal = False


    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )

    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )

    if torch.jit.is_scripting():
        if effective_heads is not None:
            assert isinstance(effective_heads, int) and effective_heads > 0, (
                f"effective_heads must be a positive integer, but got {effective_heads}"
            )
        
            assert effective_heads <= num_heads if effective_heads is not None else True, (
                f"effective_heads {effective_heads} must be less than or equal to num_heads {num_heads}"
            )

    effective_heads = effective_heads if effective_heads is not None else num_heads
    actual_embed_dim = effective_heads * head_dim

    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        if use_separate_proj_weight:
            assert key.shape[:2] == value.shape[:2], (
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )
        else:
            assert key.shape == value.shape, (
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

    if not use_separate_proj_weight:
        assert in_proj_weight is not None, (
            "use_separate_proj_weight is False but in_proj_weight is None"
        )
        if effective_heads is not None and effective_heads < num_heads:
            q, k, v = _select_heads_projection_packed(
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                effective_heads=effective_heads,
                num_heads=num_heads,
                head_dim=head_dim,
                embed_dim=embed_dim,
            )
        else:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, (
            "use_separate_proj_weight is True but q_proj_weight is None"
        )
        assert k_proj_weight is not None, (
            "use_separate_proj_weight is True but k_proj_weight is None"
        )
        assert v_proj_weight is not None, (
            "use_separate_proj_weight is True but v_proj_weight is None"
        )
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)

        if effective_heads is not None and effective_heads < num_heads:
            q, k, v = _select_heads_projection(
                query,
                key,
                value,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
                effective_heads=effective_heads,
                num_heads=num_heads,
                head_dim=head_dim,
                embed_dim=embed_dim,
            )
        else:
            q, k, v = _in_projection(
                query,
                key,
                value,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
            )


    if attn_mask is not None:
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * effective_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.view(tgt_len, bsz * effective_heads, head_dim).transpose(0, 1) # (bsz * effective_heads, tgt_len, head_dim)
    if static_k is None:
        k = k.view(k.shape[0], bsz * effective_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * effective_heads, (
            f"expecting static_k.size(0) of {bsz * effective_heads}, but got {static_k.size(0)}"
        )
        assert static_k.size(2) == head_dim, (
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        )
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * effective_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * effective_heads, (
            f"expecting static_v.size(0) of {bsz * effective_heads}, but got {static_v.size(0)}"
        )
        assert static_v.size(2) == head_dim, (
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        )
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * effective_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _check_key_padding_mask(key_padding_mask, src_len, bsz)

        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, effective_heads, -1, -1)
            .reshape(bsz * effective_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        _B, _Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / E)  # avoid float() cast on shape value (TracerWarning)

        assert not (is_causal and attn_mask is None), (
            "FIXME: is_causal not implemented for need_weights"
        )

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)


        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, actual_embed_dim)
        )
        
        # Adjust out_proj_weight if effective_heads is specified
        if effective_heads is not None and effective_heads < num_heads:
            attn_output = _select_heads_output_projection(attn_output, out_proj_weight, out_proj_bias, actual_embed_dim)
        else:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, effective_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, effective_heads, -1, src_len)

        q = q.view(bsz, effective_heads, tgt_len, head_dim)
        k = k.view(bsz, effective_heads, src_len, head_dim)
        v = v.view(bsz, effective_heads, src_len, head_dim)

        # change mask shape.
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, actual_embed_dim)
        )

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        return attn_output, None
