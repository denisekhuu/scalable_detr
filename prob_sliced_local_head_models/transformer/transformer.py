import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
import copy
from torch import nn

from typing import Optional
from torch import Tensor
from .mha import SlicedMultiheadAttention
from ..layers.linear import SlicedLinear
from ..layers.norm import SlicedGroupNorm
from ..layers.head_local_ffn import HeadLocalFFN


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _slice_embedding_dim(tensor, effective_embed_dim: int = None):
    """Slice the embedding dimension of a tensor if effective_embed_dim is provided."""
    if effective_embed_dim is not None:  # explicit None check avoids tensor-to-bool (TracerWarning)
        return tensor[..., :effective_embed_dim]
    return tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class SlicedTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = nhead  
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.self_attn = SlicedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Head-local FFN: each head has independent FFN weights (block-diagonal)
        self.ffn = HeadLocalFFN(d_model, dim_feedforward, nhead, dropout=dropout, activation=activation)

        self.norm1 = SlicedGroupNorm(d_model, number_slice=nhead)
        self.norm2 = SlicedGroupNorm(d_model, number_slice=nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     effective_heads: Optional[int] = None,
                     ):
        effective_dim = self.head_dim * effective_heads if effective_heads is not None else self.d_model
        effective_heads = effective_heads if effective_heads is not None else self.num_heads
        q = k = self.with_pos_embed(src, pos)
        out, _ = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, effective_heads=effective_heads)
        src = _slice_embedding_dim(src, effective_embed_dim=effective_dim) + self.dropout1(out)
        src = self.norm1(src, effective_embed_dim=effective_dim)
        out = self.ffn(src, effective_heads=effective_heads)
        src = _slice_embedding_dim(src, effective_embed_dim=effective_dim) + self.dropout2(out)
        src = self.norm2(src, effective_embed_dim=effective_dim)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    effective_heads: Optional[int] = None):
        effective_dim = self.head_dim * effective_heads if effective_heads is not None else self.d_model
        out = self.norm1(src, effective_embed_dim=effective_dim)
        q = k = self.with_pos_embed(out, pos)    
        out, _ = self.self_attn(q, k, value=out, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, effective_heads=effective_heads)
        src = _slice_embedding_dim(src, effective_embed_dim=effective_dim) + self.dropout1(out)
        out = self.norm2(src, effective_embed_dim=effective_dim)
        out = self.ffn(out, effective_heads=effective_heads)
        src = _slice_embedding_dim(src, effective_embed_dim=effective_dim) + self.dropout2(out)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                effective_heads: Optional[int] = None):
        if effective_heads is not None:
            pos = _slice_embedding_dim(pos, effective_embed_dim=effective_heads * self.head_dim) if pos is not None else None
            src = _slice_embedding_dim(src, effective_embed_dim=effective_heads * self.head_dim) if src is not None else None
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, effective_heads=effective_heads)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, effective_heads=effective_heads)
    


class SlicedTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.head_dim = d_model // nhead
        self.self_attn = SlicedMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = SlicedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Head-local FFN: each head has independent FFN weights (block-diagonal)
        self.ffn = HeadLocalFFN(d_model, dim_feedforward, nhead, dropout=dropout, activation=activation)

        self.norm1 = SlicedGroupNorm(d_model, number_slice=nhead)
        self.norm2 = SlicedGroupNorm(d_model, number_slice=nhead)
        self.norm3 = SlicedGroupNorm(d_model, number_slice=nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     effective_heads: Optional[int] = None):
        effective_embed_dim = effective_heads * self.head_dim if effective_heads is not None else tgt.size(-1)
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, effective_heads=effective_heads)[0]
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout1(tgt2)
        tgt = self.norm1(tgt, effective_embed_dim=effective_embed_dim)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, effective_heads=effective_heads)[0]
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout2(tgt2)
        tgt = self.norm2(tgt, effective_embed_dim=effective_embed_dim)
        tgt2 = self.ffn(tgt, effective_heads=effective_heads)
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout3(tgt2)
        tgt = self.norm3(tgt, effective_embed_dim=effective_embed_dim)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    effective_heads: Optional[int] = None):
        effective_embed_dim = effective_heads * self.head_dim if effective_heads is not None else tgt.size(-1)
        tgt2 = self.norm1(tgt, effective_embed_dim=effective_embed_dim)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, effective_heads=effective_heads)[0]
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt, effective_embed_dim=effective_embed_dim)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, effective_heads=effective_heads)[0]
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt, effective_embed_dim=effective_embed_dim)
        tgt2 = self.ffn(tgt2, effective_heads=effective_heads)
        tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_embed_dim) + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                effective_heads: Optional[int] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, effective_heads)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, effective_heads)

    
class SlicedTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                effective_heads: Optional[int] = None):
        
        if effective_heads is None:
            output = src
        else:
            output = _slice_embedding_dim(src, effective_embed_dim=effective_heads * self.layers[0].head_dim)
        for layer in self.layers[:self.num_layers]:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, effective_heads=effective_heads)
        if self.norm is not None:
            effective_embed_dim = effective_heads * self.layers[0].head_dim if effective_heads is not None else None
            output = self.norm(output, effective_embed_dim=effective_embed_dim)

        return output
    

class SlicedTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, 
                effective_heads: Optional[int] = None):

        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, effective_heads=effective_heads)
            if self.return_intermediate:
                effective_embed_dim = effective_heads * self.layers[0].head_dim if effective_heads is not None else None
                intermediate.append(self.norm(output, effective_embed_dim=effective_embed_dim))

        if self.norm is not None:
            effective_embed_dim = effective_heads * self.layers[0].head_dim if effective_heads is not None else None
            output = self.norm(output, effective_embed_dim=effective_embed_dim)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)



class SlicedTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = SlicedTransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = SlicedGroupNorm(d_model, number_slice=nhead) if normalize_before else None
        self.encoder = SlicedTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = SlicedTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = SlicedGroupNorm(d_model, number_slice=nhead)
        self.decoder = SlicedTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, effective_heads=None):
        # flatten NxCxHxW to HWxNxC 
        # MultiHeadAttention expects [sequence_length, batch_size, embed_dim]
    
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        if effective_heads is not None:
            head_dim = self.d_model // self.nhead
            pos_embed = _slice_embedding_dim(pos_embed, effective_embed_dim=effective_heads * head_dim)
            query_embed = _slice_embedding_dim(query_embed, effective_embed_dim=effective_heads * head_dim)
            tgt = _slice_embedding_dim(tgt, effective_embed_dim=effective_heads * head_dim)
        
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, effective_heads=effective_heads)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed, effective_heads=effective_heads)
        
        if effective_heads is not None:
            effective_embed_dim = effective_heads * head_dim
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, effective_embed_dim, h, w)
        else:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

def build_transformer(args):
    return SlicedTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )