from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from utils import positions

import torch.nn as nn


class Sequential(nn.Sequential):
    def forward(self, *x, **kwargs):
        for i, module in enumerate(self):
            if type(x) == tuple:
                if i == 0:
                    x = module(*x, **kwargs)
                else:
                    x = module(*x)
            else:
                x = module(x)
        return x

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.Mish(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q,
        x_kv,
        pad_mask=None
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)
        attn_max_neg = -torch.finfo(attn.dtype).max

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)
    
class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x, pad_mask=None):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask)
    
class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(self_attn),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias)),
        )

class ModelHead(nn.Module):
    def __init__(
            self,
            num_layers:int, 
            num_heads: int,
            in_channels:int,
            num_channels: int,
            out_channels:int, 
            widening_factor: int = 1,
            dropout: float = 0.0,
        ):
        super().__init__()

        self.linear = nn.Linear(in_channels, num_channels)

        self.attn_layers = nn.ModuleList([
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                widening_factor=widening_factor,
                dropout=dropout
            )
            for _ in range(num_layers)]
            )
        
        self.decoder = nn.ModuleList([
            nn.Linear(num_channels, num_channels // 2),
            nn.Linear(num_channels // 2, num_channels // 4), 
            nn.Linear(num_channels // 4, out_channels)]
        )

        
    def forward(self, x, pad_mask) :
        x = self.linear(x)
        for layer in self.attn_layers :
            x = layer(x, pad_mask)

        x = x.flatten(0, 1)
        for layer in self.decoder :
            x = layer(x)

        return x
