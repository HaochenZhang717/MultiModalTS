import numpy.random as random
from timm.models.vision_transformer import Mlp

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import *
from math import pi
from timm.models.vision_transformer import Attention
from functools import partial
from einops import rearrange, repeat
from collections.abc import Callable
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
               ), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    grid = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def _compute_default_rope_parameters(
        rope_theta,
        head_dim,
) -> tuple["torch.Tensor", float]:
    base = rope_theta
    partial_rotary_factor = 1.0
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    return inv_freq, attention_factor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, head_dim, rope_theta=10000, max_position_embeddings=4096):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = "default"
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        # self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(rope_theta, head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x):
        # 假设 x 的形状是 [Batch, Num_Heads, Seq_Len, Head_Dim]
        batch_size, _, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        cos = cos.unsqueeze(1).to(device=x.device, dtype=x.dtype)
        sin = sin.unsqueeze(1).to(device=x.device, dtype=x.dtype)
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed


class RelativePositionBias2D(nn.Module):
    """
    2D relative positional bias for full self-attention.
    Creates a learnable bias table of size (2*H-1) (2*W-1) per head,
    and a fixed index map to look up bias for any pair of token positions.
    """

    def __init__(self, height: int, width: int, num_heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_heads = num_heads

        # Create a bias table: one bias for every possible relative offset
        # in y ∈ [-(H-1)..(H-1)] and x ∈ [-(W-1)..(W-1)]
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * height - 1) * (2 * width - 1), num_heads)
        )
        # Precompute a (H*W)×(H*W) index matrix of which bias entry each pair (i,j) uses
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        # meshgrid of absolute coords, shape (H*W, 2)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1).view(-1, 2)

        # Compute all pairwise relative coords
        relative_coords = coords[:, None, :] - coords[None, :, :]  # shape (HW, HW, 2)
        # shift to positive
        relative_coords[..., 0] += height - 1  # y
        relative_coords[..., 1] += width - 1  # x

        # flatten 2D index into single index: idx = y*(2W-1) + x
        relative_index = relative_coords[..., 0] * (2 * width - 1) + relative_coords[..., 1]
        # register as buffer so it’s on the right device / dtype
        self.register_buffer("relative_index", relative_index.long())

    def forward(self):
        """
        Returns:
           bias: Tensor of shape (1, num_heads, HW, HW)
        to be added to the raw attention logits before softmax.
        """
        # Lookup and reshape to (HW, HW, num_heads)
        bias = self.relative_bias_table[self.relative_index.view(-1)]  # (HW*HW, num_heads)
        bias = bias.view(self.height * self.width,
                         self.height * self.width,
                         self.num_heads)  # (HW, HW, heads)
        # permute to (heads, HW, HW) and add batch-dim
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, HW, HW)
        return bias


class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = None,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)  # rope may change the q,k's dtype
            # todo: I was here.need to finish the code revision
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
            )
        else:
            raise NotImplementedError
            # q = q * self.scale
            # attn = q @ k.transpose(-2, -1)
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormCrossAttention(nn.Module):
    """
    Cross-attention module (query from x, key/value from context).
    """

    def __init__(
            self,
            dim: int,
            context_dim: int = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        if context_dim is None:
            context_dim = dim
        self.context_dim = context_dim

        if use_rmsnorm:
            norm_layer = RMSNorm

        # query from x
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # key/value from context
        self.kv_proj = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,  # (B, Nq, C)
            context: torch.Tensor,  # (B, Nk, context_dim)
            rope=None,
            attn_mask=None
    ) -> torch.Tensor:

        B, Nq, C = x.shape
        _, Nk, _ = context.shape

        # q
        q = self.q_proj(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        # kv
        kv = self.kv_proj(context).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        # rope (usually only apply to q,k if both represent positions in same space)
        if rope is not None:
            q = rope(q)

        # attention
        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)

            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else attn_mask)
        else:
            raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps.
    """
    embedding_size: int = 256
    scale: float = 1.0

    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W  # stop gradient manually
        t = t[:, None] * W[None, :] * 2 * torch.pi
        # Concatenate sine and cosine transformations
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


from typing import Optional, Union, Callable


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift,
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate,
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class PatchEmbed1D(nn.Module):
    """ 1D Sequence to Patch Embedding
    适用于音频、时间序列或一维信号处理
    """

    def __init__(
            self,
            sig_size: Optional[int] = None,  # 相当于原来的 img_size
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_sig_size: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.patch_size = patch_size  # 1D 只需要一个标量
        self.sig_size = sig_size

        # 计算 patch 数量
        if sig_size is not None:
            self.num_patches = sig_size // patch_size
        else:
            self.num_patches = None

        self.flatten = flatten
        self.strict_sig_size = strict_sig_size

        # 核心：使用 Conv1d 进行切片投影
        # kernel_size 和 stride 都等于 patch_size
        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **dd
        )

        self.norm = norm_layer(embed_dim, **dd) if norm_layer else nn.Identity()

    def forward(self, x):
        # x shape: [B, C, L] (Batch, Channels, Length)
        B, C, L = x.shape

        if self.sig_size is not None:
            if self.strict_sig_size:
                assert L == self.sig_size, f"输入长度 ({L}) 与模型设定 ({self.sig_size}) 不符"
            else:
                assert L % self.patch_size == 0, f"输入长度 ({L}) 必须能被 patch_size ({self.patch_size}) 整除"

        # 1. 卷积投影: [B, C, L] -> [B, Embed_Dim, L/Patch_Size]
        x = self.proj(x)

        # 2. 展平并转置: [B, E, L_patch] -> [B, L_patch, E]
        if self.flatten:
            # 在 1D 中，其实就是交换最后两个维度
            x = x.transpose(1, 2)

        x = self.norm(x)
        return x


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        dropout=0.0,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            proj_drop=dropout,
            **block_kwargs
        )

        # Initialize MLP layer

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None, attn_mask=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x),
                        shift_msa, scale_msa), rope=feat_rope, attn_mask=attn_mask), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x



class LightningDDTBlockDecoder(nn.Module):
    """
    Lightning DiT Block with cross attention. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            proj_drop=dropout,
            **block_kwargs
        )

        self.cross_attn = NormCrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            proj_drop=dropout,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, history_c, feat_rope=None, mask=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_cross, gate_cross, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
            shift_msa = None
            shift_cross = None
            shift_mlp = None

        else:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(9, dim=-1)


        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)

        x = x + DDTGate(self.cross_attn(DDTModulate(self.norm3(x), shift_cross, scale_cross), history_c, rope=feat_rope, mask=mask), gate_cross)

        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x



class DiTModel(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        in_channels: int = 1,
        hidden_size: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        vl_embed_dim: int = 2048,
        dropout: float = 0.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed: bool = True,
    ):
        super().__init__()

        patch_size = 1

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # patch embed
        channel_per_token = in_channels * patch_size
        self.embedder = PatchEmbed1D(
            sig_size=input_size,
            patch_size=patch_size,
            in_chans=channel_per_token,
            embed_dim=hidden_size,
            bias=True
        )

        # timestep embedding
        self.t_embedder = GaussianFourierEmbedding(hidden_size)

        # text embedding
        self.text_embedder = nn.Sequential(
            nn.Linear(vl_embed_dim, hidden_size)
        )

        # positional embedding
        if use_pos_embed:
            num_patches = self.embedder.num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size),
                requires_grad=False
            )
        self.use_pos_embed = use_pos_embed

        # rotary embedding
        if use_rope:
            half_head_dim = hidden_size // num_heads
            seq_len = self.embedder.num_patches
            self.feat_rope = LlamaRotaryEmbedding(
                head_dim=half_head_dim,
                max_position_embeddings=seq_len
            )
        else:
            self.feat_rope = None

        # transformer blocks
        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # final linear output
        self.final_layer = nn.Linear(hidden_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):

        # patch embed init
        w = self.embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.embedder.proj.bias, 0)

        # pos embed
        if self.use_pos_embed:
            pos_embed = get_1d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.embedder.num_patches
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )

        # adaLN init
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # final layer
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, ts, tp, t, text_embed, attn_mask):

        # timestep embedding
        t_embed = self.t_embedder(t)

        # text embedding
        text_embed = self.text_embedder(text_embed).mean(dim=1)

        # conditioning
        c_embed = torch.nn.functional.silu(t_embed + text_embed)

        # patch embedding
        x = self.embedder(ts)

        if self.use_pos_embed:
            x = x + self.pos_embed

        # transformer encoder
        for block in self.blocks:
            x = block(x, c_embed, feat_rope=self.feat_rope, attn_mask=attn_mask)

        # final projection
        x = self.final_layer(x)

        # reshape to (B, C, L)
        x = x.transpose(1, 2)

        return x