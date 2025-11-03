# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# PE-Field: Positional Encoding Field for 3D-aware positional encoding
# --------------------------------------------------------

# CDiT v9: Using PE-Field (Positional Encoding Field) instead of PRoPE
# PE-Field provides depth-aware encodings and hierarchical encodings for better geometry modeling

import math
import numpy as np
from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


#################################################################################
#                    PE-Field Positional Encoding Functions                    #
#################################################################################

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int, torch.Tensor],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray`, `int`, or `torch.Tensor`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0): Scaling factor for frequency computation.
        use_real (`bool`, *optional*): If True, return real part and imaginary part separately.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`): If `True` and `use_real`, real part and imaginary part are each interleaved.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`): the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2] or ([S, D], [S, D])
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    if isinstance(pos, torch.Tensor) and pos.device.type in ["mps", "npu"]:
        pos = pos.to(freqs_dtype)

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos.float(), freqs)  # [S, D/2]
    
    is_npu = isinstance(pos, torch.Tensor) and pos.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
        
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        x (`torch.Tensor`): Query or key tensor to apply rotary embeddings. [B, H, S, D] or [B, S, H, D]
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D])
        use_real (`bool`, *optional*, defaults to `True`): If True, use real representation.
        use_real_unbind_dim (`int`, *optional*, defaults to `-1`): Dimension for unbinding real/imaginary parts.
    
    Returns:
        `torch.Tensor`: Modified tensor with rotary embeddings applied.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]  # [1, 1, S, D]
        sin = sin[None, None]  # [1, 1, S, D]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class FluxPosEmbed(nn.Module):
    """
    Positional embedding for 3D coordinates (z, u, v) using RoPE.
    Modified from PE-Field/diffusers.
    """
    def __init__(self, theta: int = 10000, axes_dim: List[int] = None):
        super().__init__()
        self.theta = theta
        if axes_dim is None:
            axes_dim = [64, 64, 64]  # default: 64 dims for each axis (z, u, v)
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ids: [N, 3] tensor of 3D coordinates (z, u, v)
        Returns:
            (freqs_cos, freqs_sin): Each is [N, D] where D = sum(axes_dim)
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def encode_groups(ids: torch.Tensor, pos_embed_fn):
    """
    Encode multiple groups of 3D coordinates.
    
    Args:
        ids: shape (N, num_groups * 3) - multiple groups of (z, u, v)
        pos_embed_fn: function that takes (N, 3) and returns (freqs_cos, freqs_sin)
    Returns:
        (final_cos, final_sin): Each is (N, num_groups * dim)
    """
    N = ids.shape[0]
    groups = torch.split(ids, 3, dim=-1)  # split into multiple (N, 3) groups

    cos_list = []
    sin_list = []

    for group in groups:
        cos, sin = pos_embed_fn(group)
        cos_list.append(cos)
        sin_list.append(sin)

    # Concatenate as (N, num_groups * dim)
    final_cos = torch.cat(cos_list, dim=-1)
    final_sin = torch.cat(sin_list, dim=-1)
    return final_cos, final_sin


def compose_24d(t1: torch.Tensor, t4: torch.Tensor, t16: torch.Tensor) -> torch.Tensor:
    """
    Compose a (N, 24d) tensor from:
      t1: (N, d)
      t4: (N, 4d)
      t16: (N, 16d)
    by creating 4 groups, each group is (N, 6d):
      - first d from t1 (repeated)
      - second d from t4, one of its 4 parts
      - next 4d from t16, corresponding chunk
    """
    N, d = t1.shape
    out_chunks = []

    for i in range(4):
        # From t1: repeated, same for all groups
        chunk_t1 = t1

        # From t4: slice out i-th d
        chunk_t4 = t4[:, i*d:(i+1)*d]

        # From t16: slice out 4 consecutive d
        chunk_t16 = t16[:, i*4*d:(i+1)*4*d]
        # Concatenate these to form (N, 6d)
        group = torch.cat([chunk_t1, chunk_t4, chunk_t16], dim=-1)
        out_chunks.append(group)

    # Finally concatenate all groups along last dimension: (N, 24d)
    out = torch.cat(out_chunks, dim=-1)
    return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size//4
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.z_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size -3*hsize, frequency_embedding_size)

    def forward(self, xyza):
        return torch.cat([self.x_emb(xyza[...,0:1]), self.y_emb(xyza[...,1:2]), self.z_emb(xyza[...,2:3]), self.angle_emb(xyza[...,3:4])], dim=-1)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Modified to support PE-Field positional encoding via RoPE.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # For PE-Field: need to extract q, k separately to apply RoPE
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads

    def forward(self, x, c, x_cond, x_encoded, x_cond_encoded, image_rotary_emb=None):
        """
        Args:
            x: [B, N_patches, D] current frame tokens
            c: [B, D] conditioning
            x_cond: [B, T_ctx*N_patches, D] context tokens
            x_encoded: [B, N_patches, D] encoded current tokens (same as x if no PE-Field)
            x_cond_encoded: [B, T_ctx*N_patches, D] encoded context tokens (same as x_cond if no PE-Field)
            image_rotary_emb: Optional Tuple of (cos, sin) for RoPE encoding, each [N_patches, D_rope]
        """
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
        
        # Self-attention with optional PE-Field RoPE
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)  # [B, N_patches, D]
        
        if image_rotary_emb is not None:
            # Apply RoPE to query and key manually
            # Extract q, k, v from attention module
            B, N, D = x_norm.shape
            qkv = self.attn.qkv(x_norm)  # [B, N, 3*D]
            q, k, v = qkv.chunk(3, dim=-1)  # Each [B, N, D]
            
            # Reshape to [B, N, H, head_dim] then [B, H, N, head_dim]
            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
            
            # Apply RoPE
            cos, sin = image_rotary_emb  # Each [N, D_rope]
            # Ensure dimensions match
            if cos.shape[0] != N:
                # Broadcast if needed (should match N_patches)
                cos = cos[:N] if cos.shape[0] >= N else F.pad(cos, (0, 0, 0, N - cos.shape[0]))
                sin = sin[:N] if sin.shape[0] >= N else F.pad(sin, (0, 0, 0, N - sin.shape[0]))
            
            # Apply RoPE: need to handle dimension mismatch
            # RoPE typically works on head_dim, so we need to slice cos/sin if D_rope != head_dim
            rope_dim = cos.shape[-1]
            if rope_dim == self.head_dim:
                q = apply_rotary_emb(q, (cos, sin))
                k = apply_rotary_emb(k, (cos, sin))
            else:
                # If dimensions don't match, we need to interpolate or pad
                # For now, use only the first head_dim dimensions
                cos_subset = cos[:, :self.head_dim] if rope_dim >= self.head_dim else F.pad(cos, (0, self.head_dim - rope_dim))
                sin_subset = sin[:, :self.head_dim] if rope_dim >= self.head_dim else F.pad(sin, (0, self.head_dim - rope_dim))
                q = apply_rotary_emb(q, (cos_subset, sin_subset))
                k = apply_rotary_emb(k, (cos_subset, sin_subset))
            
            # Reshape back and use scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
            attn_output = self.attn.proj(attn_output)  # [B, N, D]
            attn_output = self.attn.proj_drop(attn_output)
            x = x + gate_msa.unsqueeze(1) * attn_output
        else:
            # Standard attention without RoPE
            x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)
        
        # Cross-attention
        x_cond_norm = modulate(self.norm_cond(x_cond_encoded), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x_encoded), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT (x_sup disabled).
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_supervised_token=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone using PE-Field for positional encoding.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        latent_size=20,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)

        self.num_patches = self.x_embedder.num_patches
        # 注意：grid_size = (H_patches, W_patches)
        self.num_patches_y, self.num_patches_x = self.x_embedder.grid_size
        self.input_size = input_size
        
        # PE-Field positional embedding
        # For 3D coordinates (z, u, v), we use RoPE with dimensions matching head_dim
        head_dim = hidden_size // num_heads
        # Each axis gets head_dim // 3 dimensions, so total output is head_dim (matching RoPE requirement)
        # This ensures the RoPE encoding dimension matches the head dimension for direct application
        axes_dims = [head_dim // 3, head_dim // 3, head_dim - 2 * (head_dim // 3)]
        # Ensure each axis_dim is even (required by get_1d_rotary_pos_embed)
        axes_dims = [(d // 2) * 2 for d in axes_dims]  # Make each even
        # Adjust the last one to ensure sum is still approximately head_dim
        total_so_far = sum(axes_dims[:2])
        axes_dims[2] = (head_dim - total_so_far) // 2 * 2  # Make even
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims)
        
        self.blocks = nn.ModuleList([CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize action embedding (x, y, z, angle)
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.z_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.z_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLPs
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t, x_sup, pix_coords_downs=None, viewmats=None, Ks=None):
        """
        Forward pass of DiT with PE-Field positional encoding.
        
        Args:
            x:      (B*num_goals, C, H, W) - current frame
            x_cond: (B*num_goals, num_cond+1, C, H, W) - context frames
            t:      (B*num_goals,) - timestep
            y:      (B*num_goals, 4) - action (x,y,z,angle)
            rel_t:  (B*num_goals,) - relative timestep
            x_sup:  (unused, kept for compatibility)
            pix_coords_downs: List[torch.Tensor] - PE-Field multi-scale 3D coordinates (z, u, v)
                             Each tensor shape: [N_patches, 3] or [N_patches*num_groups, 3]
            viewmats/Ks: (deprecated, kept for compatibility) - not used with PE-Field
        """
        # Embed patches
        x = self.x_embedder(x)  # [B,  Np, D]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1]))
        # x_cond: [B, T_ctx, Np, D] → [B, T_ctx*Np, D]
        _, TT, _ = x.shape  # TT = Np (per image)
        x_cond = x_cond.flatten(1, 2)  # [B, (num_cond+1)*Np, D]

        t = self.t_embedder(t[..., None])
        y = self.y_embedder(y)
        time_emb = self.time_embedder(rel_t[..., None])
        c = t + time_emb + y  # condition token

        # Compute PE-Field positional encoding if provided
        image_rotary_emb = None
        if pix_coords_downs is not None:
            # pix_coords_downs is a list of tensors for different scales
            # For now, we use the first/last scale or compose them
            # Following PE-Field paper, we typically use multi-scale encoding
            if isinstance(pix_coords_downs, list) and len(pix_coords_downs) > 0:
                # Use the scale that matches current patches (usually the last one)
                # Each element should be [N_patches, 3] for (z, u, v)
                pix_coords = pix_coords_downs[-1]  # Use the finest scale
                
                # Ensure it matches the number of patches for current frame
                if pix_coords.shape[0] == TT:
                    # Single scale encoding
                    image_rotary_emb = self.pos_embed(pix_coords)  # (cos, sin), each [TT, D_rope]
                elif len(pix_coords_downs) >= 3:
                    # Multi-scale encoding (like PE-Field paper)
                    # We expect at least 3 scales: [scale_1, scale_4, scale_16]
                    cos_1, sin_1 = self.pos_embed(pix_coords_downs[-3])  # 1x1 scale
                    cos_4, sin_4 = encode_groups(pix_coords_downs[-2], self.pos_embed)  # 2x2 groups
                    cos_16, sin_16 = encode_groups(pix_coords_downs[-1], self.pos_embed)  # 4x4 groups
                    
                    # Compose multi-scale encoding
                    composed_cos = compose_24d(cos_1, cos_4, cos_16)
                    composed_sin = compose_24d(sin_1, sin_4, sin_16)
                    image_rotary_emb = (composed_cos, composed_sin)
                else:
                    # Fallback: use first available scale and pad/trim if needed
                    pix_coords = pix_coords_downs[0]
                    if pix_coords.shape[0] >= TT:
                        pix_coords = pix_coords[:TT]
                    else:
                        # Pad with zeros (last coordinate repeated)
                        padding = torch.zeros(TT - pix_coords.shape[0], 3, device=pix_coords.device, dtype=pix_coords.dtype)
                        pix_coords = torch.cat([pix_coords, padding], dim=0)
                    image_rotary_emb = self.pos_embed(pix_coords)
            elif isinstance(pix_coords_downs, torch.Tensor):
                # Single tensor instead of list
                pix_coords = pix_coords_downs
                if pix_coords.shape[0] >= TT:
                    pix_coords = pix_coords[:TT]
                else:
                    padding = torch.zeros(TT - pix_coords.shape[0], 3, device=pix_coords.device, dtype=pix_coords.dtype)
                    pix_coords = torch.cat([pix_coords, padding], dim=0)
                image_rotary_emb = self.pos_embed(pix_coords)

        # For now, x_encoded and x_cond_encoded are the same as x and x_cond
        # In the future, we could add separate encoding if needed
        x_encoded = x
        x_cond_encoded = x_cond

        # Pass through transformer blocks with PE-Field encoding
        for block in self.blocks:
            x = block(x, c, x_cond, x_encoded, x_cond_encoded, image_rotary_emb=image_rotary_emb)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}

