# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# PRoPE: "Cameras as Relative Positional Encoding"
# --------------------------------------------------------

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.vision_transformer import PatchEmbed, Mlp

# ====== PRoPE attention core (change the import path if needed) ======
from prope.torch import PropeDotProductAttention


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------

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
        t: (B,) or (B,1)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if t.ndim > 1:
            t = t.squeeze(-1)
            if t.ndim > 1:
                t = t.reshape(t.shape[0])
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ActionEmbedder(nn.Module):
    """
    Embeds action (x,y,z,angle) into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size // 4
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.z_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size - 3 * hsize, frequency_embedding_size)

    def forward(self, xyza):
        return torch.cat([
            self.x_emb(xyza[..., 0:1]),
            self.y_emb(xyza[..., 1:2]),
            self.z_emb(xyza[..., 2:3]),
            self.angle_emb(xyza[..., 3:4]),
        ], dim=-1)


# -----------------------------------------------------------------------------
# PRoPE Multi-Head Attentions (batch_first)
# -----------------------------------------------------------------------------

class PropeMHSelfAttention(nn.Module):
    """
    PRoPE 多头自注意力（batch_first: x=[B,L,D]）
    - Q,K,V 同序列；直接调用 PropeDotProductAttention.forward（最快）
    - 要求 head_dim % 4 == 0
    """
    def __init__(self, dim, num_heads, patches_x, patches_y, image_size, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        assert head_dim % 4 == 0, "PRoPE requires head_dim % 4 == 0"
        self.num_heads = num_heads
        self.head_dim  = head_dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out    = nn.Linear(dim, dim, bias=True)

        self.prope = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_size,
            image_height=image_size,
        )

    def forward(self, x, *, viewmats_q, Ks_q: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        x: [B, L, D] (L=Hy*Wx)
        viewmats_q: [B, 1, 4, 4]
        Ks_q: [B, 1, 3, 3] or None
        """
        B, L, D = x.shape
        q = einops.rearrange(self.q_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)

        # 自注意力：q/k/v 序列长度一致，这里 cameras=1（L=1*P）
        out = self.prope(
            q, k, v,
            viewmats=viewmats_q,
            Ks=Ks_q,
            attn_mask=attn_mask,
        )  # [B,H,L,Dh]

        out = einops.rearrange(out, 'b h l d -> b l (h d)')
        return self.out(out)


class PropeMHCrossAttention(nn.Module):
    """
    PRoPE 多头交叉注意力（batch_first）
    - Q=target（cameras=1），KV=contexts（cameras=S）
    - 采用“预计算 + 手动 SDPA + 输出再映射”的官方范式
    """
    def __init__(self, dim, num_heads, patches_x, patches_y, image_size, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        assert head_dim % 4 == 0, "PRoPE requires head_dim % 4 == 0"
        self.num_heads = num_heads
        self.head_dim  = head_dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out    = nn.Linear(dim, dim, bias=True)

        self.prope_q  = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x, patches_y=patches_y,
            image_width=image_size, image_height=image_size,
        )
        self.prope_kv = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x, patches_y=patches_y,
            image_width=image_size, image_height=image_size,
        )

    def forward(self, query, key, value, *,
                viewmats_q, viewmats_kv,
                Ks_q: Optional[torch.Tensor] = None, Ks_kv: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0):
        """
        query:   [B, Lq, D]  （target，Lq=P）
        key:     [B, Lk, D]  （contexts 展平，Lk=S*P）
        value:   [B, Lk, D]
        viewmats_q:  [B,1,4,4]
        viewmats_kv: [B,S,4,4]
        """
        B, Lq, D = query.shape
        Lk = key.shape[1]
        q = einops.rearrange(self.q_proj(query), 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(key),   'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(value), 'b l (h d) -> b h l d', h=self.num_heads)

        # 预计算几何映射
        self.prope_q._precompute_and_cache_apply_fns(viewmats_q,  Ks_q)
        self.prope_kv._precompute_and_cache_apply_fns(viewmats_kv, Ks_kv)

        # 应用到 Q 与 KV
        q = self.prope_q._apply_to_q(q)     # [B,H,Lq,Dh]
        k = self.prope_kv._apply_to_kv(k)   # [B,H,Lk,Dh]
        v = self.prope_kv._apply_to_kv(v)   # [B,H,Lk,Dh]

        # 标准 SDPA（允许 Lq != Lk）
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )  # [B,H,Lq,Dh]

        # 把输出投回 Q 侧相机参考系
        out = self.prope_q._apply_to_o(out)  # [B,H,Lq,Dh]

        out = einops.rearrange(out, 'b h l d -> b l (h d)')
        return self.out(out)


# -----------------------------------------------------------------------------
# CDiT Block
# -----------------------------------------------------------------------------

class CDiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning + PRoPE 注意力。
    - 自注意：目标视图内部（使用目标位姿）
    - 交叉注意：Q=目标，K/V=上下文；若 S==0，回环为 self 以保证层参与训练
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = PropeMHSelfAttention(
            hidden_size, num_heads,
            patches_x=block_kwargs.get("patches_x"),
            patches_y=block_kwargs.get("patches_y"),
            image_size=block_kwargs.get("image_size"),
            qkv_bias=True,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn  = PropeMHCrossAttention(
            hidden_size, num_heads,
            patches_x=block_kwargs.get("patches_x"),
            patches_y=block_kwargs.get("patches_y"),
            image_size=block_kwargs.get("image_size"),
            qkv_bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond, *, viewmats, Ks=None):
        B, Lt, D = x.shape
        S1 = viewmats.shape[1]
        S  = S1 - 1
        Lk = x_cond.shape[1]

        # ---------- adaLN ----------
        mods = self.adaLN_modulation(c).view(B, 11, -1)
        (shift_msa, scale_msa, gate_msa,
        shift_ca_xcond, scale_ca_xcond,   # 这俩现在不再作用在 K/V 上，先保留参数位
        shift_ca_x, scale_ca_x, gate_ca_x,
        shift_mlp, scale_mlp, gate_mlp) = mods.unbind(dim=1)

        # ---------- 自注意（QKV=target；cameras=1） ----------
        msa_in  = self.norm1(x)  # 只 LN
        msa_out = self.attn(
            msa_in,
            viewmats_q=viewmats[:, -1:],                      # target 相机
            Ks_q=None if Ks is None else Ks[:, -1:],
        )
        x = x + gate_msa.unsqueeze(1) * modulate(msa_out, shift_msa, scale_msa)

        # ---------- 交叉注意（Q=target；KV=contexts 或回环） ----------
        # 先根据是否有上下文视图，确定 KV 使用的相机参数（以及形状检查）
        if S > 0:
            # 有 S 个上下文相机：x_cond 应该是 S*Lt 个 token
            assert Lk == S * Lt, f"x_cond 应为 S*P（S={S}, P={Lt}），got Lk={Lk}"
            viewmats_kv = viewmats[:, :-1]                         # (B,S,4,4)
            Ks_kv       = None if Ks is None else Ks[:, :-1]       # (B,S,3,3) or None
        else:
            # 无上下文：回环到 target 自身，让 cross-attn 仍然生效
            assert Lk == Lt, "S==0 时 x_cond 应为 (B,P,D)"
            viewmats_kv = viewmats[:, -1:]                         # (B,1,4,4)
            Ks_kv       = None if Ks is None else Ks[:, -1:]       # (B,1,3,3) or None

        q_in  = self.norm2(x)          # 只 LN
        kv_in = self.norm_cond(x_cond) # 只 LN

        ca_out = self.cttn(
            query=q_in, key=kv_in, value=kv_in,
            viewmats_q=viewmats[:, -1:], Ks_q=None if Ks is None else Ks[:, -1:],
            viewmats_kv=viewmats_kv,     Ks_kv=Ks_kv,
        )
        x = x + gate_ca_x.unsqueeze(1) * modulate(ca_out, shift_ca_x, scale_ca_x)

        # ---------- MLP ----------
        mlp_in  = self.norm3(x)
        mlp_out = self.mlp(mlp_in)
        x = x + gate_mlp.unsqueeze(1) * modulate(mlp_out, shift_mlp, scale_mlp)
        return x


# -----------------------------------------------------------------------------
# Final Layer
# -----------------------------------------------------------------------------

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# -----------------------------------------------------------------------------
# Sine/Cosine Positional Embeddings
# -----------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# -----------------------------------------------------------------------------
# CDiT Model (PRoPE inside attention)
# -----------------------------------------------------------------------------

class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone (PRoPE inside attention).
    viewmats: (B, S+1, 4,4)，最后一个 target
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
        gate_eps: float = 1e-3,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gate_eps = gate_eps

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        self.num_patches = self.x_embedder.num_patches
        self.num_patches_y, self.num_patches_x = self.x_embedder.grid_size
        self.input_size = input_size

        # Positional embedding per camera slot (S contexts + 1 target)
        p_side = int(self.num_patches ** 0.5)
        pos_one = get_2d_sincos_pos_embed(hidden_size, p_side)  # (P, D)
        pos_stack = np.stack([pos_one for _ in range(self.context_size + 1)], axis=0)  # (S+1,P,D)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_stack).float(), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                patches_x=self.num_patches_x, patches_y=self.num_patches_y, image_size=self.input_size
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # PRoPE 形状断言（运行期检查 head_dim%4）
        head_dim = hidden_size // num_heads
        assert head_dim % 4 == 0, "PRoPE 要求 head_dim%4==0；请调整 hidden_size/num_heads"

    def initialize_weights(self):
        # Linear init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # PatchEmbed like Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Embedders
        for emb in [self.y_embedder.x_emb, self.y_embedder.y_emb, self.y_embedder.z_emb, self.y_embedder.angle_emb,
                    self.t_embedder, self.time_embedder]:
            nn.init.normal_(emb.mlp[0].weight, std=0.02)
            nn.init.normal_(emb.mlp[2].weight, std=0.02)

        # adaLN-Zero + gate ε
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            with torch.no_grad():
                lin = block.adaLN_modulation[-1]  # Linear(H, 11H)
                H = lin.bias.shape[0] // 11
                for idx in (2, 7, 10):  # gates: msa/ca/mlp
                    lin.bias[idx * H:(idx + 1) * H].fill_(self.gate_eps)

        # Final head
        nn.init.normal_(self.final_layer.linear.weight, std=1e-5)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * C)
        imgs: (B, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "L 必须是正方形网格"
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t, x_sup, viewmats, Ks=None):
        """
        x:        (B, C, H, W)            目标视图输入
        x_cond:   (B, S, C, H, W)         上下文视图
        t:        (B,)                     diffusion step
        y:        (B, 4)                   行为/控制信号
        rel_t:    (B,)                     相对时间
        x_sup:    (B, C, H, W)             监督分支（未用，但保持接口）
        viewmats: (B, S+1, 4,4)            最后一个为 target
        Ks:       (B, S+1, 3,3) 或 None
        """
        B = x.shape[0]
        S = self.context_size
        assert viewmats.shape[1] == S + 1, "viewmats 维度应为 (B, S+1, 4,4)"

        # Target tokens
        x = self.x_embedder(x) + self.pos_embed[S:S+1]  # (B,P,D)

        # Context tokens
        if S > 0:
            x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (B, S))  # (B,S,P,D)
            x_cond = x_cond + self.pos_embed[:S]                                  # (B,S,P,D)
            x_cond = x_cond.flatten(1, 2)                                         # (B,S*P,D)
        else:
            # 回环：保证 cross-attn 始终执行
            x_cond = x.clone()                                                    # (B,P,D)

        # Condition vector
        t_emb     = self.t_embedder(t)
        time_emb  = self.time_embedder(rel_t)
        y_emb     = self.y_embedder(y)
        c = t_emb + time_emb + y_emb                                              # (B,H)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, x_cond, viewmats=viewmats, Ks=Ks)

        # Output
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

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
