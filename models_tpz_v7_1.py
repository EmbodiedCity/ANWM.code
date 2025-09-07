# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import einops
from timm.models.vision_transformer import PatchEmbed, Mlp


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
# CaPE: 6DoF / 4DoF
# -----------------------------------------------------------------------------

class _CaPE6DoF(nn.Module):
    """
    6DoF CaPE：按 4 通道一组右乘 4x4 位姿矩阵。
    要求 head_dim % 4 == 0
    """
    @staticmethod
    def cape_embed(f, P):
        # f: (..., D), D%4==0; P: (...,4,4)
        f = einops.rearrange(f, '... (d k) -> ... d k', k=4)
        f = f @ P
        return einops.rearrange(f, '... d k -> ... (d k)', k=4)

    def xform_qk(self, q, k, Pq_invT, Pk):
        # q/k: (B,H,L,Dh); P*: (B,L,4,4)
        B, H, L, Dh = q.shape
        assert Dh % 4 == 0, "6DoF CaPE 需要 head_dim % 4 == 0"
        qf = einops.rearrange(q, 'b h l d -> (b h) l d')
        kf = einops.rearrange(k, 'b h l d -> (b h) l d')
        Pq_invT = einops.repeat(Pq_invT, 'b l m n -> (b h) l m n', h=H)
        Pk      = einops.repeat(Pk,      'b l m n -> (b h) l m n', h=H)
        qf = self.cape_embed(qf, Pq_invT)
        kf = self.cape_embed(kf, Pk)
        q  = einops.rearrange(qf, '(b h) l d -> b h l d', b=B, h=H)
        k  = einops.rearrange(kf, '(b h) l d -> b h l d', b=B, h=H)
        return q, k


class _CaPE4DoF(nn.Module):
    """
    4DoF 类 RoPE：两两旋转 + 相位调制
    需要 head_dim % (2*n_phase) == 0
    """
    @staticmethod
    def _rot2(x):
        x = einops.rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(-1)
        x = torch.stack((-x2, x1), dim=-1)
        return einops.rearrange(x, '... d j -> ... (d j)')

    @staticmethod
    def _phase_expand(x, p):  # x:(B,H,L,Dh), p:(B,L,n_phase)
        B, H, L, Dh = x.shape
        n = p.shape[-1]
        assert Dh % (2 * n) == 0, "4DoF CaPE 需要 head_dim % (2*n_phase) == 0"
        k = Dh // n
        return einops.repeat(p, 'b l n -> b h l (n k)', h=H, k=k)

    def xform_qk(self, q, k, pq, pk):
        phase_q = self._phase_expand(q, pq)
        phase_k = self._phase_expand(k, pk)
        q2 = (q * phase_q.cos()) + (self._rot2(q) * phase_q.sin())
        k2 = (k * phase_k.cos()) + (self._rot2(k) * phase_k.sin())
        return q2, k2


# -----------------------------------------------------------------------------
# Pose-aware Attention（统一投影）
# -----------------------------------------------------------------------------

class PoseAwareAttention(nn.Module):
    """
    自/交注意力统一使用 q_proj/k_proj/v_proj（**移除 qkv**）。
    CaPE 在 q/k 上执行等变变换，然后做 SDPA。
    """
    def __init__(self, dim, num_heads, cape_mode='6dof', qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # 统一的 q/k/v 投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj   = nn.Linear(dim, dim)

        self.cape_mode = cape_mode
        if cape_mode == '6dof':
            self.c6 = _CaPE6DoF()
            self.c4 = None
        elif cape_mode == '4dof':
            self.c6 = None
            self.c4 = _CaPE4DoF()
        else:
            self.c6 = None
            self.c4 = None

    def _apply_cape(self, q, k, pose_q, pose_k):
        if self.c6 is not None:
            assert pose_q is not None and pose_k is not None
            Pq_invT = torch.inverse(pose_q).permute(0, 1, 3, 2).contiguous()
            q, k = self.c6.xform_qk(q, k, Pq_invT, pose_k)
        elif self.c4 is not None:
            assert pose_q is not None and pose_k is not None
            q, k = self.c4.xform_qk(q, k, pose_q, pose_k)
        return q, k

    def _sdpa(self, q, k, v, attn_mask=None, key_padding_mask=None):
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B,H,Lq,Lk)
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(-1)
        out = attn @ v                                   # (B,H,Lq,Dh)
        out = einops.rearrange(out, 'b h l d -> b l (h d)')
        return self.proj(out)

    def forward_self(self, x, *, pose_q=None, pose_k=None,
                     attn_mask=None, key_padding_mask=None):
        """
        自注意：query=key=value=x（同一序列）。
        """
        q = einops.rearrange(self.q_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(x), 'b l (h d) -> b h l d', h=self.num_heads)

        if self.c6 is not None or self.c4 is not None:
            q, k = self._apply_cape(q, k, pose_q, pose_k)

        return self._sdpa(q, k, v, attn_mask, key_padding_mask)

    def forward_cross(self, query, key, value, *, pose_q=None, pose_k=None,
                      attn_mask=None, key_padding_mask=None):
        """
        交叉注意：query 与 key/value 为不同序列。
        """
        q = einops.rearrange(self.q_proj(query), 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(key),   'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(value), 'b l (h d) -> b h l d', h=self.num_heads)

        if self.c6 is not None or self.c4 is not None:
            q, k = self._apply_cape(q, k, pose_q, pose_k)

        return self._sdpa(q, k, v, attn_mask, key_padding_mask)


# -----------------------------------------------------------------------------
# CDiT Block
# -----------------------------------------------------------------------------

class CDiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning + CaPE。
    - 自注意：目标视图内部（使用目标位姿）
    - 交叉注意：Q=目标，K/V=上下文；若 `S==0`，用自回环保证始终执行（所有参数每步都参与）。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cape_mode='6dof', **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = PoseAwareAttention(hidden_size, num_heads, cape_mode=cape_mode, qkv_bias=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn  = PoseAwareAttention(hidden_size, num_heads, cape_mode=cape_mode, qkv_bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond, *, viewmats, Ks=None):
        """
        x:       (B, l, D)          目标视图 patch tokens
        x_cond:  (B, S*l, D) 或 (B, l, D)  当 S=0 时 (B,l,D)
        viewmats:(B, S+1, 4, 4)     最后一个是 target
        """
        B, Lt, D = x.shape
        S1 = viewmats.shape[1]  # = S+1
        S = S1 - 1
        Lk = x_cond.shape[1]

        # 目标位姿（自注意的 q/k）
        pose_qk_target = einops.repeat(viewmats[:, -1], 'b m n -> b l m n', l=Lt)   # (B,l,4,4)

        # ---------- adaLN 稳健拆分 ----------
        mods = self.adaLN_modulation(c)     # (B, 11*H) 或 (B,1,11*H)
        mods = mods.view(mods.shape[0], -1) # -> (B, 11*H)
        assert mods.shape[1] % 11 == 0, f"adaLN dims {mods.shape} 不是 11 的整数倍"
        h = mods.shape[1] // 11
        mods = mods.view(B, 11, h)
        (shift_msa, scale_msa, gate_msa,
         shift_ca_xcond, scale_ca_xcond,
         shift_ca_x, scale_ca_x, gate_ca_x,
         shift_mlp, scale_mlp, gate_mlp) = mods.unbind(dim=1)
        # -----------------------------------

        # 自注意（目标）
        x = x + gate_msa.unsqueeze(1) * self.attn.forward_self(
            modulate(self.norm1(x), shift_msa, scale_msa),
            pose_q=pose_qk_target, pose_k=pose_qk_target
        )

        # 交叉注意（始终执行）
        # 两种合法情形：
        # 1) S>0：Lk == S*Lt，pose_k 来自 viewmats[:, :-1]
        # 2) S==0：Lk == Lt（x_cond==x），pose_k := pose_qk_target
        if S > 0:
            assert Lk == S * Lt, f"x_cond 长度必须等于 S*l（S={S}, l={Lt}），got {Lk}"
            pose_k = einops.repeat(viewmats[:, :-1], 'b s m n -> b (s l) m n', l=Lt)  # (B,S*l,4,4)
        else:
            assert Lk == Lt, f"S==0 时 x_cond 应为 (B,l,D)，got Lk={Lk}, Lt={Lt}"
            pose_k = pose_qk_target

        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn.forward_cross(
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
            key=x_cond_norm, value=x_cond_norm,
            pose_q=pose_qk_target, pose_k=pose_k
        )

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
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
# CDiT Model
# -----------------------------------------------------------------------------

class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone (CaPE inside attention).
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
        cape_mode='6dof',     # '6dof' | '4dof' | None
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

        p_side = int(self.num_patches ** 0.5)
        pos_one = get_2d_sincos_pos_embed(hidden_size, p_side)  # (P, D)
        pos_stack = np.stack([pos_one for _ in range(self.context_size + 1)], axis=0)  # (S+1,P,D)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_stack).float(), requires_grad=False)

        self.blocks = nn.ModuleList([
            CDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, cape_mode=cape_mode,
                patches_x=self.num_patches_x, patches_y=self.num_patches_y, image_size=self.input_size
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

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

        # Final head：小 std，确保梯度回传
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
        Ks:       (B, S+1, 3,3) 或 None    （未用）
        """
        B = x.shape[0]
        S = self.context_size
        assert viewmats.shape[1] == S + 1, "viewmats 维度应为 (B, S+1, 4,4)"

        # Target
        x = self.x_embedder(x) + self.pos_embed[S:S+1]  # (B,P,D)

        # Contexts
        if S > 0:
            x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (B, S))  # (B,S,P,D)
            x_cond = x_cond + self.pos_embed[:S]                                  # (B,S,P,D)
            x_cond = x_cond.flatten(1, 2)                                         # (B,S*P,D)
        else:
            # 自回环：保证交叉注意力必执行、必用参数
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
