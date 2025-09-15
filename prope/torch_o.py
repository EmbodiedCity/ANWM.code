# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        cameras: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.cameras = cameras
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y * cameras,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(torch.arange(patches_y), patches_x),
                (cameras,),
            ),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            **kwargs,
        )


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """

    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(
                torch.arange(patches_x, device=q.device), (patches_y * cameras,)
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=q.device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]
    out = F.scaled_dot_product_attention(
        query=_apply_block_diagonal(q, transforms_q),
        key=_apply_block_diagonal(k, transforms_kv),
        value=_apply_block_diagonal(v, transforms_kv),
        **kwargs,
    )
    out = _apply_block_diagonal(out, transforms_o)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


# ==================== [ADDED] 支持 Cq ≠ Ckv 的 Cross-Attn ====================
def prope_cross_dot_product_attention(  # [ADDED]
    q: torch.Tensor,  # (B, H, Tq, Hd)
    k: torch.Tensor,  # (B, H, Tk, Hd)
    v: torch.Tensor,  # (B, H, Tk, Hd)
    *,
    viewmats_q: torch.Tensor,    # (B, Cq, 4, 4)
    viewmats_kv: torch.Tensor,   # (B, Ckv, 4, 4)
    Ks_q: Optional[torch.Tensor],    # (B, Cq, 3, 3)
    Ks_kv: Optional[torch.Tensor],   # (B, Ckv, 3, 3)
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    **kwargs,
) -> torch.Tensor:
    """
    PRoPE cross-attention that supports different #cameras for Q and KV.
    Token order must be camera-major: T? = C? * patches_x * patches_y.
    """
    B, H, Tq, Hd = q.shape
    _, _, Tk, _ = k.shape
    Cq = viewmats_q.shape[1]
    Ck = viewmats_kv.shape[1]
    assert Tq == Cq * patches_x * patches_y, "Q length must be Cq*px*py"
    assert Tk == Ck * patches_x * patches_y, "KV length must be Ckv*px*py"
    assert viewmats_q.shape == (B, Cq, 4, 4)
    assert viewmats_kv.shape == (B, Ck, 4, 4)
    assert (Ks_q is None) or Ks_q.shape == (B, Cq, 3, 3)
    assert (Ks_kv is None) or Ks_kv.shape == (B, Ck, 3, 3)

    # --- Q side: build Pq, Pq^T ---
    if Ks_q is not None:
        Ks_qn = torch.zeros_like(Ks_q)
        Ks_qn[..., 0, 0] = Ks_q[..., 0, 0] / image_width
        Ks_qn[..., 1, 1] = Ks_q[..., 1, 1] / image_height
        Ks_qn[..., 0, 2] = Ks_q[..., 0, 2] / image_width - 0.5
        Ks_qn[..., 1, 2] = Ks_q[..., 1, 2] / image_height - 0.5
        Ks_qn[..., 2, 2] = 1.0
        Pq = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_qn), viewmats_q)  # image<-world
        Pq_T = Pq.transpose(-1, -2)
    else:
        Pq = viewmats_q
        Pq_T = Pq.transpose(-1, -2)

    # --- KV side: build Pkv_inv ---
    if Ks_kv is not None:
        Ks_kvn = torch.zeros_like(Ks_kv)
        Ks_kvn[..., 0, 0] = Ks_kv[..., 0, 0] / image_width
        Ks_kvn[..., 1, 1] = Ks_kv[..., 1, 1] / image_height
        Ks_kvn[..., 0, 2] = Ks_kv[..., 0, 2] / image_width - 0.5
        Ks_kvn[..., 1, 2] = Ks_kv[..., 1, 2] / image_height - 0.5
        Ks_kvn[..., 2, 2] = 1.0
        Pkv_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats_kv),
            _lift_K(_invert_K(Ks_kvn)),
        )
    else:
        Pkv_inv = _invert_SE3(viewmats_kv)

    # --- RoPE coeffs for Q and KV (lengths differ) ---
    assert Hd % 4 == 0
    def rope_coeffs(cams: int):
        cx = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=q.device), (patches_y * cams,)),
            freq_base=100.0, freq_scale=1.0, feat_dim=Hd // 4,
        )
        cy = _rope_precompute_coeffs(
            torch.tile(torch.repeat_interleave(torch.arange(patches_y, device=q.device), patches_x), (cams,)),
            freq_base=100.0, freq_scale=1.0, feat_dim=Hd // 4,
        )
        return cx, cy

    (cx_q, cy_q) = rope_coeffs(Cq)
    (cx_kv, cy_kv) = rope_coeffs(Ck)

    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=Pq_T), Hd // 2),
        (partial(_rope_apply_coeffs, coeffs=cx_q),   Hd // 4),
        (partial(_rope_apply_coeffs, coeffs=cy_q),   Hd // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=Pkv_inv), Hd // 2),
        (partial(_rope_apply_coeffs, coeffs=cx_kv),     Hd // 4),
        (partial(_rope_apply_coeffs, coeffs=cy_kv),     Hd // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=Pq), Hd // 2),
        (partial(_rope_apply_coeffs, coeffs=cx_q, inverse=True), Hd // 4),
        (partial(_rope_apply_coeffs, coeffs=cy_q, inverse=True), Hd // 4),
    ]

    q_ = _apply_block_diagonal(q, transforms_q)      # (B,H,Tq,Hd)
    k_ = _apply_block_diagonal(k, transforms_kv)     # (B,H,Tk,Hd)
    v_ = _apply_block_diagonal(v, transforms_kv)     # (B,H,Tk,Hd)

    out = F.scaled_dot_product_attention(q_, k_, v_, **kwargs)  # supports Tq != Tk
    out = _apply_block_diagonal(out, transforms_o)
    return out  # (B,H,Tq,Hd)
# ==================== [ADDED END] ============================================


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out

# ===========================================================
# ======== High-level wrappers for (B,T,D) inputs ===========
# ===========================================================

# [ADDED] Batch-first 自注意力
class PropeSelfAttention(nn.Module):
    """
    Batch-first PRoPE self-attention wrapper for (B,T,D) tensors.
    Reuses original prope_dot_product_attention path.
    """
    def __init__(self, embed_dim, num_heads, *, patches_x, patches_y, image_size,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 freq_base=100.0, freq_scale=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.px = patches_x
        self.py = patches_y
        self.im = image_size

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 预创建一个核心对象（持有 buffer 尺寸），但具体系数在函数内部根据 seqlen 生成/广播
        self.core = PropeDotProductAttention(
            head_dim=self.head_dim,
            cameras=1,  # 仅作占位；真正使用由 seqlen//(px*py) 推相机数
            patches_x=self.px,
            patches_y=self.py,
            image_width=self.im,
            image_height=self.im,
            freq_base=freq_base,
            freq_scale=freq_scale,
        )

    def forward(self, x, *, viewmats_q, Ks_q=None, **sdpa_kwargs):
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, Hd).transpose(1, 2)
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        out = prope_dot_product_attention(
            q, k, v,
            viewmats=viewmats_q,
            Ks=Ks_q,
            patches_x=self.px,
            patches_y=self.py,
            image_width=self.im,
            image_height=self.im,
            **sdpa_kwargs,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.proj_drop(self.proj(out))
        return out


# [MOD] Batch-first 交叉注意力：支持 Cq ≠ Ckv
class PropeCrossAttention(nn.Module):
    """
    Batch-first PRoPE cross-attention wrapper.
    query: (B, Tq, D) from target cameras (viewmats_q)
    key/value: (B, Tk, D) from context cameras (viewmats_kv)
    """
    def __init__(self, embed_dim, num_heads, *, patches_x, patches_y, image_size,
                 bias=True, attn_drop=0.0, proj_drop=0.0,
                 freq_base=100.0, freq_scale=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.px = patches_x
        self.py = patches_y
        self.im = image_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, *, viewmats_q, viewmats_kv, Ks_q=None, Ks_kv=None, **sdpa_kwargs):
        B, Tq, D = query.shape
        Bk, Tk, Dk = key.shape
        assert (Bk, Tk, Dk) == value.shape == (B, Tk, D)

        H, Hd = self.num_heads, self.head_dim
        q = self.q_proj(query).view(B, Tq, H, Hd).transpose(1, 2)  # (B,H,Tq,Hd)
        k = self.k_proj(key).view(B, Tk, H, Hd).transpose(1, 2)    # (B,H,Tk,Hd)
        v = self.v_proj(value).view(B, Tk, H, Hd).transpose(1, 2)  # (B,H,Tk,Hd)

        Cq = viewmats_q.shape[1]
        Ck = viewmats_kv.shape[1]
        assert Tq == Cq * self.px * self.py, "Tq must equal Cq*px*py"
        assert Tk == Ck * self.px * self.py, "Tk must equal Ck*px*py"

        # [MOD] 使用支持 Cq≠Ckv 的实现
        out = prope_cross_dot_product_attention(
            q, k, v,
            viewmats_q=viewmats_q,
            viewmats_kv=viewmats_kv,
            Ks_q=Ks_q,
            Ks_kv=Ks_kv,
            patches_x=self.px,
            patches_y=self.py,
            image_width=self.im,
            image_height=self.im,
            **sdpa_kwargs,
        )

        out = out.transpose(1, 2).reshape(B, Tq, D)
        out = self.proj_drop(self.out_proj(out))
        return out
