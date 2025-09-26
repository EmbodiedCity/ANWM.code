#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_compare.py

GT 与多组预测帧横向拼接为对比视频；并默认额外生成：
1) 论文静态大图（PNG）：compare_id{id}.png（Times New Roman；顶部列标题写 idx 与时间；无覆盖到图内；无圆点）
2) 纯轨迹对比图（PNG）：traj_id{id}.png（Times New Roman；图例仅彩色文字，无圆点）
3) 所有输出按 id 打包到子目录：{out_dir or .}/id_{id}/

保持原有 CLI 参数不变：
--gt / --pred / --id / --out / --out_dir / --fps / --start / --end / --labels / --safe_even / --strict

依赖：opencv-python、numpy；（可选）Pillow 用于 TrueType 字体渲染（Times New Roman）
"""
import argparse
import os
import sys
import cv2
import numpy as np
import json
import csv
from typing import List, Tuple, Dict, Optional

# ====== 论文图默认风格（不增加 CLI 开关） ======
FIG_K = 6             # 静态图均匀采帧数（不足则全取）
GRID_GAP = 10         # 子图间距
GRID_PAD = 24         # 外边距
HEADER_H = 56         # 顶部列标题栏高度
SIDEBAR_W_MIN = 160   # 左侧行标题最小宽（自适应）
SIDEBAR_GAP = 6       # 侧栏到首列的间距（更贴近第一张图）
TRAJ_TAIL = 999999    # 轨迹历史长度
TRAJ_THICK = 3        # 轨迹线宽
TRAJ_RADIUS = 6       # 轨迹当前点半径
TRAJ_ALPHA = 1.0      # 轨迹叠加透明度
LAYOUT_ROWS_METHODS = True  # 行=方法, 列=时间帧
PNG_COMPRESSION = 3   # 0-9，越大压缩越高（无损）

# ====== 文本排版（Times New Roman） ======
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

_FONT_CACHE = {}  # size_px -> PIL ImageFont
_TTF_PATH_CACHE = None

def _find_ttf_font() -> Optional[str]:
    """在常见系统路径中寻找 Times New Roman；否则找兼容衬线体；找不到返回 None。"""
    global _TTF_PATH_CACHE
    if _TTF_PATH_CACHE is not None:
        return _TTF_PATH_CACHE
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        # Windows
        "C:\\Windows\\Fonts\\times.ttf",
        "C:\\Windows\\Fonts\\Times New Roman.ttf",
        # Linux (可能需安装 mscorefonts)
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        # 常见替代 serif
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerif-Regular.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            _TTF_PATH_CACHE = p
            return p
    _TTF_PATH_CACHE = None
    return None

def _get_pil_font(size_px: int) -> Optional["ImageFont.FreeTypeFont"]:
    if not _HAS_PIL:
        return None
    if size_px in _FONT_CACHE:
        return _FONT_CACHE[size_px]
    ttf = _find_ttf_font()
    if ttf is None:
        return None
    try:
        font = ImageFont.truetype(ttf, size_px)
        _FONT_CACHE[size_px] = font
        return font
    except Exception:
        return None

def _draw_text(img_bgr: np.ndarray,
               text: str,
               org_xy: Tuple[int, int],
               size_px: int,
               color_bgr: Tuple[int, int, int] = (0, 0, 0),
               stroke_px: int = 0,
               stroke_color_bgr: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """在 BGR 图像上以 TrueType 绘制（优先 Times New Roman），左上角对齐 org_xy。"""
    font = _get_pil_font(size_px)
    if font is None:
        # Fallback 到 OpenCV Hershey（非新罗马）
        scale = max(0.5, size_px / 28.0)
        thickness = max(1, int(round(size_px / 18)))
        cv2.putText(img_bgr, text, org_xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    stroke_rgb = (stroke_color_bgr[2], stroke_color_bgr[1], stroke_color_bgr[0])
    draw.text(org_xy, text, font=font, fill=color_rgb,
              stroke_width=stroke_px if stroke_px > 0 else 0,
              stroke_fill=stroke_rgb if stroke_px > 0 else None)
    img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def _measure_text(text: str, size_px: int) -> Tuple[int, int]:
    font = _get_pil_font(size_px)
    if font is not None:
        try:
            bbox = font.getbbox(text)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            return max(1, w), max(1, h)
        except Exception:
            pass
    scale = max(0.5, size_px / 28.0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(round(size_px/18))))
    return max(1, tw), max(1, th)

# ===================== 基础工具 =====================

def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def resize_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / h)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def ensure_even(w: int, h: int) -> Tuple[int, int]:
    if w % 2: w += 1
    if h % 2: h += 1
    return w, h

def last_dir_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))

def img_path(base_dir: str, id_val: int, fid: int) -> str:
    return os.path.join(base_dir, f"id_{id_val}", f"{fid}.png")

def bundle_dir_for_id(out_dir: Optional[str], id_val: int) -> str:
    """确定并创建每个 id 的打包子目录。"""
    base = out_dir if out_dir else os.getcwd()
    bundle = os.path.join(base, f"id_{id_val}")
    os.makedirs(bundle, exist_ok=True)
    return bundle

def decide_out_path(user_out: Optional[str], out_dir: Optional[str], id_val: int, default_name: str) -> str:
    """
    所有输出统一放在 {out_dir or .}/id_{id}/ 子目录。
    若 user_out 提供，仅取其 basename 并套用 {id}（避免用户路径破坏打包结构）。
    """
    bundle = bundle_dir_for_id(out_dir, id_val)
    filename = (user_out or default_name).replace("{id}", str(id_val))
    filename = os.path.basename(filename)
    return os.path.join(bundle, filename)

def find_all_ids(gt_dir: str) -> List[int]:
    ids = []
    for name in os.listdir(gt_dir):
        if name.startswith("id_"):
            try:
                ids.append(int(name.split("_")[1]))
            except ValueError:
                pass
    return sorted(ids)

# ===================== 轨迹加载与绘制（自动发现） =====================

def _parse_traj_colors(n: int) -> List[Tuple[int, int, int]]:
    # BGR（OpenCV）
    base = [
        (60, 180, 60),
        (60, 60, 220),
        (220, 160, 60),
        (180, 60, 180),
        (60, 180, 180),
        (128, 128, 255),
        (128, 255, 128),
        (255, 128, 128),
        (255, 200, 100),
        (80, 80, 80),
    ]
    while len(base) < n:
        base = base + base
    return base[:n]

def _coord_mode_auto(traj: Dict[int, Tuple[float, float]]) -> str:
    if not traj:
        return "pixel"
    maxv = max(max(abs(x), abs(y)) for (x, y) in traj.values())
    return "norm" if maxv <= 2.0 else "pixel"

def _scale_point(x: float, y: float, src_wh: Tuple[int, int],
                 dst_wh: Tuple[int, int], mode: str) -> Tuple[int, int]:
    sw, sh = src_wh
    dw, dh = dst_wh
    if mode == "norm":
        px = int(round(x * dw))
        py = int(round(y * dh))
    else:
        sx = dw / max(1, sw)
        sy = dh / max(1, sh)
        px = int(round(x * sx))
        py = int(round(y * sy))
    px = max(0, min(dw - 1, px))
    py = max(0, min(dh - 1, py))
    return px, py

def overlay_trajectory(img: np.ndarray,
                       traj: Dict[int, Tuple[float, float]],
                       fid: int,
                       src_wh: Tuple[int, int],
                       color: Tuple[int, int, int],
                       tail: int = TRAJ_TAIL,
                       radius: int = TRAJ_RADIUS,
                       thickness: int = TRAJ_THICK,
                       alpha: float = TRAJ_ALPHA) -> np.ndarray:
    if not traj:
        return img
    mode = _coord_mode_auto(traj)

    keys = [k for k in traj.keys() if k <= fid]
    if not keys:
        return img
    keys.sort()
    if len(keys) > tail:
        keys = keys[-tail:]

    h, w = img.shape[:2]
    overlay = img.copy()
    prev_pt = None
    for k in keys:
        x, y = traj[k]
        px, py = _scale_point(x, y, src_wh, (w, h), mode)
        if prev_pt is not None:
            cv2.line(overlay, prev_pt, (px, py), color, thickness, cv2.LINE_AA)
        prev_pt = (px, py)

    if fid in traj:
        x, y = traj[fid]
        cx, cy = _scale_point(x, y, src_wh, (w, h), mode)
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)

    if alpha >= 1.0:
        img[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def load_traj_file(path: str) -> Dict[int, Tuple[float, float]]:
    if path is None or path.lower() == "none":
        return {}
    if not os.path.isfile(path):
        return {}

    ext = os.path.splitext(path)[1].lower()
    data: Dict[int, Tuple[float, float]] = {}

    try:
        if ext in [".csv", ".tsv"]:
            delim = "," if ext == ".csv" else "\t"
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delim)
                for row in reader:
                    frame_key = row.get("frame", row.get("fid", None))
                    if frame_key is None:
                        continue
                    try:
                        fid = int(frame_key)
                    except:
                        continue
                    if "x" in row and "y" in row:
                        x, y = float(row["x"]), float(row["y"])
                    elif "u" in row and "v" in row:
                        x, y = float(row["u"]), float(row["v"])
                    else:
                        continue
                    data[fid] = (x, y)
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "points" in obj:
                obj = obj["points"]
            if isinstance(obj, list):
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    frame_key = it.get("frame", it.get("fid", None))
                    if frame_key is None:
                        continue
                    try:
                        fid = int(frame_key)
                    except:
                        continue
                    if "x" in it and "y" in it:
                        x, y = float(it["x"]), float(it["y"])
                    elif "u" in it and "v" in it:
                        x, y = float(it["u"]), float(it["v"])
                    else:
                        continue
                    data[fid] = (x, y)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        fid = int(k)
                    except:
                        continue
                    if isinstance(v, dict):
                        if "x" in v and "y" in v:
                            data[fid] = (float(v["x"]), float(v["y"]))
                        elif "u" in v and "v" in v:
                            data[fid] = (float(v["u"]), float(v["v"]))
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        data[fid] = (float(v[0]), float(v[1]))
    except Exception:
        return {}
    return data

def _guess_traj_file(base_dir: str, id_val: int) -> Optional[str]:
    """
    自动发现轨迹文件：
    1) {base}/id_{id}/[traj|trajectory|path|trace].(csv|json)
    2) {base}/[traj|trajectory|path|trace]_id{id}.(csv|json)
    """
    iddir = os.path.join(base_dir, f"id_{id_val}")
    names = ["traj", "trajectory", "path", "trace"]
    exts = [".csv", ".json"]
    for stem in names:
        for ext in exts:
            p = os.path.join(iddir, stem + ext)
            if os.path.isfile(p):
                return p
    for stem in names:
        for ext in exts:
            p = os.path.join(base_dir, f"{stem}_id{id_val}{ext}")
            if os.path.isfile(p):
                return p
    return None

# ===================== 主流程 =====================

def draw_video_overlay_label(img: np.ndarray, text: str,
                             topleft=(10, 35),
                             font_scale_cli=1.0,
                             thickness_cli=2) -> np.ndarray:
    """视频帧上的覆盖文本：半透明底 + Times New Roman 文本（若可用）。"""
    x, y = topleft
    size_px = max(18, int(round(24 * font_scale_cli)))
    tw, th = _measure_text(text, size_px)
    pad = 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad),
                  (x + tw + pad, y + pad // 2),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    _draw_text(img, text, (x, y - th), size_px,
               color_bgr=(255, 255, 255), stroke_px=1, stroke_color_bgr=(0, 0, 0))
    return img

def process_one_id(args, id_val: int):
    cols = [args.gt] + args.pred
    n_cols = len(cols)

    # 标签
    if args.labels is None or len(args.labels) != n_cols:
        labels = [last_dir_name(p) for p in cols]
    else:
        labels = args.labels

    # 输出到绑定子目录
    bundle = bundle_dir_for_id(args.out_dir, id_val)
    print(f"[INFO] 输出目录：{bundle}")

    out_video = decide_out_path(args.out, args.out_dir, id_val, default_name=f"compare_id{id_val}.mp4")
    frame_ids = list(range(args.start, args.end + 1))

    kept_ids = []
    for fid in frame_ids:
        ok_all = True
        for col in cols:
            if not os.path.isfile(img_path(col, id_val, fid)):
                ok_all = False
        if ok_all:
            kept_ids.append(fid)
        elif args.strict:
            print(f"[ERR] id_{id_val} 缺帧 {fid}", file=sys.stderr)
            sys.exit(1)

    if not kept_ids:
        print(f"[WARN] id_{id_val} 没有完整帧，跳过")
        return

    # 目标高度：取首帧各方法最大高度
    first_imgs = [read_image(img_path(col, id_val, kept_ids[0])) for col in cols]
    target_h = max(im.shape[0] for im in first_imgs)

    # ---------- 1) 生成视频 ----------
    resized_w = [resize_to_h(im, target_h).shape[1] for im in first_imgs]
    out_w, out_h = sum(resized_w), target_h
    if args.safe_even:
        out_w, out_h = ensure_even(out_w, out_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[ERR] 无法写入：{out_video}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 输出视频：{out_video} 尺寸：{out_w}x{out_h} FPS={args.fps} 栏位：{labels}")

    for i, fid in enumerate(kept_ids, 1):
        row_imgs = []
        for col_idx, col in enumerate(cols):
            img = read_image(img_path(col, id_val, fid))
            img = resize_to_h(img, target_h)
            if args.safe_even and (img.shape[1] % 2):
                img = cv2.resize(img, (img.shape[1] + 1, target_h))
            img = draw_video_overlay_label(img, labels[col_idx], (10, 35),
                                           args.font_scale, args.thickness)
            row_imgs.append(img)
        combo = np.hstack(row_imgs)
        if args.safe_even:
            h, w = combo.shape[:2]
            w2, h2 = ensure_even(w, h)
            if (w2, h2) != (w, h):
                combo = cv2.resize(combo, (w2, h2))
        writer.write(combo)
        if i % 5 == 0 or i == len(kept_ids):
            print(f"[INFO] id_{id_val} 写入 {i}/{len(kept_ids)}")
    writer.release()
    print(f"[OK] 完成视频 id_{id_val}")

    # ---------- 2) 论文静态大图 & 3) 轨迹面板 ----------
    _post_static_and_traj_images(args, id_val, cols, labels, kept_ids, target_h)

def _choose_fig_frames(kept_ids: List[int], k: int) -> List[int]:
    if len(kept_ids) <= k:
        return kept_ids
    idxs = np.linspace(0, len(kept_ids) - 1, k, dtype=int).tolist()
    return [kept_ids[i] for i in idxs]

def _read_first_dims_per_col(cols: List[str], id_val: int, fid0: int, target_h: int) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    src_wh, dst_wh = [], []
    for col in cols:
        path = img_path(col, id_val, fid0)
        im0 = read_image(path)
        h0, w0 = im0.shape[:2]
        src_wh.append((w0, h0))
        im1 = resize_to_h(im0, target_h)
        h1, w1 = im1.shape[:2]
        dst_wh.append((w1, h1))
    return src_wh, dst_wh

def _compose_grid_images_for_paper(args,
                                   labels: List[str],
                                   fig_fids: List[int],
                                   cols: List[str],
                                   id_val: int,
                                   target_h: int,
                                   trajs: List[Dict[int, Tuple[float,float]]],
                                   src_wh_list: List[Tuple[int,int]],
                                   colors: List[Tuple[int,int,int]]) -> np.ndarray:
    # 对每列（帧）计算统一宽度；tile 居中填充到该宽度；白底
    col_widths = []
    tiles: Dict[Tuple[int,int], np.ndarray] = {}
    for c, fid in enumerate(fig_fids):
        max_w = 0
        resized_per_row = []
        for r, base_dir in enumerate(cols):
            img = read_image(img_path(base_dir, id_val, fid))
            img = resize_to_h(img, target_h)
            resized_per_row.append(img)
            max_w = max(max_w, img.shape[1])
        col_widths.append(max_w)
        for r, img in enumerate(resized_per_row):
            if trajs and r < len(trajs) and trajs[r]:
                img = overlay_trajectory(
                    img, trajs[r], fid,
                    src_wh=src_wh_list[r],
                    color=colors[r],
                    tail=TRAJ_TAIL,
                    radius=TRAJ_RADIUS,
                    thickness=TRAJ_THICK,
                    alpha=TRAJ_ALPHA,
                )
            h, w = img.shape[:2]
            pad_w = col_widths[c] - w
            if pad_w > 0:
                left = pad_w // 2
                right = pad_w - left
                pad_left = np.full((h, left, 3), 255, np.uint8)
                pad_right = np.full((h, right, 3), 255, np.uint8)
                img = np.hstack([pad_left, img, pad_right])
            tiles[(r, c)] = img

    n_rows = len(cols) if LAYOUT_ROWS_METHODS else len(fig_fids)
    n_cols = len(fig_fids) if LAYOUT_ROWS_METHODS else len(cols)

    body_w = sum(col_widths) + GRID_GAP * (n_cols - 1)
    body_h = n_rows * target_h + GRID_GAP * (n_rows - 1)

    # 侧栏宽度自适应（右对齐到首列），更贴近第一张图
    sidebar_w = SIDEBAR_W_MIN
    for lab in labels:
        tw, th = _measure_text(lab, 28)
        sidebar_w = max(sidebar_w, tw + 12)   # 只留 12px 余量

    total_w = GRID_PAD + sidebar_w + SIDEBAR_GAP + body_w + GRID_PAD
    total_h = GRID_PAD + HEADER_H + GRID_GAP + body_h + GRID_PAD
    canvas = np.full((total_h, total_w, 3), 255, np.uint8)  # 白底

    # 列起点：从侧栏右缘 + SIDEBAR_GAP 开始
    col_x0 = []
    cur = GRID_PAD + sidebar_w + SIDEBAR_GAP
    for c in range(n_cols):
        col_x0.append(cur)
        cur += col_widths[c] + GRID_GAP

    # 顶部列标题 —— 同时写 idx 与时间（时间=idx+1s）
    for c, _fid in enumerate(fig_fids):
        text = f"idx={c}, {c+1}s"
        tw, th = _measure_text(text, 26)
        cx = col_x0[c] + col_widths[c] // 2 - tw // 2
        cy = GRID_PAD + HEADER_H // 2 - th // 2
        _draw_text(canvas, text, (cx, cy), 26, color_bgr=(0, 0, 0), stroke_px=0)

    # 左侧行标题：右对齐到侧栏右缘（更靠近首列；无圆点）
    row_y0 = GRID_PAD + HEADER_H
    for r, lab in enumerate(labels):
        y_center = row_y0 + r * (target_h + GRID_GAP) + target_h // 2
        tw, th = _measure_text(lab, 28)
        tx = GRID_PAD + sidebar_w - tw          # 右对齐到侧栏
        ty = y_center - th // 2
        _draw_text(canvas, lab, (tx, ty), 28, color_bgr=(0, 0, 0), stroke_px=0)

    # 放置 tiles
    for r in range(n_rows):
        for c in range(n_cols):
            tile = tiles[(r, c)]
            x = col_x0[c]
            y = row_y0 + r * (target_h + GRID_GAP)
            h, w = tile.shape[:2]
            canvas[y:y+h, x:x+w] = tile

    # 分隔线（淡灰）
    gray = (220, 220, 220)
    for r in range(1, n_rows):
        y = row_y0 + r * (target_h + GRID_GAP) - GRID_GAP // 2
        cv2.line(canvas, (GRID_PAD + sidebar_w + SIDEBAR_GAP, y), (total_w - GRID_PAD, y), gray, 1, cv2.LINE_AA)
    for c in range(1, n_cols):
        x = col_x0[c] - GRID_GAP // 2
        cv2.line(canvas, (x, GRID_PAD + HEADER_H), (x, total_h - GRID_PAD), gray, 1, cv2.LINE_AA)

    return canvas

def _make_traj_panel_for_paper(labels: List[str],
                               fig_fids: List[int],
                               trajs: List[Dict[int, Tuple[float,float]]],
                               src_wh_list: List[Tuple[int,int]],
                               target_h: int,
                               colors: List[Tuple[int,int,int]]) -> Optional[np.ndarray]:
    if not any(bool(t) for t in trajs):
        return None
    # 画布大小：按第一方法缩放后的尺寸
    w0, h0 = src_wh_list[0]
    dummy = np.zeros((h0, w0, 3), dtype=np.uint8)
    dummy = resize_to_h(dummy, target_h)
    H, W = dummy.shape[:2]
    canvas = np.full((H + GRID_PAD*2, W + GRID_PAD*2, 3), 255, np.uint8)
    ox, oy = GRID_PAD, GRID_PAD

    # 绘制轨迹
    for midx, traj in enumerate(trajs):
        if not traj:
            continue
        mode = _coord_mode_auto(traj)
        keys = sorted(traj.keys())
        if TRAJ_TAIL < 999999 and len(keys) > TRAJ_TAIL:
            keys = keys[-TRAJ_TAIL:]
        prev_pt = None
        for k in keys:
            x, y = traj[k]
            px, py = _scale_point(x, y, src_wh_list[midx], (W, H), mode)
            if prev_pt is not None:
                cv2.line(canvas, (ox+prev_pt[0], oy+prev_pt[1]), (ox+px, oy+py), colors[midx], TRAJ_THICK, cv2.LINE_AA)
            prev_pt = (px, py)
        if keys:
            x, y = traj[keys[-1]]
            cx, cy = _scale_point(x, y, src_wh_list[midx], (W, H), mode)
            cv2.circle(canvas, (ox+cx, oy+cy), TRAJ_RADIUS + 1, colors[midx], -1, cv2.LINE_AA)

    # 图例：仅文字（有色文字，无前置圆点）
    legend_x, legend_y = ox, oy
    yy = legend_y
    for midx, lab in enumerate(labels):
        tw, th = _measure_text(lab, 24)
        _draw_text(canvas, lab, (legend_x, yy), 24, color_bgr=colors[midx], stroke_px=0)
        yy += th + 6

    return canvas

def _post_static_and_traj_images(args, id_val, cols, labels, kept_ids, target_h):
    # 选帧
    fig_fids = _choose_fig_frames(kept_ids, FIG_K)

    # 加载轨迹（自动发现）
    traj_list: List[Dict[int, Tuple[float,float]]] = []
    for base in cols:
        tpath = _guess_traj_file(base, id_val)
        traj_list.append(load_traj_file(tpath) if tpath else {})

    colors = _parse_traj_colors(len(cols))
    src_wh_list, _ = _read_first_dims_per_col(cols, id_val, fig_fids[0], target_h)

    # 论文静态大图（写入 id_{id} 子目录）
    mosaic = _compose_grid_images_for_paper(
        args, labels, fig_fids, cols, id_val, target_h,
        traj_list, src_wh_list, colors
    )
    static_path = decide_out_path(None, args.out_dir, id_val, default_name=f"compare_id{id_val}.png")
    cv2.imwrite(static_path, mosaic, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    print(f"[OK] 静态图已保存：{static_path}（列标题 idx 与时间已写）")

    # 轨迹面板（写入 id_{id} 子目录）
    panel = _make_traj_panel_for_paper(labels, fig_fids, traj_list, src_wh_list, target_h, colors)
    if panel is not None:
        traj_path = decide_out_path(None, args.out_dir, id_val, default_name=f"traj_id{id_val}.png")
        cv2.imwrite(traj_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        print(f"[OK] 轨迹面板已保存：{traj_path}")
    else:
        print("[INFO] 未发现任何轨迹文件，跳过轨迹面板。")

# ===================== CLI & 入口 =====================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="GT rollout_1fps 目录（不带 id_x）")
    p.add_argument("--pred", required=True, nargs="+",
                   help="一个或多个预测 rollout_1fps 目录（不带 id_x）")
    p.add_argument("--id", required=True,
                   help="指定 id（整数）或 auto 自动扫描所有 id_*")
    p.add_argument("--out", default=None,
                   help="输出视频文件名，可包含模板 {id}（默认 compare_id{id}.mp4）")
    p.add_argument("--out_dir", default=None, help="输出文件夹（默认当前目录）")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=15)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--font_scale", type=float, default=1.0)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--safe_even", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.id == "auto":
        ids = find_all_ids(args.gt)
        print(f"[INFO] auto 模式，找到 {ids}")
        for id_val in ids:
            process_one_id(args, id_val)
    else:
        try:
            id_val = int(args.id)
        except ValueError:
            print("[ERR] --id 必须是整数或 'auto'", file=sys.stderr)
            sys.exit(1)
        process_one_id(args, id_val)

if __name__ == "__main__":
    main()
