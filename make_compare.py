#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_compare.py

GT 与多组预测帧横向拼接为对比视频。

特性：
- 帧文件固定：0.png ~ 15.png（可调范围）
- 所有目录写到 rollout_1fps，id 从 --id 获取
- --id 可以是整数或 auto
  - 整数：只处理该 id
  - auto：自动扫描 id_0, id_1, ...，全部生成视频
- 输出文件名默认：compare_id{id}.mp4
- --out 可自定义文件名，支持 {id} 模板
- --out_dir 指定输出目录

依赖：pip install opencv-python
"""
import argparse
import os
import sys
import cv2
import numpy as np
from typing import List, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="GT rollout_1fps 目录（不带 id_x）")
    p.add_argument("--pred", required=True, nargs="+",
                   help="一个或多个预测 rollout_1fps 目录（不带 id_x）")
    p.add_argument("--id", required=True,
                   help="指定 id（整数）或 auto 自动扫描所有 id_*")
    p.add_argument("--out", default=None,
                   help="输出文件名，可包含模板 {id}（默认 compare_id{id}.mp4）")
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


def draw_label(img: np.ndarray, text: str, topleft=(10, 35),
               font_scale=1.0, thickness=2) -> np.ndarray:
    overlay = img.copy()
    x, y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 8
    cv2.rectangle(overlay, (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad // 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def ensure_even(w: int, h: int) -> Tuple[int, int]:
    if w % 2: w += 1
    if h % 2: h += 1
    return w, h


def last_dir_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def img_path(base_dir: str, id_val: int, fid: int) -> str:
    return os.path.join(base_dir, f"id_{id_val}", f"{fid}.png")


def decide_out_path(user_out: str, out_dir: str, id_val: int) -> str:
    result = user_out or f"compare_id{id_val}.mp4"
    result = result.replace("{id}", str(id_val))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isabs(result):
            result = os.path.join(out_dir, os.path.basename(result))
    return result


def find_all_ids(gt_dir: str) -> List[int]:
    ids = []
    for name in os.listdir(gt_dir):
        if name.startswith("id_"):
            try:
                ids.append(int(name.split("_")[1]))
            except ValueError:
                pass
    return sorted(ids)


def process_one_id(args, id_val: int):
    cols = [args.gt] + args.pred
    n_cols = len(cols)

    # 标签
    if args.labels is None or len(args.labels) != n_cols:
        labels = [last_dir_name(p) for p in cols]
    else:
        labels = args.labels

    out_path = decide_out_path(args.out, args.out_dir, id_val)
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

    # 高度
    first_imgs = [read_image(img_path(col, id_val, kept_ids[0])) for col in cols]
    target_h = max(im.shape[0] for im in first_imgs)
    resized_w = [resize_to_h(im, target_h).shape[1] for im in first_imgs]
    out_w, out_h = sum(resized_w), target_h
    if args.safe_even:
        out_w, out_h = ensure_even(out_w, out_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[ERR] 无法写入：{out_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 输出：{out_path} 尺寸：{out_w}x{out_h} FPS={args.fps} 栏位：{labels}")

    for i, fid in enumerate(kept_ids, 1):
        row_imgs = []
        for col_idx, col in enumerate(cols):
            img = read_image(img_path(col, id_val, fid))
            img = resize_to_h(img, target_h)
            if args.safe_even and (img.shape[1] % 2):
                img = cv2.resize(img, (img.shape[1] + 1, target_h))
            img = draw_label(img, labels[col_idx], (10, 35),
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
    print(f"[OK] 完成 id_{id_val}")


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
