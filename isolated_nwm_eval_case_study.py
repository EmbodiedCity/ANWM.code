# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import json

from PIL import Image

# Eval
import lpips
from dreamsim import dreamsim
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
import distributed as dist


def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == 'lpips':
        general_lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(device) # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
                
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = general_lpips_loss_fn.forward(all_img0, all_img1)
            dist_avg = dist.mean()
            
            return dist_avg
    elif loss_fn_type == 'dreamsim':
        dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, device=device)
        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = preprocess(Image.open(img0_path)).to(device)
                img1 = preprocess(Image.open(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
            
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = dreamsim_loss_fn(all_img0, all_img1)
            dist_mean = dist.mean()
            
            return dist_mean
    elif loss_fn_type == 'fid':
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)
        
        return fid_metrics
    else:
        raise NotImplementedError
    
    return loss_fn


# ======================================================================
# ====================== CASE STUDY ADDITIONS BEGIN =====================
# ======================================================================
# 仅新增：允许把 case study 的保存目录指定为根目录（与总 json 对齐）
# base_exp_dir 若为 None，则回落到 exp_dir（保持兼容）
# ======================================================================

def evaluate(args, dataset_name, eval_type, metric_logger, loss_fns, gt_dir, exp_dir, secs, rollout_fps,
             base_exp_dir=None):
# ======================================================================
# ======================= CASE STUDY ADDITIONS END ======================
# ======================================================================
    lpips_loss_fn, dreamsim_loss_fn, fid_loss_fn = loss_fns
    
    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        image_idxs = (secs * rollout_fps) - 1
    elif eval_type == 'time':
        eval_name = eval_type
        image_idxs = secs.copy()
        
    # eps = os.listdir(gt_dir)
    # Make an intersection between GT and EXP episodes:
    # Only keep episodes that exist in both gt_dir and exp_dir
    all_eps = list(set(os.listdir(gt_dir)).intersection(set(os.listdir(exp_dir))))
    eps = []
    for ep in all_eps:
        missing = False
        for idx in image_idxs:
            frame_idx = int(idx)
            exp_img_path = os.path.join(exp_dir, ep, f"{frame_idx}.png")
            if not os.path.exists(exp_img_path):
                print(f"[Missing] {exp_img_path}")
                missing = True
                break
        if not missing:
            eps.append(ep)
        else:
            print(f"[Skip] {ep} missing frames for {eval_name}")

    # ======================================================================
    # ====================== CASE STUDY ADDITIONS BEGIN =====================
    # ======================================================================
    # 逐轨迹指标记录：case_records[ep][sec] = [{frame_idx, gt, pred, lpips, dreamsim}, ...]
    case_records = {}
    # 为了对齐 batch 中各 sec 的 (gt/pred) 列表项对应哪个 episode，需要显式记录映射
    # valid_eps_per_sec[sec] 与 gt_paths_batch[sec] / exp_paths_batch[sec] 的下标一一对应
    # 同时记录 frame_idx_batch[sec] 以便保存帧号
    # ======================================================================
    # ======================= CASE STUDY ADDITIONS END ======================
    # ======================================================================

    for batch_start in tqdm(range(0, len(eps), args.batch_size), total=(len(eps) + args.batch_size - 1) // args.batch_size):
        batch_eps = eps[batch_start:batch_start + args.batch_size]
        
        gt_batch, exp_batch = {}, {}
        gt_paths_batch, exp_paths_batch = {}, {}

        # ====================== CASE STUDY ADDITIONS BEGIN =====================
        frame_idx_batch = {}
        valid_eps_per_sec = {}
        # ======================= CASE STUDY ADDITIONS END ======================

        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []
            gt_paths_batch[sec] = []
            exp_paths_batch[sec] = []

            # ====================== CASE STUDY ADDITIONS BEGIN =================
            frame_idx_batch[sec] = []
            valid_eps_per_sec[sec] = []
            # ======================= CASE STUDY ADDITIONS END ==================

        
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)
        
            if not os.path.isdir(gt_ep_dir) and not os.path.isdir(exp_ep_dir):
                continue
        
            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f'{image_idx}.png')
                gt_sec_img = transforms.ToTensor()(Image.open(gt_sec_img_path).convert("RGB")).unsqueeze(0)
                exp_sec_img_path = os.path.join(exp_ep_dir, f'{image_idx}.png')
                exp_sec_img = transforms.ToTensor()(Image.open(exp_sec_img_path).convert("RGB")).unsqueeze(0)
                
                gt_batch[sec].append(gt_sec_img)
                gt_paths_batch[sec].append(gt_sec_img_path)
                exp_batch[sec].append(exp_sec_img)
                exp_paths_batch[sec].append(exp_sec_img_path)

                # ====================== CASE STUDY ADDITIONS BEGIN =============
                frame_idx_batch[sec].append(int(image_idx))
                valid_eps_per_sec[sec].append(ep)
                if ep not in case_records:
                    case_records[ep] = {}
                if int(sec) not in case_records[ep]:
                    case_records[ep][int(sec)] = []
                # ======================= CASE STUDY ADDITIONS END ==============

        for sec in secs:
            lpips_dists = lpips_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            dreamsim_dists = dreamsim_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            
            metric_logger.meters[f'{dataset_name}_{eval_name}_lpips_{sec}s'].update(lpips_dists, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_dreamsim_{sec}s'].update(dreamsim_dists, n=1)
            
            sec_gt_batch = torch.cat(gt_batch[sec], dim=0)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0)
            
            fid_loss_fn[sec].update(images=sec_gt_batch, is_real=True)
            fid_loss_fn[sec].update(images=sec_exp_batch, is_real=False)

            # ==================================================================
            # ====================== CASE STUDY ADDITIONS BEGIN =================
            # 逐条轨迹指标（不参与原评估统计，只做记录）：
            # 这里严格调用与“原评估”相同的函数，但以单元素列表形式逐条计算，
            # 确保不改变原评估逻辑/结果。
            for i, ep in enumerate(valid_eps_per_sec[sec]):
                single_lpips = lpips_loss_fn([gt_paths_batch[sec][i]], [exp_paths_batch[sec][i]])
                single_dreamsim = dreamsim_loss_fn([gt_paths_batch[sec][i]], [exp_paths_batch[sec][i]])

                case_records[ep][int(sec)].append({
                    "frame_idx": int(frame_idx_batch[sec][i]),
                    "gt": gt_paths_batch[sec][i],
                    "pred": exp_paths_batch[sec][i],
                    "lpips": float(single_lpips.item()) if hasattr(single_lpips, "item") else float(single_lpips),
                    "dreamsim": float(single_dreamsim.item()) if hasattr(single_dreamsim, "item") else float(single_dreamsim)
                    # FID 是分布指标，不记录逐条
                })
            # ======================= CASE STUDY ADDITIONS END ==================
            # ==================================================================
            
    for sec in secs:
        metric_logger.meters[f'{dataset_name}_{eval_name}_fid_{sec}s'].update(fid_loss_fn[sec].compute().item(), n=1)

    # ======================================================================
    # ====================== CASE STUDY ADDITIONS BEGIN =====================
    # ======================================================================
    # 仅主进程（若有分布式）写出 case study JSON；保存到根目录（与总 json 对齐）
    should_write = True
    if hasattr(dist, "is_main_process"):
        try:
            should_write = dist.is_main_process()
        except Exception:
            should_write = True

    save_root = base_exp_dir if base_exp_dir is not None else exp_dir
    if should_write:
        case_json_path = os.path.join(save_root, f'{dataset_name}_{eval_name}_cases.json')

        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)

        with open(case_json_path, 'w') as f:
            json.dump(case_records, f, indent=2, default=_default)
        print(f"[CaseStudy] saved per-episode records: {case_json_path}")
    # ======================================================================
    # ======================= CASE STUDY ADDITIONS END ======================
    # ======================================================================


def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    log_stats = {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}
    with open(log_p, 'w') as json_file:
        json.dump(log_stats, json_file, indent=4)  # indent=4 adds indentation for readability            


def main(args):
    device = 'cuda'
          
    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
    
    # These loss functions do not accumulate
    lpips_loss_fn = get_loss_fn('lpips', secs, device)
    dreamsim_loss_fn = get_loss_fn('dreamsim', secs, device)

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                try:
                    metric_logger = dist.MetricLogger(delimiter="  ")
                    print("Evaluating rollout", rollout_fps, dataset_name)
                    # Rollout (LPIPS, DreamSim, FID)
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    rollout_fid_loss_fn = get_loss_fn('fid', secs, device)
                    rollout_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, rollout_fid_loss_fn)
                    with torch.no_grad():
                        evaluate(args, dataset_name, 'rollout', metric_logger, rollout_loss_fns,
                                 gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps,
                                 base_exp_dir=args.exp_dir)
                    output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                    save_metric_to_disk(metric_logger, output_fn)
                except Exception as e:
                    print(e)

        if 'time' in args.eval_types:
            try:
                metric_logger = dist.MetricLogger(delimiter="  ")
                print("Evaluating time", dataset_name)
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                time_fid_loss_fn = get_loss_fn('fid', secs, device)
                time_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, time_fid_loss_fn)
                with torch.no_grad():
                    evaluate(args, dataset_name, eval_name, metric_logger, time_loss_fns,
                             gt_dataset_time_dir, exp_dataset_time_dir, secs, None,
                             base_exp_dir=args.exp_dir)
                output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--eval_types", type=str, default='time,rollout,rollout_video', help="evluations")
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument("--num_sec_eval", type=int, default=5, help="experiment name")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    
    parser.add_argument("--input_fps", type=int, default=4, help="experiment name")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    args.eval_types = args.eval_types.split(',')
    
    main(args)
