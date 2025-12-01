#!/usr/bin/env python3
"""
精简脚本：检查特定checkpoint的epoch和step信息
"""

import torch
from pathlib import Path

# ============ 配置区域：直接修改这里的路径 ============
NAME = "nwm_cdit_airvln_v5_7"
CHECKPOINT_DIR = f"/data1/tpz/nwm-main/logs/{NAME}/checkpoints"
LOG_FILE = f"/data1/tpz/nwm-main/logs/{NAME}/log.txt"
# =======================================================


def check_checkpoint(checkpoint_path):
    """检查单个checkpoint"""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        epoch = ckpt.get('epoch', None)
        train_steps = ckpt.get('train_steps', None)
        
        print(f"Epoch:        {epoch}")
        print(f"Train Steps:  {train_steps}")
        print(f"Keys:         {', '.join(ckpt.keys())}")
        
        return {'epoch': epoch, 'steps': train_steps, 'path': checkpoint_path}
    except Exception as e:
        print(f"错误: {e}")
        return None


def main():
    # 扫描checkpoint目录
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        print(f"错误: Checkpoint目录不存在: {CHECKPOINT_DIR}")
        return
    
    checkpoint_files = sorted(ckpt_dir.glob("*.pth.tar"))
    
    if not checkpoint_files:
        print(f"未找到checkpoint文件")
        return
    
    print(f"找到 {len(checkpoint_files)} 个checkpoint文件:")
    
    # 显示所有checkpoint信息
    infos = []
    for ckpt_file in checkpoint_files:
        info = check_checkpoint(str(ckpt_file))
        if info:
            infos.append(info)
    
    # 汇总表格
    if infos:
        print(f"\n{'='*70}")
        print("汇总信息")
        print(f"{'='*70}")
        print(f"{'文件名':<50} {'Epoch':<10} {'Steps':<15}")
        print("-" * 75)
        for info in infos:
            filename = Path(info['path']).name
            epoch_str = str(info['epoch']) if info['epoch'] is not None else "N/A"
            steps_str = str(info['steps']) if info['steps'] is not None else "N/A"
            print(f"{filename:<50} {epoch_str:<10} {steps_str:<15}")


if __name__ == "__main__":
    main()
