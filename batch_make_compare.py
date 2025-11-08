#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用 make_compare.py 为多个ID生成对比视频
"""
import subprocess
import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path

def read_id_list(id_file: str) -> list:
    """读取ID列表文件，返回ID数字列表（用于make_compare.py）"""
    ids = []
    with open(id_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 处理 id_xxx 格式，提取数字部分
                if line.startswith('id_'):
                    id_num = line.replace('id_', '')
                    try:
                        ids.append(int(id_num))
                    except ValueError:
                        print(f"[WARN] 无法解析ID: {line}，跳过")
                else:
                    # 如果只是数字，直接使用
                    try:
                        ids.append(int(line))
                    except ValueError:
                        print(f"[WARN] 无法解析ID: {line}，跳过")
    return ids

def find_max_frame_in_dir(base_dir: str, id_val: int) -> int:
    """查找目录中最大的帧索引"""
    id_dir = os.path.join(base_dir, f"id_{id_val}")
    if not os.path.isdir(id_dir):
        return -1
    max_frame = -1
    for fname in os.listdir(id_dir):
        if fname.endswith('.png'):
            try:
                frame_num = int(fname.replace('.png', ''))
                max_frame = max(max_frame, frame_num)
            except ValueError:
                pass
    return max_frame

def run_make_compare_combined(id_val: int, args_dict: dict, dry_run: bool = False):
    """运行 make_compare.py 为单个ID生成合并视频（包含4fps和1fps）"""
    # 只使用 4fps 的 GT，合并 4fps 和 1fps 的 pred
    gt_dir = args_dict['gt']  # 只用 4fps 的 GT
    pred_dirs = args_dict['pred'].copy()  # 4fps 的 pred
    
    # 如果有 1fps 数据，只添加 pred（不添加 GT）
    if args_dict.get('pred_1fps'):
        pred_dirs.extend(args_dict['pred_1fps'])
    
    # 合并所有列：GT(4fps) + Pred(4fps) + Pred(1fps)
    all_cols = [gt_dir] + pred_dirs
    
    # 为 1fps 列创建缺失帧的符号链接（映射到对应的 1fps 帧）
    # 1fps 列每4帧更新一次，所以帧 0,1,2,3 -> 显示帧0，帧 4,5,6,7 -> 显示帧4，等等
    end_frame = args_dict['end']
    if args_dict.get('pred_1fps'):
        import tempfile
        import shutil
        
        # 为每个 1fps 列创建临时目录，包含所有需要的帧（通过符号链接）
        temp_dirs = []
        for pred_1fps_dir in args_dict['pred_1fps']:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix=f'temp_1fps_{id_val}_')
            temp_dirs.append(temp_dir)
            
            # 复制 id 目录结构
            src_id_dir = os.path.join(pred_1fps_dir, f"id_{id_val}")
            dst_id_dir = os.path.join(temp_dir, f"id_{id_val}")
            os.makedirs(dst_id_dir, exist_ok=True)
            
            # 先找到 1fps 列中实际存在的最大帧
            max_available_frame = find_max_frame_in_dir(pred_1fps_dir, id_val)
            
            # 为每一帧创建符号链接（映射到对应的 1fps 帧）
            # 映射规则：4fps 的帧 fid -> 1fps 的帧 (fid // 4)
            # 例如：4fps 帧 0,1,2,3 -> 1fps 帧 0
            #       4fps 帧 4,5,6,7 -> 1fps 帧 1
            #       4fps 帧 8,9,10,11 -> 1fps 帧 2
            start_frame = args_dict.get('start') if args_dict.get('start') is not None else 0
            for fid in range(start_frame, end_frame + 1):
                # 映射规则：fid -> fid // 4（每4帧更新一次，1fps的1帧对应4fps的4帧）
                mapped_fid = fid // 4
                
                # 如果映射的帧超过了实际存在的最大帧，使用最大帧
                if mapped_fid > max_available_frame:
                    mapped_fid = max_available_frame
                
                src_frame = os.path.join(src_id_dir, f"{mapped_fid}.png")
                dst_frame = os.path.join(dst_id_dir, f"{fid}.png")
                
                if os.path.isfile(src_frame):
                    # 创建符号链接
                    if os.path.exists(dst_frame):
                        os.remove(dst_frame)
                    os.symlink(src_frame, dst_frame)
                else:
                    # 如果源帧不存在，尝试找最近的帧（向下查找）
                    found = False
                    for check_fid in range(mapped_fid, -1, -1):
                        check_src = os.path.join(src_id_dir, f"{check_fid}.png")
                        if os.path.isfile(check_src):
                            if os.path.exists(dst_frame):
                                os.remove(dst_frame)
                            os.symlink(check_src, dst_frame)
                            found = True
                            break
                    if not found:
                        print(f"[WARN] id_{id_val} 1fps 列找不到任何帧")
            
            # 更新 pred_dirs 中的路径，使用临时目录
            pred_dirs[pred_dirs.index(pred_1fps_dir)] = temp_dir
        
        # 保存临时目录列表，以便后续清理
        args_dict['_temp_dirs'] = temp_dirs
    
    # 使用用户提供的标签（如果提供了完整的标签列表）
    if args_dict.get('labels_combined'):
        labels = args_dict['labels_combined']
    else:
        # 否则自动生成标签
        labels = args_dict['labels'].copy()  # GT + 4fps pred
        if args_dict.get('pred_1fps'):
            # 为 1fps 的 pred 添加后缀
            labels_1fps = [f"{label}_1fps" for label in args_dict['labels'][1:]]  # 跳过 GT
            labels = labels + labels_1fps
    
    # 使用 4fps 的 fps 值（因为合并后视频统一使用一个 fps）
    fps = args_dict.get('fps', 1)
    output_name = f'compare_id{id_val}.mp4'
    
    cmd = [
        sys.executable,
        'make_compare.py',
        '--gt', gt_dir,  # 只用 4fps 的 GT
        '--pred'] + pred_dirs + [  # 4fps pred + 1fps pred
        '--id', str(id_val),
        '--out_dir', args_dict['out_dir'],
        '--end', str(end_frame),
        '--labels'] + labels + [
        '--fps', str(int(fps)) if fps == int(fps) else str(fps),
        '--out', output_name,
    ]
    
    if args_dict.get('safe_even', False):
        cmd.append('--safe_even')
    
    if args_dict.get('start', None):
        cmd.extend(['--start', str(args_dict['start'])])
    
    print(f"\n{'='*80}")
    print(f"处理 ID: id_{id_val} (合并 4fps + 1fps, fps={fps})")
    print(f"列数: {len(all_cols)} (GT_4fps + {len(args_dict['pred'])} pred_4fps + {len(args_dict.get('pred_1fps', []))} pred_1fps)")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    if dry_run:
        print("[DRY RUN] 不会实际执行")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, cwd=args_dict.get('work_dir', '.'))
        print(f"[OK] id_{id_val} (合并视频) 完成")
        
        # 清理临时目录
        if args_dict.get('_temp_dirs'):
            for temp_dir in args_dict['_temp_dirs']:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"[WARN] 清理临时目录失败 {temp_dir}: {e}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] id_{id_val} (合并视频) 失败: {e}", file=sys.stderr)
        
        # 清理临时目录
        if args_dict.get('_temp_dirs'):
            for temp_dir in args_dict['_temp_dirs']:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        
        return False
    except Exception as e:
        print(f"[ERROR] id_{id_val} (合并视频) 出错: {e}", file=sys.stderr)
        
        # 清理临时目录
        if args_dict.get('_temp_dirs'):
            for temp_dir in args_dict['_temp_dirs']:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        
        return False

def main():
    parser = argparse.ArgumentParser(description='批量生成对比视频')
    parser.add_argument('--id_file', type=str, required=True,
                       help='ID列表文件路径')
    parser.add_argument('--gt', type=str, required=True,
                       help='GT rollout目录')
    parser.add_argument('--pred', nargs='+', required=True,
                       help='预测 rollout 目录（一个或多个）')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='标签列表（与GT+pred数量对应）')
    parser.add_argument('--end', type=int, default=51,
                       help='结束帧索引')
    parser.add_argument('--start', type=int, default=None,
                       help='开始帧索引（可选）')
    parser.add_argument('--fps', type=float, default=1,
                       help='视频FPS（用于rollout_4fps，默认1）')
    parser.add_argument('--gt_1fps', type=str, default=None,
                       help='GT rollout_1fps目录（已废弃，不再使用）')
    parser.add_argument('--pred_1fps', nargs='+', default=None,
                       help='预测 rollout_1fps 目录（一个或多个，将合并到视频中，但不包含GT）')
    parser.add_argument('--fps_1fps', type=float, default=0.25,
                       help='视频FPS（用于rollout_1fps，默认0.25）')
    parser.add_argument('--safe_even', action='store_true',
                       help='使用安全偶数尺寸')
    parser.add_argument('--work_dir', type=str, default=None,
                       help='工作目录（默认当前目录）')
    parser.add_argument('--dry_run', action='store_true',
                       help='仅显示命令，不实际执行')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的视频文件')
    parser.add_argument('--max_workers', type=int, default=1,
                       help='最大并行数（默认1，串行执行）')
    
    args = parser.parse_args()
    
    # 读取ID列表
    ids = read_id_list(args.id_file)
    print(f"[INFO] 从 {args.id_file} 读取到 {len(ids)} 个ID")
    
    if not ids:
        print("[ERROR] 没有找到有效的ID", file=sys.stderr)
        sys.exit(1)
    
    # 确定处理模式：合并模式（如果指定了 1fps pred 数据）
    combine_mode = args.pred_1fps is not None
    if combine_mode:
        if len(args.pred_1fps) != len(args.pred):
            print(f"[ERROR] rollout_1fps 的预测目录数量 ({len(args.pred_1fps)}) 与 rollout_4fps ({len(args.pred)}) 不匹配", file=sys.stderr)
            sys.exit(1)
    
    # 准备参数
    args_dict = {
        'gt': args.gt,
        'pred': args.pred,
        'out_dir': args.out_dir,
        'labels': args.labels,
        'labels_combined': args.labels if combine_mode and len(args.labels) == (1 + len(args.pred) + len(args.pred_1fps)) else None,
        'end': args.end,
        'start': args.start,
        'fps': args.fps,
        'fps_1fps': args.fps_1fps,
        'gt_1fps': args.gt_1fps,
        'pred_1fps': args.pred_1fps,
        'safe_even': args.safe_even,
        'work_dir': args.work_dir or os.getcwd(),
        'dry_run': args.dry_run,
    }
    
    # 验证标签数量
    if combine_mode:
        expected_labels = 1 + len(args.pred) + len(args.pred_1fps)  # GT + 4fps pred + 1fps pred
        if len(args.labels) != expected_labels:
            print(f"[WARN] 标签数量 ({len(args.labels)}) 与预期 ({expected_labels}) 不匹配")
            print(f"      预期: GT + {len(args.pred)} 个 4fps 预测方法 + {len(args.pred_1fps)} 个 1fps 预测方法")
    else:
        expected_labels = 1 + len(args.pred)  # GT + pred数量
        if len(args.labels) != expected_labels:
            print(f"[WARN] 标签数量 ({len(args.labels)}) 与预期 ({expected_labels}) 不匹配")
            print(f"      预期: GT + {len(args.pred)} 个预测方法")
    
    # 检查输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 执行批量处理
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_tasks = len(ids)
    current_task = 0
    
    if combine_mode:
        print(f"\n开始批量处理 {len(ids)} 个ID（合并模式：4fps + 1fps）...")
    else:
        print(f"\n开始批量处理 {len(ids)} 个ID（仅 4fps）...")
    print(f"输出目录: {args.out_dir}")
    print(f"并行数: {args.max_workers}")
    print(f"跳过已存在: {args.skip_existing}")
    print(f"总任务数: {total_tasks}\n")
    
    for i, id_val in enumerate(ids, 1):
        current_task += 1
        
        # 检查是否已存在
        if args.skip_existing:
            video_path = os.path.join(args.out_dir, f"id_{id_val}", f"compare_id{id_val}.mp4")
            if os.path.exists(video_path):
                print(f"[SKIP] id_{id_val} 已存在，跳过 ({current_task}/{total_tasks})")
                skip_count += 1
                continue
        
        print(f"\n[{current_task}/{total_tasks}] 处理 id_{id_val}...")
        if combine_mode:
            success = run_make_compare_combined(id_val, args_dict, args.dry_run)
        else:
            # 如果没有 1fps 数据，只生成 4fps 视频（简化版本）
            cmd = [
                sys.executable,
                'make_compare.py',
                '--gt', args_dict['gt'],
                '--pred'] + args_dict['pred'] + [
                '--id', str(id_val),
                '--out_dir', args_dict['out_dir'],
                '--end', str(args_dict['end']),
                '--labels'] + args_dict['labels'] + [
                '--fps', str(int(args_dict['fps'])) if args_dict['fps'] == int(args_dict['fps']) else str(args_dict['fps']),
            ]
            if args_dict.get('safe_even', False):
                cmd.append('--safe_even')
            if args_dict.get('start', None):
                cmd.extend(['--start', str(args_dict['start'])])
            
            if args_dict.get('dry_run', False):
                print(f"[DRY RUN] 命令: {' '.join(cmd)}")
                success = True
            else:
                try:
                    result = subprocess.run(cmd, check=True, cwd=args_dict.get('work_dir', '.'))
                    success = True
                except Exception as e:
                    print(f"[ERROR] id_{id_val} 失败: {e}", file=sys.stderr)
                    success = False
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            if not args.dry_run:
                # 询问是否继续
                response = input(f"\nid_{id_val} 失败，是否继续？(y/n): ").strip().lower()
                if response != 'y':
                    print("用户取消")
                    return
    
    # 总结
    print(f"\n{'='*80}")
    print("批量处理完成！")
    if combine_mode:
        print(f"总计: {len(ids)} 个ID（每个包含 4fps + 1fps 合并视频）")
    else:
        print(f"总计: {len(ids)} 个ID（仅 4fps）")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

