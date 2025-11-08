#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新筛选效果最好的cases，过滤掉量化指标异常好的（可能对着水拍的）
"""
import json
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple

def load_cases_json(json_path: str) -> Dict:
    """加载 cases.json 文件"""
    if not os.path.exists(json_path):
        print(f"[WARN] 文件不存在: {json_path}")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_trajectory_metrics(cases_data: Dict, metric_name: str = 'lpips') -> Dict[str, float]:
    """计算每条轨迹的指标"""
    trajectory_metrics = {}
    
    for traj_id, time_data in cases_data.items():
        all_metrics = []
        for time_key, frames in time_data.items():
            if isinstance(frames, list) and len(frames) > 0:
                for frame_data in frames:
                    if metric_name in frame_data:
                        all_metrics.append(frame_data[metric_name])
        
        if all_metrics:
            trajectory_metrics[traj_id] = np.mean(all_metrics)
        else:
            trajectory_metrics[traj_id] = None
    
    return trajectory_metrics

def filter_outliers(values: List[float], method: str = 'iqr', factor: float = 1.5) -> Tuple[float, float]:
    """
    计算异常值阈值
    
    Args:
        values: 数值列表
        method: 方法 ('iqr' 或 'std')
        factor: 因子（默认1.5用于IQR，2用于std）
    
    Returns:
        (lower_bound, upper_bound) - 正常值的范围
    """
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (0, float('inf'))
    
    values = np.array(values)
    
    if method == 'iqr':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
    elif method == 'std':
        mean = np.mean(values)
        std = np.std(values)
        lower = mean - factor * std
        upper = mean + factor * std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return (lower, upper)

def find_best_cases_filtered(cases_files: List[str], 
                             metric_name: str = 'lpips',
                             method: str = 'average',
                             top_k: int = 100,
                             filter_outliers_flag: bool = True,
                             min_lpips: float = None,
                             max_lpips: float = None,
                             min_dreamsim: float = None,
                             max_dreamsim: float = None) -> List[Tuple[str, float, float, str]]:
    """
    找出效果最好的 cases，过滤异常值
    
    Args:
        cases_files: cases.json 文件路径列表
        metric_name: 使用的指标名称 ('lpips' 或 'dreamsim')
        method: 计算方法 ('average' 或 'best')
        top_k: 返回前 k 个最好的 cases
        filter_outliers_flag: 是否过滤异常值
        min_lpips, max_lpips: LPIPS 的上下限
        min_dreamsim, max_dreamsim: DreamSim 的上下限
    
    Returns:
        List of (traj_id, combined_score, lpips_avg, dreamsim_avg, methods_info)
    """
    all_results = []
    
    # 加载所有数据
    for cases_file in cases_files:
        method_name = os.path.basename(os.path.dirname(cases_file))
        cases_data = load_cases_json(cases_file)
        
        if not cases_data:
            continue
        
        lpips_metrics = calculate_trajectory_metrics(cases_data, 'lpips')
        dreamsim_metrics = calculate_trajectory_metrics(cases_data, 'dreamsim')
        
        for traj_id in lpips_metrics.keys():
            if traj_id in lpips_metrics and traj_id in dreamsim_metrics:
                lpips_val = lpips_metrics[traj_id]
                dreamsim_val = dreamsim_metrics[traj_id]
                if lpips_val is not None and dreamsim_val is not None:
                    all_results.append((traj_id, lpips_val, dreamsim_val, method_name))
    
    # 收集所有指标用于计算阈值
    all_lpips = [r[1] for r in all_results]
    all_dreamsim = [r[2] for r in all_results]
    
    # 计算异常值阈值（只过滤下限，因为指标越小越好，异常好的是异常值）
    if filter_outliers_flag:
        lpips_lower, lpips_upper = filter_outliers(all_lpips, method='iqr', factor=1.5)
        dreamsim_lower, dreamsim_upper = filter_outliers(all_dreamsim, method='iqr', factor=1.5)
        
        # 对于指标越小越好的情况，异常好的是低于下界的值
        # 但我们需要保留好的，所以设置一个合理的最小值阈值
        # 使用更保守的方法：如果指标异常低（太好），可能是异常情况
        lpips_percentile_5 = np.percentile(all_lpips, 5)  # 前5%可能异常好
        dreamsim_percentile_5 = np.percentile(all_dreamsim, 5)
        
        print(f"[INFO] LPIPS 统计: min={np.min(all_lpips):.4f}, 5%={lpips_percentile_5:.4f}, median={np.median(all_lpips):.4f}, max={np.max(all_lpips):.4f}")
        print(f"[INFO] DreamSim 统计: min={np.min(all_dreamsim):.4f}, 5%={dreamsim_percentile_5:.4f}, median={np.median(all_dreamsim):.4f}, max={np.max(all_dreamsim):.4f}")
    else:
        lpips_percentile_5 = None
        dreamsim_percentile_5 = None
    
    # 应用用户指定的阈值
    if min_lpips is not None:
        lpips_threshold = min_lpips
    elif filter_outliers_flag and lpips_percentile_5 is not None:
        lpips_threshold = lpips_percentile_5
    else:
        lpips_threshold = None
    
    if min_dreamsim is not None:
        dreamsim_threshold = min_dreamsim
    elif filter_outliers_flag and dreamsim_percentile_5 is not None:
        dreamsim_threshold = dreamsim_percentile_5
    else:
        dreamsim_threshold = None
    
    if max_lpips is not None:
        lpips_max_threshold = max_lpips
    else:
        lpips_max_threshold = None
    
    if max_dreamsim is not None:
        dreamsim_max_threshold = max_dreamsim
    else:
        dreamsim_max_threshold = None
    
    # 过滤结果
    filtered_results = []
    for traj_id, lpips_val, dreamsim_val, method_name in all_results:
        # 应用过滤条件
        if lpips_threshold is not None and lpips_val < lpips_threshold:
            continue  # 跳过异常好的 LPIPS
        if dreamsim_threshold is not None and dreamsim_val < dreamsim_threshold:
            continue  # 跳过异常好的 DreamSim
        if lpips_max_threshold is not None and lpips_val > lpips_max_threshold:
            continue  # 跳过太差的
        if dreamsim_max_threshold is not None and dreamsim_val > dreamsim_max_threshold:
            continue  # 跳过太差的
        
        filtered_results.append((traj_id, lpips_val, dreamsim_val, method_name))
    
    print(f"[INFO] 过滤前: {len(all_results)} 个cases")
    print(f"[INFO] 过滤后: {len(filtered_results)} 个cases")
    if lpips_threshold is not None:
        print(f"[INFO] 过滤条件: LPIPS >= {lpips_threshold:.4f}")
    if dreamsim_threshold is not None:
        print(f"[INFO] 过滤条件: DreamSim >= {dreamsim_threshold:.4f}")
    
    # 按 traj_id 分组，计算综合指标
    traj_groups = {}
    for traj_id, lpips_val, dreamsim_val, method_name in filtered_results:
        if traj_id not in traj_groups:
            traj_groups[traj_id] = []
        traj_groups[traj_id].append((lpips_val, dreamsim_val, method_name))
    
    # 计算每个轨迹的综合指标
    traj_scores = {}
    for traj_id, metrics_list in traj_groups.items():
        lpips_values = [m[0] for m in metrics_list]
        dreamsim_values = [m[1] for m in metrics_list]
        
        if method == 'average':
            lpips_avg = np.mean(lpips_values)
            dreamsim_avg = np.mean(dreamsim_values)
            combined_score = (lpips_avg + dreamsim_avg) / 2.0
        elif method == 'best':
            lpips_avg = np.min(lpips_values)
            dreamsim_avg = np.min(dreamsim_values)
            combined_score = (lpips_avg + dreamsim_avg) / 2.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        methods_info = ', '.join([f"{m[2]}: lpips={m[0]:.4f}, dreamsim={m[1]:.4f}" for m in metrics_list])
        traj_scores[traj_id] = (combined_score, lpips_avg, dreamsim_avg, methods_info)
    
    # 排序并返回 top_k
    sorted_trajs = sorted(traj_scores.items(), key=lambda x: x[1][0])
    
    results = []
    for traj_id, (combined_score, lpips_avg, dreamsim_avg, methods_info) in sorted_trajs[:top_k]:
        results.append((traj_id, combined_score, lpips_avg, dreamsim_avg, methods_info))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='重新筛选效果最好的cases，过滤异常值')
    parser.add_argument('--cases_files', nargs='+', required=True,
                       help='cases.json 文件路径列表')
    parser.add_argument('--metric', type=str, default='lpips',
                       choices=['lpips', 'dreamsim'],
                       help='主要使用的指标 (默认: lpips)')
    parser.add_argument('--method', type=str, default='average',
                       choices=['average', 'best'],
                       help='计算方法: average(平均), best(最好)')
    parser.add_argument('--top_k', type=int, default=100,
                       help='返回前 k 个最好的 cases (默认: 100)')
    parser.add_argument('--filter_outliers', action='store_true',
                       help='过滤异常值（异常好的指标）')
    parser.add_argument('--min_lpips', type=float, default=None,
                       help='LPIPS 最小值阈值（过滤异常好的，默认自动计算）')
    parser.add_argument('--max_lpips', type=float, default=None,
                       help='LPIPS 最大值阈值（过滤太差的）')
    parser.add_argument('--min_dreamsim', type=float, default=None,
                       help='DreamSim 最小值阈值（过滤异常好的，默认自动计算）')
    parser.add_argument('--max_dreamsim', type=float, default=None,
                       help='DreamSim 最大值阈值（过滤太差的）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (可选)')
    parser.add_argument('--output_ids', type=str, default=None,
                       help='输出ID列表文件路径 (可选)')
    
    args = parser.parse_args()
    
    print(f"[INFO] 分析 {len(args.cases_files)} 个 cases.json 文件")
    print(f"[INFO] 使用指标: {args.metric}")
    print(f"[INFO] 计算方法: {args.method}")
    print(f"[INFO] 过滤异常值: {args.filter_outliers}")
    print(f"[INFO] 返回前 {args.top_k} 个最好的 cases\n")
    
    results = find_best_cases_filtered(
        args.cases_files, 
        args.metric, 
        args.method, 
        args.top_k,
        args.filter_outliers,
        args.min_lpips,
        args.max_lpips,
        args.min_dreamsim,
        args.max_dreamsim
    )
    
    print(f"\n{'='*100}")
    print(f"效果最好的 {len(results)} 个 cases (综合分数 = (lpips + dreamsim) / 2, 越小越好):")
    print(f"{'='*100}\n")
    
    output_lines = []
    id_lines = []
    for i, (traj_id, combined_score, lpips_avg, dreamsim_avg, methods_info) in enumerate(results, 1):
        if i <= 20:  # 只打印前20个的详细信息
            print(f"{i}. {traj_id}:")
            print(f"   综合分数: {combined_score:.6f}")
            print(f"   LPIPS 平均: {lpips_avg:.6f}")
            print(f"   DreamSim 平均: {dreamsim_avg:.6f}")
            print(f"   方法详情: {methods_info}")
            print()
        
        output_lines.append(f"{i}\t{traj_id}\t{combined_score:.6f}\t{lpips_avg:.6f}\t{dreamsim_avg:.6f}\t{methods_info}")
        id_lines.append(traj_id)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"# 效果最好的 {len(results)} 个 cases（已过滤异常值）\n")
            f.write(f"# 格式: 排名\ttraj_id\tcombined_score\tlpips_avg\tdreamsim_avg\tmethods_info\n")
            for line in output_lines:
                f.write(line + '\n')
        print(f"[OK] 结果已保存到: {args.output}")
    
    if args.output_ids:
        with open(args.output_ids, 'w', encoding='utf-8') as f:
            for traj_id in id_lines:
                f.write(traj_id + '\n')
        print(f"[OK] ID列表已保存到: {args.output_ids}")
    elif args.output:
        # 自动生成ID列表文件
        output_ids = args.output.replace('.txt', '_ids.txt')
        with open(output_ids, 'w', encoding='utf-8') as f:
            for traj_id in id_lines:
                f.write(traj_id + '\n')
        print(f"[OK] ID列表已保存到: {output_ids}")
    
    # 输出最好的一个
    if results:
        best_traj_id = results[0][0]
        best_combined = results[0][1]
        best_lpips = results[0][2]
        best_dreamsim = results[0][3]
        print(f"\n{'='*100}")
        print(f"🏆 效果最好的 case: {best_traj_id}")
        print(f"   综合分数: {best_combined:.6f}")
        print(f"   LPIPS: {best_lpips:.6f}")
        print(f"   DreamSim: {best_dreamsim:.6f}")
        print(f"{'='*100}")

if __name__ == '__main__':
    main()

