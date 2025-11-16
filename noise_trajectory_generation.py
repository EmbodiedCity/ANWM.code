# A 3D trajectory genration module for a drone. Generates several trajectories by sampling from random positions near certain waypoints.
# Trajectory is a sequence of 3D points of (x,y,z,yaw).

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sdtw import SoftDTW
from scipy.spatial.distance import euclidean

action_space = {
    "forward": (5, 0, 0, 0),
    "backward": (-5, 0, 0, 0),
    "up": (0, 0, 2, 0), 
    "down": (0, 0, -2, 0),
    "left": (0, 0, 0, np.pi/12),  # Turn left by 15 degrees
    "right": (0, 0, 0, -np.pi/12)  # Turn right by 15 degrees
}

def random_trajectory_v2(gt_traj, pos_std = 1.0, yaw_std = 1.0, clip_tol=1.0, eps=1e-6):
    """
    在每步位移的单位方向上添加高斯噪声（沿位移方向），返回加噪轨迹
    
    参数:
        gt_traj: list of (x, y, z, yaw)
            原始无人机轨迹
        pos_std: float
            位置增量噪声标准差（米）
        yaw_std: float
            偏航角增量噪声标准差（弧度）
    
    返回:
        noise_traj: list of (x, y, z, yaw)
    """
    gt_traj = np.array(gt_traj, dtype=float)
    noise_traj = [gt_traj[0].copy()]  # 起点不加噪
    # step_len = 5.0  # 固定水平位移长度

    for i in range(1, len(gt_traj)):
        # # 新版本1 对yaw加噪声，位置按固定步长移动
        # delta = gt_traj[i] - gt_traj[i - 1]          # 真正的增量 (dx,dy,dz,dyaw)
        # prev = noise_traj[-1]
        # gt_yaw = gt_traj[i][3]
        # # 1️⃣ 在yaw上加高斯噪声
        # noisy_yaw = gt_yaw + np.random.normal(0.0, yaw_std)

        # pos_delta = delta[:3]
        # pos_norm = np.linalg.norm(pos_delta)
        # if pos_norm > eps:
        #     # 2️⃣ 保证水平位移长度固定为 step_len
        #     dx = step_len * np.cos(noisy_yaw)
        #     dy = step_len * np.sin(noisy_yaw)

        #     # 3️⃣ 垂直方向加独立噪声
        #     dz = (gt_traj[i][2] - gt_traj[i-1][2]) + np.random.normal(0.0, pos_std)
        #     # dz = gt_traj[i][2] - gt_traj[i-1][2]  # 保持垂直位移不变
        # else:
        #     dx, dy = 0.0, 0.0
        #     dz = 0.0
        # # 4️⃣ 更新位置
        # new_point = np.array([
        #     prev[0] + dx,
        #     prev[1] + dy,
        #     prev[2] + dz,
        #     noisy_yaw
        # ])
        # noise_traj.append(new_point)

        # # 旧版本2 对单位向量加噪声
        # delta = gt_traj[i] - gt_traj[i - 1]          # 真正的增量 (dx,dy,dz,dyaw)
        # delta_noisy = delta.copy()

        # # 位置部分 (x,y,z)
        # pos_delta = delta[:3]
        # pos_norm = np.linalg.norm(pos_delta)
        # if pos_norm > eps:
        #     u = pos_delta / pos_norm      # 单位方向向量 (3,)
        #     r = np.random.normal(0.0, pos_std)  # 标量噪声（可能为正或负）
        #     delta_noisy[:3] += u * r
        # # else: pos_delta 近似为 0，不在位置上加噪

        # # 偏航部分 (yaw) —— 1D 情况，用符号表示方向
        # yaw_delta = delta[3]
        # # if abs(yaw_delta) > eps:
        # #     sign = np.sign(yaw_delta)    # +1 或 -1
        # r_yaw = np.random.normal(0.0, yaw_std)
        # delta_noisy[3] += r_yaw
        # # else: yaw_delta 近似为 0，不在 yaw 上加噪

        # # 从上一步累积得到新的位置
        # new_point = noise_traj[-1] + delta_noisy
        # noise_traj.append(new_point)

        # 旧版本：对每个维度单独判断是否加噪声
        delta = gt_traj[i] - gt_traj[i - 1]
        delta_noisy = delta.copy()

        # 对每个维度单独判断是否加噪声
        for j in range(4):
            # if delta[j] != 0:  # 只对非零位移维度加噪声
            std = pos_std if j < 3 else yaw_std
            delta_noisy[j] += np.random.normal(0, std)

        # 计算在累积后的 yaw（以当前累积轨迹的最后yaw为基准 + yaw增量）
        prev_yaw = noise_traj[-1][3]
        new_yaw = prev_yaw + delta_noisy[3]

        # 目标的水平增量（基于 new_yaw）
        # target_dx = 5 * np.cos(new_yaw)
        # target_dy = 5 * np.sin(new_yaw)

        # 用 clip 限制 dx, dy 在 [target - tol, target + tol] 范围内
        # delta_noisy[0] = np.clip(delta_noisy[0], target_dx - clip_tol, target_dx + clip_tol)
        # delta_noisy[1] = np.clip(delta_noisy[1], target_dy - clip_tol, target_dy + clip_tol)

        pos_delta = delta[:3]
        pos_norm = np.linalg.norm(pos_delta)
        if pos_norm < eps: # 仅转向 没有位移
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
            delta_noisy[2] = 0.0
        if pos_delta[0] == 0 and pos_delta[1] == 0: # 仅垂直位移
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
        if pos_delta[2] == 0: # 仅水平位移
            delta_noisy[2] = 0.0
        # 累积得到下一点
        # if pos_std == 0 and yaw_std == 0:
        #     delta_noisy = delta  # 不加噪声
        new_point = noise_traj[-1] + delta_noisy
        noise_traj.append(new_point)
    
    # 确保返回的轨迹起点与(0,0,0,0)对齐
    noise_traj = np.array(noise_traj, dtype=float)
    start_offset = noise_traj[0, :].copy()  # 保存起点偏移
    noise_traj = noise_traj - start_offset  # 将整个轨迹平移到以(0,0,0,0)为起点
    noise_traj[0, :] = 0.0  # 确保起点精确为(0,0,0,0)
    
    return noise_traj.tolist()

def random_trajectory_v2_2d(gt_traj, pos_std = 1.0, yaw_std = 1.0, clip_tol=1.0, eps=1e-6):
    """
    在每步位移的单位方向上添加高斯噪声（沿位移方向），返回加噪轨迹
    
    参数:
        gt_traj: list of (x, y, z, yaw)
            原始无人机轨迹
        pos_std: float
            位置增量噪声标准差（米）
        yaw_std: float
            偏航角增量噪声标准差（弧度）
    
    返回:
        noise_traj: list of (x, y, z, yaw)
    """
    gt_traj = np.array(gt_traj, dtype=float)
    noise_traj = [gt_traj[0].copy()]  # 起点不加噪
    # step_len = 5.0  # 固定水平位移长度

    for i in range(1, len(gt_traj)):

        # 旧版本：对每个维度单独判断是否加噪声
        delta = gt_traj[i] - gt_traj[i - 1]
        delta_noisy = delta.copy()

        # 对每个维度单独判断是否加噪声
        for j in range(4):
            if j == 2:  # 垂直位移不加噪声
                continue
            # if delta[j] != 0:  # 只对非零位移维度加噪声
            std = pos_std if j < 3 else yaw_std
            delta_noisy[j] += np.random.normal(0, std)

        # 计算在累积后的 yaw（以当前累积轨迹的最后yaw为基准 + yaw增量）
        prev_yaw = noise_traj[-1][3]
        new_yaw = prev_yaw + delta_noisy[3]

        # 目标的水平增量（基于 new_yaw）
        # target_dx = 5 * np.cos(new_yaw)
        # target_dy = 5 * np.sin(new_yaw)

        # 用 clip 限制 dx, dy 在 [target - tol, target + tol] 范围内
        # delta_noisy[0] = np.clip(delta_noisy[0], target_dx - clip_tol, target_dx + clip_tol)
        # delta_noisy[1] = np.clip(delta_noisy[1], target_dy - clip_tol, target_dy + clip_tol)

        pos_delta = delta[:3]
        pos_norm = np.linalg.norm(pos_delta)
        if pos_norm < eps: # 仅转向 没有位移
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
            delta_noisy[2] = 0.0
        if pos_delta[0] == 0 and pos_delta[1] == 0: # 仅垂直位移
            delta_noisy[0] = 0.0
            delta_noisy[1] = 0.0
        if pos_delta[2] == 0: # 仅水平位移
            delta_noisy[2] = 0.0
        # 累积得到下一点
        # if pos_std == 0 and yaw_std == 0:
        #     delta_noisy = delta  # 不加噪声
        new_point = noise_traj[-1] + delta_noisy
        noise_traj.append(new_point)
    
    # 确保返回的轨迹起点与(0,0,0,0)对齐
    noise_traj = np.array(noise_traj, dtype=float)
    start_offset = noise_traj[0, :].copy()  # 保存起点偏移
    noise_traj = noise_traj - start_offset  # 将整个轨迹平移到以(0,0,0,0)为起点
    noise_traj[0, :] = 0.0  # 确保起点精确为(0,0,0,0)
    
    return noise_traj.tolist()


def random_trajectory(start, steps=6):
    trajectory = [start]
    current_pos = np.array(start[:4], dtype=float)  # (x, y, z, yaw) - 确保为浮点类型
    
    # 定义相反动作映射
    opposite_action = {
        "forward": "backward",
        "backward": "forward",
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    }

    last_action = None
    for _ in range(steps):
        last_pos = trajectory[-1]
        # 生成可选动作列表
        actions = list(action_space.keys())
        if last_action is not None:
            # 排除上一步的相反动作
            actions.remove(opposite_action[last_action])
            action_choice = np.random.choice(actions)
            action = action_space[action_choice]
            actions.append(opposite_action[last_action])
        else:
            action_choice = np.random.choice(actions)
            action = action_space[action_choice]
        if action_choice in ["left", "right", "up", "down"]:
            current_pos = last_pos + np.array(action)
        elif action_choice in ["forward", "backward"]:
            dx = action[0] * np.cos(last_pos[3])
            dy = action[0] * np.sin(last_pos[3])
            current_pos = last_pos + np.array([dx, dy, 0, 0])
        trajectory.append(current_pos)
        last_action = action_choice
    
    return trajectory

def trajectory_similarity(traj1, traj2):
    """
    计算两条轨迹之间的相似度，使用Soft-DTW方法。
    
    参数:
    traj1: list - 第一条轨迹，包含多个点 (x, y, z, yaw)
    traj2: list - 第二条轨迹，包含多个点 (x, y, z, yaw)
    返回:
    float - 相似度分数，范围在0到1之间，1表示完全相似，0表示完全不同
    """

    # 提取轨迹点的坐标
    traj1_points = np.array([[point[0], point[1], point[2]] for point in traj1])
    traj2_points = np.array([[point[0], point[1], point[2]] for point in traj2])

    # 计算距离矩阵
    D = np.zeros((len(traj1_points), len(traj2_points)))
    for i in range(len(traj1_points)):
        for j in range(len(traj2_points)):
            D[i, j] = np.linalg.norm(traj1_points[i] - traj2_points[j])

    # 计算Soft-DTW距离
    soft_dtw = SoftDTW(D, gamma=1.0)
    distance = soft_dtw.compute()

    max_distance = np.sqrt(len(traj1) + len(traj2))
    similarity_score = 1 - (distance / max_distance)
    return similarity_score

def trajectory_generation_random(GT_trajectory, candidate_number, dim=3):
    """
    生成多个轨迹的函数
    
    参数:
    start_pos: tuple - 起始坐标 (x, y, z, yaw)
    steps: int - 每条轨迹的步数
    candidate_number: int - 候选轨迹数量
    
    返回:
    list - 包含多个轨迹的列表，每个轨迹是一个包含坐标点的列表
    """
    candidate_trajectoires = []
    start_pos = GT_trajectory[0]  # Use the first point of GT trajectory as the start position
    steps = len(GT_trajectory) - 1  # Use the length of GT trajectory minus one for steps

    while len(candidate_trajectoires) < candidate_number:
        # Generate a random starting direction
        starting_yaw = np.random.uniform(-np.pi, np.pi)
        # trajectory = random_trajectory(
        #     (start_pos[0], start_pos[1], start_pos[2], starting_yaw),
        #     steps
        # )
        if dim == 3:
            trajectory = random_trajectory_v2(GT_trajectory, pos_std = 0.1, yaw_std = 0.1)
        elif dim == 2:
            trajectory = random_trajectory_v2_2d(GT_trajectory, pos_std = 0.1, yaw_std = 0.1)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        
        # Ensure the trajectory is valid (not too similar to GT)
        # Use Soft-DTW to estimate similarity between two trajectories
        if True or 0.5 < trajectory_similarity(trajectory, GT_trajectory) < 0.8:  # Adjust threshold as needed
            # print(f"similarity: {trajectory_similarity(trajectory, GT_trajectory)}")
            candidate_trajectoires.append(trajectory)
        # print(f"similarity: {trajectory_similarity(trajectory, GT_trajectory)}")

    return candidate_trajectoires

def plot_3d_trajectories(GT_trajectory, candidate_trajectoires, save_path="./plot", filename="trajectories_3d.png", show_plot=True):
    """
    绘制3D轨迹图的函数
    
    参数:
    candidate_trajectoires: list - 包含多个轨迹的列表
    save_path: str - 图片保存路径，默认为 "./plot"
    filename: str - 图片文件名，默认为 "trajectories_3d.png"
    show_plot: bool - 是否显示图片，默认为 True
    """
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'yellow', 'black']  # 扩展颜色列表
    

    # 绘制GT轨迹, 红色， 粗
    gt_xs, gt_ys, gt_zs = zip(*[(point[0], point[1], point[2]) for point in GT_trajectory])
    ax.plot(gt_xs, gt_ys, gt_zs, color='red', alpha=0.9, linewidth=3, label='GT Trajectory')
    ax.scatter(gt_xs[0], gt_ys[0], gt_zs[0], color='red', s=100, marker='o', edgecolors='black', linewidth=1, label='GT Start')
    ax.scatter(gt_xs[-1], gt_ys[-1], gt_zs[-1], color='red', s=80, marker='*', edgecolors='black', linewidth=1, label='GT End')

    for i, traj in enumerate(candidate_trajectoires):
        xs, ys, zs = zip(*[(point[0], point[1], point[2]) for point in traj])
        color = colors[i % len(colors)]
        
        # 绘制轨迹线
        ax.plot(xs, ys, zs, color=color, alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
        
        # 标记起点
        ax.scatter(xs[0], ys[0], zs[0], color=color, s=100, marker='o', edgecolors='black', linewidth=1)
        
        # 标记中间节点（除了起点和终点）
        if len(xs) > 2:
            ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], color=color, s=30, marker='o', alpha=0.8)
        
        # 标记终点
        ax.scatter(xs[-1], ys[-1], zs[-1], color=color, s=80, marker='s', edgecolors='black', linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories Visualization')
    
    # 只在轨迹数量较少时显示图例
    if len(candidate_trajectoires) <= 10:
        ax.legend()
    
    # 保存图片
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {full_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # Initiate
    origin_crd = (0.0, 0.0, 0.0, 0.0)  # (x, y, z, yaw)
    candidate_number = 5
    # step_number = 6

    candidate_trajectoires = []

    GT_traj = [
        (0.0, 0.0, 0.0, 1.57), (0.0, 5.0, 0.0, 1.57), (0.0, 10.0, 0.0, 1.57), (0.0, 10.0, 0.0, 1.83), (0.0, 10.0, 0.0, 2.09), (0.0, 10.0, 2.0, 2.09), (-2.5, 14.3, 2.0, 2.09), (-3, 18.6, 2.0, 2.09), (-3, 18.6, 4.0, 2.09) ]
    GT_traj_2 = [
        (0.0, 0.0, 0.0, -1.5708),           # start yaw = -pi/2 (facing -y)
        (0.0, 0.0, 2.0, -1.5708),          # up
        (0.0, 0.0, 4.0, -1.5708),          # up
        (0.0, -5.0, 4.0, -1.5708),         # forward (toward -y)
        (0.0, -10.0, 4.0, -1.5708),        # forward
        (0.0, -10.0, 2.0, -1.5708),        # down
        (0.0, -10.0, 0.0, -1.5708),        # down
        (0.0, -10.0, 0.0, -1.3090),        # left (yaw increases by pi/12)
        (1.2941, -14.8296, 0.0, -1.3090)   # forward (with new yaw)
    ]
    GT_traj_3 = [
        (0.0, 0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0, -0.2618),           # right turn
        (9.8296, -1.2941, 0.0, -0.2618),    # forward
        (9.8296, -1.2941, 0.0, -0.5236),    # right
        (7.8657, -5.0, 0.0, -0.5236),       # forward
        (7.8657, -5.0, 0.0, -0.7854),       # right
        (12.1958, -8.5355, 0.0, -0.7854)    # forward
    ]
    GT_traj_4 = [
        (0.0, 0.0, 0.0, 0.5236),            # start yaw = pi/6
        (4.3301, 2.5, 0.0, 0.5236),         # forward
        (4.3301, 2.5, 0.0, 0.7854),         # left
        (7.8657, 6.0355, 0.0, 0.7854),      # forward
        (7.8657, 6.0355, 0.0, 0.5236),      # right
        (12.1958, 8.5355, 0.0, 0.5236),     # forward
        (12.1958, 8.5355, 0.0, 0.7854),     # left
        (15.7313, 12.0711, 0.0, 0.7854)     # forward
    ]
    # Generate candidate trajectories
    candidate_trajectoires = trajectory_generation_random(
        GT_traj_3, 
        candidate_number=candidate_number,
        dim=2
    )
    print(f"GT_traj: {GT_traj_3}")
    for i, traj in enumerate(candidate_trajectoires):
        print(f"Candidate Traj {i+1}: {traj}")
    # Plot the trajectories
    plot_3d_trajectories(
        GT_traj_3,
        candidate_trajectoires, 
        save_path="./plot", 
        filename=f"trajectories_GT_with_noise_{datetime.datetime.now():%m-%d_%H-%M-%S}.png"
    )



    



