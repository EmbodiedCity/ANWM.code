# A 3D trajectory genration module for a drone. Generates several trajectories by sampling from random positions near certain waypoints.
# Trajectory is a sequence of 3D points of (x,y,z,yaw).





import os
import numpy as np
import matplotlib.pyplot as plt
from soft_dtw import SoftDTW
from scipy.spatial.distance import euclidean

action_space = {
    "forward": (5, 0, 0, 0),
    "backward": (-5, 0, 0, 0),
    "up": (0, 0, 2, 0), 
    "down": (0, 0, -2, 0),
    "left": (0, 0, 0, np.pi/12),  # Turn left by 15 degrees
    "right": (0, 0, 0, -np.pi/12)  # Turn right by 15 degrees
}

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

def trajectory_generation_random(GT_trajectory, candidate_number=10):
    """
    生成多个轨迹的函数
    
    参数:
    start_pos: tuple - 起始坐标 (x, y, z, yaw)
    steps: int - 每条轨迹的步数，默认6步
    candidate_number: int - 候选轨迹数量，默认20条
    
    返回:
    list - 包含多个轨迹的列表，每个轨迹是一个包含坐标点的列表
    """
    candidate_trajectoires = []
    start_pos = GT_trajectory[0]  # Use the first point of GT trajectory as the start position
    steps = len(GT_trajectory) - 1  # Use the length of GT trajectory minus one for steps

    while len(candidate_trajectoires) < candidate_number:
        # Generate a random starting direction
        starting_yaw = np.random.uniform(-np.pi, np.pi)
        trajectory = random_trajectory(
            (start_pos[0], start_pos[1], start_pos[2], starting_yaw),
            steps
        )
        # Ensure the trajectory is valid (not too similar to GT)
        # Use Soft-DTW to estimate similarity between two trajectories
        if trajectory_similarity(trajectory, GT_trajectory) < 0.85:  # Adjust threshold as needed
            candidate_trajectoires.append(trajectory)

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
    candidate_number = 20
    step_number = 6

    candidate_trajectoires = []

    GT_traj = [
        (0.0, 0.0, 0.0, 1.57), (0.0, 5.0, 0.0, 1.57), (0.0, 10.0, 0.0, 1.57), (0.0, 10.0, 0.0, 1.83), (0.0, 10.0, 0.0, 2.09), (0.0, 10.0, 2.0, 2.09), (-2.5, 14.3, 2.0, 2.09), (-3, 18.6, 2.0, 2.09), (-3, 18.6, 4.0, 2.09) ]

    # Generate candidate trajectories
    candidate_trajectoires = trajectory_generation_random(
        GT_traj, 
        candidate_number=candidate_number
    )
    
    # Plot the trajectories
    plot_3d_trajectories(
        GT_traj,
        candidate_trajectoires, 
        save_path="./plot", 
        filename="trajectories_GT_with_noise.png"
    )



    



