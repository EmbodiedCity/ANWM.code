# A 3D trajectory genration module for a drone. Generates several trajectories by sampling from random positions near certain waypoints.
# Trajectory is a sequence of 3D points of (x,y,z,yaw).




import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def sample_trajectory_from_distribution(start, mu, sigma):
    """
    从给定分布中采样轨迹。
    每步采样 [distance, delta_yaw, vertical_flag] 来控制移动。
    
    参数:
    - start: 起始点 (x, y, z, yaw)
    - mu: Tensor[steps, 3]，每步的动作均值（distance, delta_yaw, is_vertical）
    - sigma: Tensor[steps, 3]，每步的动作方差

    返回:
    - trajectory: List of (x, y, z, yaw)
    """
    assert mu.shape == sigma.shape, "mu and sigma shape mismatch"
    steps = mu.shape[0]
    trajectory = [start]
    current_pos = np.array(start[:3], dtype=float)
    current_yaw = start[3]

    for step in range(steps):
        # 从高斯分布采样动作参数
        action = torch.randn(3) * sigma[step] + mu[step]
        distance = action[0].item()
        delta_yaw = np.clip(action[1].item(), -np.pi/2, np.pi/2)
        is_vertical = torch.sigmoid(action[2]).item() > 0.5

        current_yaw = (current_yaw + delta_yaw) % (2 * np.pi)
        dx = dy = dz = 0

        if is_vertical:
            dz = distance * np.random.choice([-1, 1])
        else:
            dx = distance * np.cos(current_yaw)
            dy = distance * np.sin(current_yaw)

        current_pos += np.array([dx, dy, dz])
        trajectory.append((current_pos[0], current_pos[1], current_pos[2], current_yaw))

    return trajectory

def random_trajectory(start, steps=6):
    trajectory = [start]
    current_position = np.array(start[:3], dtype=float)  # (x, y, z) - 确保为浮点类型
    
    for _ in range(steps):
        dx = dy = dz = 0  # Initialize movement components

        is_vertical = np.random.choice([True, False], p=[0.5, 0.5])  # 30% vertical movement
        delta_yaw = np.random.normal(0, np.pi/4)  # Normal distribution for yaw change (mean=0, std=π/4)
        delta_yaw = np.clip(delta_yaw, -np.pi/2, np.pi/2) # restrict yaw to [-π/2, π/2]
        current_yaw = (start[3] + delta_yaw) % (2 * np.pi)  # Update yaw
        distance = np.random.uniform(2, 4)  # Random distance
        if is_vertical:
            # Vertical movement
            dz = distance * np.random.choice([-1, 1])
        else:
            # Horizontal movement
            dx = distance * np.cos(current_yaw)
            dy = distance * np.sin(current_yaw)
        
        current_position += np.array([dx, dy, dz])
        trajectory.append((current_position[0], current_position[1], current_position[2], current_yaw))
    
    return trajectory

def trajectory_generation(start_coord, steps=6, candidate_number=20):
    """
    生成多个轨迹的函数
    
    参数:
    start_coord: tuple - 起始坐标 (x, y, z, yaw)
    steps: int - 每条轨迹的步数，默认6步
    candidate_number: int - 候选轨迹数量，默认20条
    
    返回:
    list - 包含多个轨迹的列表，每个轨迹是一个包含坐标点的列表
    """
    candidate_trajectoires = []
    
    while len(candidate_trajectoires) < candidate_number:
        # Generate a random starting direction
        starting_yaw = np.random.uniform(-np.pi, np.pi)
        trajectory = random_trajectory(
            (start_coord[0], start_coord[1], start_coord[2], starting_yaw),
            steps
        )
        candidate_trajectoires.append(trajectory)
    
    return candidate_trajectoires

def plot_3d_trajectories(candidate_trajectoires, save_path="./plot", filename="trajectories_3d.png", show_plot=True):
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
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta', 'yellow', 'black']  # 扩展颜色列表
    
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

    # Generate candidate trajectories
    candidate_trajectoires = trajectory_generation(
        origin_crd, 
        steps=step_number, 
        candidate_number=candidate_number
    )
    
    # Plot the trajectories
    plot_3d_trajectories(
        candidate_trajectoires, 
        save_path="./plot", 
        filename="drone_trajectories.png"
    )


    



