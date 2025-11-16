import pickle
import numpy as np

# 读取pkl文件
pkl_path = 'data_splits/airvln_16/test/airvln_16_3_trajectories_long.pkl'
with open(pkl_path, 'rb') as f:
    trajectories_data = pickle.load(f)

# 存储每条轨迹的起点到终点距离
distances = []

# 遍历100条轨迹
for traj_id in range(100):
    if traj_id not in trajectories_data:
        continue
    
    # 获取轨迹数据，形状为(3, 9, 4)
    # 3条候选轨迹，每条9个点，每个点4个值(x, y, z, angle)
    traj_candidates = trajectories_data[traj_id]  # (3, 9, 4)
    
    # 取第一条候选轨迹（索引0）
    traj = traj_candidates[0]  # (9, 4)
    
    # 提取起点和终点坐标（前3个值是x, y, z坐标）
    start_point = traj[0, :3]   # 起点 (x, y, z)
    end_point = traj[-1, :3]    # 终点 (x, y, z)
    
    # 计算欧氏距离
    distance = np.linalg.norm(end_point - start_point)
    distances.append(distance)

# 计算平均距离
if len(distances) > 0:
    avg_distance = np.mean(distances)
    print(f"共计算了 {len(distances)} 条轨迹")
    print(f"起点到终点的平均距离: {avg_distance:.2f}")
else:
    print("未找到有效轨迹数据")

