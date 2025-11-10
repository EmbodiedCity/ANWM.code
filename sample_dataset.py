import pickle
import os
import random

# 设置随机种子
random.seed(1120)

# 输入文件路径
input_pkl_path = "/data1/tpz/nwm-main/data_splits/airvln_16/test/dataset_dist_8_to_8_n16_len_traj_pred_8.pkl"
# 输出文件路径
output_pkl_path = "/data1/tpz/nwm-main/data_splits/airvln_16/test/navigation_eval_16.pkl"

print(f"正在读取文件: {input_pkl_path}")

# 读取原始数据
with open(input_pkl_path, "rb") as f:
    data = pickle.load(f)[0]

print(f"✅ 成功读取文件")
print(f"数据类型: {type(data)}")
print(f"原始数据包含 {len(data)} 条记录")

# 从list中随机采样1000条
import random
random.seed(1120)
sampled_data = random.sample(data, k=100)
print(f"采样后包含 {len(sampled_data)} 条记录")

# 保存采样后的数据
os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
with open(output_pkl_path, "wb") as f:
    pickle.dump(sampled_data, f)

print(f"\n✅ 已保存采样数据到: {output_pkl_path}")
print(f"使用随机种子: 1120")

