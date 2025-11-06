# import pickle

# pkl_path = "/data1/tpz/nwm-main/data_splits/airvln_16/test/rollout_2d.pkl"
# target = "3JTPR5MT00AIKEFWO8VP6O4YVG35KU_processed"

# with open(pkl_path, "rb") as f:
#     data = pickle.load(f)

# # 遍历查找
# found_index = None
# for i, item in enumerate(data):
#     if isinstance(item, tuple) and len(item) > 0 and item[0] == target:
#         found_index = i
#         break

# if found_index is not None:
#     print(f"✅ 找到匹配元素：索引 = {found_index}")
#     print("对应内容:", data[found_index])
# else:
#     print("❌ 没找到匹配项:", target)
import pickle
import numpy as np

pkl_path = "/data1/tpz/nwm-main/data/airvln_16_long/random_walk_data_env_16_h35+10_random_traj_0_5_processed/traj_data.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("✅ 成功读取:", pkl_path)
print("数据类型:", type(data))

# 如果是 dict，就打印键
if isinstance(data, dict):
    # print("\n字典的键:")
    # for key in data.keys():
    #     print("  ", key)
    for key, value in list(data.items())[1:3]:
        print(f"value shape: {np.array(value).shape}")
        print(f"键: {key}\n对应的值:\n", value)
elif isinstance(data, list):
    print("\n列表长度:", len(data))
    print("第一个元素类型:", type(data[0]))
    for i, item in enumerate(data[:75]):
        print(f"\n第 {i} 个元素内容:\n", item)
    # print("第一个元素内容示例:\n", data[0])
else:
    print("\n内容示例:\n", data)
