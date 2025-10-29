import os
import cv2
from skimage.metrics import structural_similarity as ssim

# ===== 路径设置 =====
root_dir = "/data0/tpz/zengx/YUME/outputs/gen/airvln_16_2d/rollout_1fps"
target_path = "/data1/tpz/nwm-main/results/1015/nwm_cdit_airvln_16_latest/airvln_16/rollout_1fps/id_1/0.png"

# ===== 读取并预处理目标图像 =====
target = cv2.imread(target_path)
if target is None:
    raise FileNotFoundError(f"无法读取目标图片: {target_path}")
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target_gray = cv2.resize(target_gray, (256, 256))

# ===== 遍历所有 scene/**/id_*/.jpg =====
results = []
for scene in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_path):
        continue

    for id_folder in os.listdir(scene_path):
        id_path = os.path.join(scene_path, id_folder)
        if not os.path.isdir(id_path):
            continue

        for fn in os.listdir(id_path):
            if fn.lower().endswith(".jpg"):
                full_path = os.path.join(id_path, fn)
                img = cv2.imread(full_path)
                if img is None:
                    continue
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.resize(img_gray, (256, 256))

                # 计算结构相似度 SSIM
                score = ssim(target_gray, img_gray)
                results.append((score, full_path))

# ===== 选出 top3 =====
results.sort(reverse=True, key=lambda x: x[0])
top3 = results[:3]

# ===== 打印结果 =====
print("\nTop 3 最相似的图片:")
for i, (score, path) in enumerate(top3, start=1):
    print(f"{i}. {path}  (SSIM={score:.4f})")
