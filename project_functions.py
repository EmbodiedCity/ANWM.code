import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def reproject_depth_to_other_pose(K, depth_map, rgb_img, pose_src, pose_dst):
    """
    输入：
        K: 内参矩阵 (3, 3)
        depth_map: 深度图 (H, W)
        rgb_img: RGB 图像 (H, W, 3)
        pose_src: 源相机位姿 (4, 4)
        pose_dst: 目标相机位姿 (4, 4)

    输出：
        points_3d_dst: (N, 3) 变换后的 3D 点
        colors: (N, 3) RGB 值
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth_map.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1)
    v = v.reshape(-1)
    z = depth_map.reshape(-1)
    valid = z > 0

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)  # shape: (N, 3)

    # 转换到齐次坐标 (N, 4)
    ones = np.ones((points_cam.shape[0], 1))
    points_cam_hom = np.concatenate([points_cam, ones], axis=1)

    # 从源位姿 -> 世界坐标
    points_world = (pose_src @ points_cam_hom.T).T  # shape: (N, 4)

    # 从世界坐标 -> 目标相机坐标
    points_dst_cam = (np.linalg.inv(pose_dst) @ points_world.T).T  # shape: (N, 4)

    # 去除齐次项
    points_3d_dst = points_dst_cam[:, :3]

    # 对应颜色
    colors = rgb_img.reshape(-1, 3)[valid]

    return points_3d_dst, colors

def project_to_2d_image(K, points_3d, colors, image_size):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = image_size

    # 取出 3D 点
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # 避免除以0
    z[z == 0] = 1e-6

    # 计算投影后的像素坐标
    u = (x * fx / z + cx).astype(np.int32)
    v = (y * fy / z + cy).astype(np.int32)

    # 过滤在图像边界外的点
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    colors_valid = colors[valid]

    # 创建空白图像
    image = np.zeros((H, W, 3), dtype=np.uint8)
    print(image.shape)
    print(colors_valid.shape)
    image[v, u] = colors_valid

    return image

def create_pose_matrix(translation, quaternion):
    rot = R.from_quat(quaternion)
    rot_mat = rot.as_matrix()  # 3x3

    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = translation
    return pose

def create_relative_pose(pose_src, delta_translation=(0, 0, 0.1), delta_angle_deg=5.0, axis='y'):
    """
    基于 pose_src 创建变换后的 pose_dst
    """
    # 旋转矩阵（绕 Y 轴旋转）
    if axis == 'y':
        rot = R.from_euler('y', delta_angle_deg, degrees=True).as_matrix()
    else:
        raise NotImplementedError("Only Y axis implemented")

    # 构造相对变换矩阵
    delta_pose = np.eye(4)
    delta_pose[:3, :3] = rot
    delta_pose[:3, 3] = np.array(delta_translation)

    # 将相对变换应用到原始位姿上
    pose_dst = pose_src @ delta_pose
    return pose_dst

def compute_intrinsic_matrix(width, height, fov_x_degree):
    fov_x = np.radians(fov_x_degree)
    fov_y = 2 * np.arctan((height*1.0 / width) * np.tan(fov_x / 2))

    intrinsic_parameters = {
        'width': width,
        'height': height,
        'fx': width / (2 * np.tan(fov_x / 2)), # 1.5 * width,
        'fy': height / (2 * np.tan(fov_y / 2)), # 1.5 * width,
        'cx': width / 2,
        'cy': height / 2,
    }

    fx = intrinsic_parameters['fx']
    fy = intrinsic_parameters['fy']
    cx = intrinsic_parameters['cx']
    cy = intrinsic_parameters['cy']

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K