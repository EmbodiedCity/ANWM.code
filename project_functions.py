import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class CoordsConverter:
    def __init__(self):
        pass
    
    def coords_img2cam(self, p, dep):
        pass
    
    def coords_cam2ego(self, P_cam):
        pass
    
    def coords_ego2world(self, P_ego, ego_pose):
        pass
    
    def coords_world2cam(self, P_w, ego_pose):
        pass


class AirsimCoordsConverter(CoordsConverter):
    def __init__(self):
        super().__init__()

        w, h = 512, 512
        fov = np.pi / 2
        fx = 0.5 * w / np.tan(fov / 2)
        fy = fx

        self.K = np.array(
            [[fx, 0, w / 2],
             [0, fy, h / 2],
             [0, 0, 1]]
        )
        self.T_ec = np.array(
            [
                [0.0, 0, 1, 0],
                [1.0, 0, 0, 0],
                [0.0, 1, 0, 0],
                [0.0, 0, 0, 1]
            ]
        )

    def coords_img2cam(self, p, dep):
        # p: [u, v, 1], dep: depth of p
        K_inv = np.linalg.inv(self.K)
        P_cam = K_inv.dot(p * dep)

        return P_cam

    def coords_cam2ego(self, P_cam):
        # p: [x, y, z, 1]
        P_ego = self.T_ec.dot(P_cam)
        return P_ego

    def coords_ego2world(self, P_ego, ego_pose):
        # P_ego: [x, y, z, 1], ego_pose: [x, y, z, p, r, y]
        rot_quat = airsim.to_quaternion(*ego_pose[3:])
        rot_mat = R.from_quat([rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = shift_mat

        P_world = trans_mat.dot(P_ego)

        return P_world

    def coords_world2cam(self, P_w, ego_pose):
        # P_ego: [x, y, z, 1], ego_pose: [x, y, z, p, r, y]
        rot_quat = airsim.to_quaternion(*ego_pose[3:])
        rot_mat = R.from_quat([rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_wc = T_we.dot(self.T_ec)
        T_cw = np.linalg.inv(T_wc)

        P_cam = T_cw.dot(P_w)

        return P_cam

    def coords_cam2img(self, P_cam):
        # P_cam: [X, Y, Z, 1]
        p_img = self.K.dot(P_cam[:3]) / P_cam[2]
        
        return p_img
    
    def trans_world2cam(self, cam_pose):
        rot_quat = airsim.to_quaternion(*ego_pose[3:])
        
        rot_mat = R.from_quat([rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_wc = T_we.dot(self.T_ec)
        return T_wc

    def trans_cam2world(self, cam_pose):
        rot_quat = airsim.to_quaternion(*ego_pose[3:])
        
        rot_mat = R.from_quat([rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_wc = T_we.dot(self.T_ec)
        T_cw = np.linalg.inv(T_wc)
        
        return T_cw


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

def reproject_depth_to_other_pose_2seq(K, depth_maps, rgb_imgs, poses_src, poses_dst):
    """
    批量处理多个视角的重投影
    
    参数:
        K: (3, 3)
        depth_maps: (H, W)
        rgb_imgs: (H, W, 3)
        poses_src: (4, 4)
        poses_dst: (goal_time_seq_len, 4, 4)
    """
    B = poses_dst.shape[0]
    points_3d_all = []
    colors_all = []

    for i in range(B):
        points_3d, colors = reproject_depth_to_other_pose(
            K,
            depth_maps,
            rgb_imgs,
            poses_src,
            poses_dst[i]
        )
        points_3d_all.append(points_3d)
        colors_all.append(colors)

    return points_3d_all, colors_all

def project_to_2d_image(K, points_3d, colors, image_size):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = image_size

    # 取出 3D 点
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # 下限限制z
    z = np.maximum(z, 1e-6) 

    # 计算投影后的像素坐标
    u = (x * fx / z + cx)
    v = (y * fy / z + cy)
    # print(u, v)

    # 过滤在图像边界外的点
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    colors_valid = colors[valid]

    # 创建空白图像
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[v, u] = colors_valid

    return image

def project_to_2d_image_2seq(K, points_3d, colors, image_size):
    B = len(points_3d)
    images = []

    for i in range(B):
        image = project_to_2d_image(K, points_3d[i], colors[i], image_size)
        images.append(image)

    return np.stack(images, axis=0)  # shape: (B, H, W, 3)

def euler_to_quaternion(yaw, pitch, roll):
    r = R.from_euler('zyx', [yaw, pitch, roll])
    return r.as_quat()

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

def resize_image_half(rgb_img):
    original_height, original_width = rgb_img.shape[:2]
    resized_width = original_width // 2
    resized_height = original_height // 2
    resized_img = cv2.resize(rgb_img, (resized_width, resized_height))
    return resized_img