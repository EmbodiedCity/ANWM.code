import cv2
import math
from scipy.spatial.transform import Rotation as R
import numpy as np


class Quaternionr:
    w_val = 0.0
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val


def to_quaternion(pitch, roll, yaw):
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    q = Quaternionr()
    q.w_val = t0 * t2 * t4 + t1 * t3 * t5  # w
    q.x_val = t0 * t3 * t4 - t1 * t2 * t5  # x
    q.y_val = t0 * t2 * t5 + t1 * t3 * t4  # y
    q.z_val = t1 * t2 * t4 - t0 * t3 * t5  # z
    return q


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

        self.K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
        self.T_ec = np.array(
            [[0.0, 0, 1, 0], [1.0, 0, 0, 0], [0.0, 1, 0, 0], [0.0, 0, 0, 1]]
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
        rot_quat = to_quaternion(*ego_pose[3:])
        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = shift_mat

        P_world = trans_mat.dot(P_ego)

        return P_world

    def coords_world2cam(self, P_w, ego_pose):
        # P_ego: [x, y, z, 1], ego_pose: [x, y, z, p, r, y]
        rot_quat = to_quaternion(*ego_pose[3:])
        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
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

    def trans_world2cam(self, ego_pose):
        rot_quat = to_quaternion(*ego_pose[3:])

        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_wc = T_we.dot(self.T_ec)
        return T_wc

    def trans_cam2world(self, ego_pose):
        rot_quat = to_quaternion(*ego_pose[3:])

        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_wc = T_we.dot(self.T_ec)
        T_cw = np.linalg.inv(T_wc)

        return T_cw

    def trans_world2ego(self, ego_pose):
        rot_quat = to_quaternion(*ego_pose[3:])

        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        return T_we

    def trans_ego2world(self, ego_pose):
        rot_quat = to_quaternion(*ego_pose[3:])

        rot_mat = R.from_quat(
            [rot_quat.x_val, rot_quat.y_val, rot_quat.z_val, rot_quat.w_val]
        ).as_matrix()
        shift_mat = np.array(ego_pose[:3])

        T_we = np.eye(4)
        T_we[:3, :3] = rot_mat
        T_we[:3, 3] = shift_mat

        T_ew = np.linalg.inv(T_we)
        return T_ew


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
    # # 打印旋转矩阵
    # print("[DEBUG] pose_src rotation matrix:", pose_src)
    # print("[DEBUG] pose_dst rotation matrix:", pose_dst)

    # 提取并打印源和目标的 yaw 角度差异
    # euler_src = R.from_matrix(pose_src[:3, :3]).as_euler('zyx', degrees=True)
    # euler_dst = R.from_matrix(pose_dst[:3, :3]).as_euler('zyx', degrees=True)
    # print("[DEBUG] pose_src euler (yaw, pitch, roll):", euler_src)
    # print("[DEBUG] pose_dst euler (yaw, pitch, roll):", euler_dst)
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
            K, depth_maps, rgb_imgs, poses_src, poses_dst[i]
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
    u = x * fx / z + cx
    v = y * fy / z + cy
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
    r = R.from_euler("zyx", [yaw, pitch, roll])
    return r.as_quat()


def create_pose_matrix(translation, quaternion):
    rot = R.from_quat(quaternion)
    rot_mat = rot.as_matrix()  # 3x3

    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = translation
    return pose


def create_relative_pose(
    pose_src, delta_translation=(0, 0, 0.1), delta_angle_deg=5.0, axis="y"
):
    """
    基于 pose_src 创建变换后的 pose_dst
    """
    # 旋转矩阵（绕 Y 轴旋转）
    if axis == "y":
        rot = R.from_euler("y", delta_angle_deg, degrees=True).as_matrix()
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


# seq2seq
def project_to_2d_image_seq2(K, points_cam, colors, image_size):
    """
    将目标相机坐标系下的点云 (N,3) 用 z-buffer 投影到一张 (H,W,3) 图像
    K: (3,3), points_cam: (N,3) [X,Y,Z], colors: (N,3) uint8, image_size: (H,W)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = image_size

    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    front = Z > 1e-6
    X, Y, Z, colors = X[front], Y[front], Z[front], colors[front]

    u = X * fx / Z + cx
    v = Y * fy / Z + cy
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img].astype(np.int32)
    v = v[in_img].astype(np.int32)
    z = Z[in_img]
    colors = colors[in_img].astype(np.uint8)

    # z-buffer：同像素保留最近深度
    lin = v * W + u
    order = np.argsort(z)  # 近 -> 远
    lin_sorted = lin[order]
    _, keep = np.unique(lin_sorted, return_index=True)
    sel = order[keep]

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[v[sel], u[sel]] = colors[sel]
    return img


def reproject_depth_to_other_pose_seq2(K, depth_seq, rgb_seq, poses_src_seq, pose_dst):
    """
    多帧 → 一张图：复用你已有的单帧函数 reproject_depth_to_other_pose，再用上面的 z-buffer 合成。
    depth_seq: (T,H,W), rgb_seq: (T,H,W,3), poses_src_seq: (T,4,4), pose_dst: (4,4)
    """
    T, H, W = depth_seq.shape
    pts_list, col_list = [], []
    for i in range(T):
        pts_i, col_i = reproject_depth_to_other_pose(
            K, depth_seq[i], rgb_seq[i], poses_src_seq[i], pose_dst
        )
        pts_list.append(pts_i)  # 在目标相机系下的 (Ni,3)
        col_list.append(col_i)  # (Ni,3)

    points_cam = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(col_list, axis=0)
    return points_cam, colors


def reproject_depth_to_other_pose_seq2seq(
    K, depth_maps, rgb_imgs, poses_src, poses_dst
):
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
        points_3d, colors = reproject_depth_to_other_pose_seq2(
            K, depth_maps, rgb_imgs, poses_src, poses_dst[i]
        )
        points_3d_all.append(points_3d)
        colors_all.append(colors)

    return points_3d_all, colors_all


def project_to_2d_image_seq2seq(K, points_3d, colors, image_size):
    B = len(points_3d)
    images = []

    for i in range(B):
        image = project_to_2d_image_seq2(K, points_3d[i], colors[i], image_size)
        images.append(image)

    return np.stack(images, axis=0)  # shape: (B, H, W, 3)
