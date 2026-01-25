# ------------------------------------------------------------------------------
# 相机坐标转换工具函数
# 参考MM_GCN实现，复用simple_depth_reader的相机参数
# ------------------------------------------------------------------------------

import numpy as np
from typing import Dict, Tuple
from utils.simple_depth_reader import SimpleDepthReader

# PyTorch为可选依赖，仅在需要时导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def pixel2cam(pixel_coord: np.ndarray, f: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    将像素坐标+深度转换为相机坐标系3D坐标（参考MM_GCN实现）
    
    Args:
        pixel_coord (np.ndarray): 像素坐标+深度 [N, 3] 格式为 (u, v, z_depth)
        f (np.ndarray): 焦距参数 [fx, fy]
        c (np.ndarray): 主点参数 [cx, cy]
        
    Returns:
        np.ndarray: 相机坐标系3D坐标 [N, 3] 格式为 (X_cam, Y_cam, Z_cam)
    """
    # MM_GCN的实现
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def cam2pixel(cam_coord: np.ndarray, f: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    将相机坐标系3D坐标转换为像素坐标+深度（参考MM_GCN实现）
    
    Args:
        cam_coord (np.ndarray): 相机坐标系3D坐标 [N, 3] 格式为 (X_cam, Y_cam, Z_cam)
        f (np.ndarray): 焦距参数 [fx, fy]  
        c (np.ndarray): 主点参数 [cx, cy]
        
    Returns:
        np.ndarray: 像素坐标+深度 [N, 3] 格式为 (u, v, z_depth)
    """
    # MM_GCN的实现
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def convert_joints_to_camera_coords(joints_pixel_depth: np.ndarray, 
                                   depth_reader: SimpleDepthReader) -> np.ndarray:
    """
    使用现有SimpleDepthReader的相机参数转换关键点坐标
    
    Args:
        joints_pixel_depth (np.ndarray): 关键点像素坐标+深度 [num_joints, 3]
        depth_reader (SimpleDepthReader): 现有的深度读取器实例
        
    Returns:
        np.ndarray: 相机坐标系关键点 [num_joints, 3]
    """
    # 直接使用现有的相机参数
    fx = depth_reader.camera_params['fx']
    fy = depth_reader.camera_params['fy']
    cx = depth_reader.camera_params['cx1']  # 注意：现有的是cx1
    cy = depth_reader.camera_params['cy']
    
    f = np.array([fx, fy], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    
    return pixel2cam(joints_pixel_depth, f, c)


def convert_joints_from_camera_coords(joints_camera: np.ndarray, 
                                     depth_reader: SimpleDepthReader) -> np.ndarray:
    """
    将相机坐标系关键点转换回像素坐标+深度（逆变换）
    
    Args:
        joints_camera (np.ndarray): 相机坐标系关键点 [num_joints, 3] (X_cam, Y_cam, Z_cam)
        depth_reader (SimpleDepthReader): 现有的深度读取器实例
        
    Returns:
        np.ndarray: 像素坐标+深度 [num_joints, 3] (u, v, z)
        
    Note:
        - 用于训练时将相机坐标GT转换回像素坐标，确保与HRNet预测的坐标空间一致
        - 遮挡点(0,0,0)相机坐标会转换为(cx, cy, 0)像素坐标，保持合理性
    """
    # 复用现有的相机参数
    fx = depth_reader.camera_params['fx']
    fy = depth_reader.camera_params['fy']
    cx = depth_reader.camera_params['cx1']  # 注意：现有的是cx1
    cy = depth_reader.camera_params['cy']
    
    f = np.array([fx, fy], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    
    return cam2pixel(joints_camera, f, c)


def torch_cam2pixel(cam_coord, f, c):
    """
    PyTorch版本的相机坐标到像素坐标转换（用于训练中的逆变换）
    
    Args:
        cam_coord (torch.Tensor): 相机坐标 [B, N, 3] 格式为 (X_cam, Y_cam, Z_cam)
        f (torch.Tensor): 焦距参数 [B, 2] 或 [2]
        c (torch.Tensor): 主点参数 [B, 2] 或 [2]
        
    Returns:
        torch.Tensor: 像素坐标+深度 [B, N, 3] 格式为 (u, v, z_depth)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch不可用")
    
    # 确保f和c的维度正确
    if len(f.shape) == 1:
        f = f.unsqueeze(0)  # [2] -> [1, 2]
    if len(c.shape) == 1:
        c = c.unsqueeze(0)  # [2] -> [1, 2]
    
    # 处理批次维度
    if f.shape[0] == 1 and cam_coord.shape[0] > 1:
        f = f.repeat(cam_coord.shape[0], 1)  # [1, 2] -> [B, 2]
    if c.shape[0] == 1 and cam_coord.shape[0] > 1:
        c = c.repeat(cam_coord.shape[0], 1)  # [1, 2] -> [B, 2]
    
    # 相机坐标到像素坐标的投影变换
    # 添加小值避免除零
    z_safe = cam_coord[:, :, 2:3] + 1e-8
    x = cam_coord[:, :, 0:1] / z_safe * f[:, 0:1].unsqueeze(1) + c[:, 0:1].unsqueeze(1)
    y = cam_coord[:, :, 1:2] / z_safe * f[:, 1:2].unsqueeze(1) + c[:, 1:2].unsqueeze(1)
    z = cam_coord[:, :, 2:3]
    
    return torch.cat([x, y, z], dim=-1)


def torch_pixel2cam(pixel_coord, f, c):
    """
    PyTorch版本的像素坐标到相机坐标转换（参考MM_GCN）
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch不可用")
    
    x = (pixel_coord[:, :, 0] - c[:, 0:1]) / f[:, 0:1] * pixel_coord[:, :, 2]
    y = (pixel_coord[:, :, 1] - c[:, 1:2]) / f[:, 1:2] * pixel_coord[:, :, 2]
    z = pixel_coord[:, :, 2]
    
    return torch.stack([x, y, z], dim=-1)


def sample_z_init_from_hrnet_and_depth(hrnet_coords: 'torch.Tensor', 
                                       depth_input: 'torch.Tensor',
                                       depth_reader: SimpleDepthReader,
                                       depth_max_value: float = 10000.0,
                                       interpolation_mode: str = 'nearest') -> 'torch.Tensor':
    """
    基于HRNet输出坐标和深度图计算z_init（归一化的相机坐标系深度值）
    
    实现完整流程：
    1. HRNet输出xy → 像素坐标（逆归一化）
    2. 在深度图上采样深度值
    3. 转换为相机坐标系
    4. 使用simple_depth_reader的归一化策略
    
    Args:
        hrnet_coords (torch.Tensor): HRNet输出的归一化坐标 [B, 17, 2] 范围约[-1,1]
        depth_input (torch.Tensor): 深度图 [B, 1, H, W] 已归一化到[0,1]
        depth_reader (SimpleDepthReader): 相机参数和归一化工具
        depth_max_value (float): 最大深度值，用于逆归一化
        interpolation_mode (str): 插值模式，'nearest'或'bilinear'
        
    Returns:
        torch.Tensor: 归一化的z坐标 [B, 17, 1] 范围[0,1]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch不可用")
    
    B, num_joints, _ = hrnet_coords.shape
    _, _, H, W = depth_input.shape
    device = hrnet_coords.device
    
    # 🔄 步骤1: 将HRNet归一化坐标转换为深度图像素坐标
    # HRNet输出大约在[-1, 1]范围，需要转换到[0, H-1]和[0, W-1]
    pixel_coords = hrnet_coords.clone()
    pixel_coords[:, :, 0] = (pixel_coords[:, :, 0] + 1) * (W - 1) / 2  # x: [-1,1] -> [0, W-1]
    pixel_coords[:, :, 1] = (pixel_coords[:, :, 1] + 1) * (H - 1) / 2  # y: [-1,1] -> [0, H-1]
    
    # 🎯 步骤2: 在深度图上采样深度值（使用grid_sample双线性插值）
    # 转换为grid_sample期望的格式[-1, 1]
    grid = pixel_coords.clone()
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1  # x: [0, W-1] -> [-1, 1]
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1  # y: [0, H-1] -> [-1, 1]
    
    # 🎯 执行深度采样（支持配置的插值模式）
    # nearest: 保持真实深度值，避免跨物体深度混合
    # bilinear: 平滑插值，可能在某些场景下提供更好的连续性
    grid_sample_format = grid.unsqueeze(2)  # [B, 17, 1, 2]
    sampled_depth_normalized = torch.nn.functional.grid_sample(
        depth_input, grid_sample_format, 
        align_corners=True, mode=interpolation_mode, padding_mode='border'
    )  # [B, 1, 17, 1]
    
    sampled_depth_normalized = sampled_depth_normalized.squeeze(1).transpose(1, 2)  # [B, 17, 1]
    
    # 🔄 步骤3: 逆归一化深度值到毫米单位（与simple_depth_reader一致）
    sampled_depth_mm = sampled_depth_normalized * depth_max_value  # [B, 17, 1]
    
    # 🏗️ 步骤4: 构建完整的像素+深度坐标用于转换
    # 需要将像素坐标转换回原始图像坐标系（非归一化）用于相机坐标转换
    fx = depth_reader.camera_params['fx']
    fy = depth_reader.camera_params['fy']
    cx = depth_reader.camera_params['cx1']
    cy = depth_reader.camera_params['cy']
    
    # 创建相机参数张量
    f = torch.tensor([fx, fy], dtype=torch.float32, device=device)
    c = torch.tensor([cx, cy], dtype=torch.float32, device=device)
    
    # 🔄 步骤5: 将像素坐标转换为相机坐标（仅为了获得物理意义的z值）
    # 注意：这里我们主要关心z坐标的物理意义，xy坐标会在后续normalize中处理
    pixel_coords_with_depth = torch.cat([pixel_coords, sampled_depth_mm], dim=-1)  # [B, 17, 3]
    
    # 转换为相机坐标系（获得物理意义的深度）
    camera_coords = torch.zeros_like(pixel_coords_with_depth)
    for b in range(B):
        # 只处理有有效深度的点
        valid_mask = sampled_depth_mm[b, :, 0] > 0
        if valid_mask.any():
            valid_pixels = pixel_coords_with_depth[b][valid_mask]  # [N_valid, 3]
            # 使用torch版本的转换函数
            f_batch = f.unsqueeze(0)  # [1, 2]
            c_batch = c.unsqueeze(0)  # [1, 2]
            valid_pixels_batch = valid_pixels.unsqueeze(0)  # [1, N_valid, 3]
            valid_camera = torch_pixel2cam(valid_pixels_batch, f_batch, c_batch)[0]  # [N_valid, 3]
            camera_coords[b][valid_mask] = valid_camera
    
    # 🎯 步骤6: 使用simple_depth_reader的归一化策略处理z坐标
    # 提取z坐标（相机坐标系的前向深度）
    z_camera_mm = camera_coords[:, :, 2:3]  # [B, 17, 1]
    
    # 使用与simple_depth_reader相同的归一化策略
    z_normalized = torch.clamp(z_camera_mm, 0, depth_max_value) / depth_max_value  # [B, 17, 1]

    return z_normalized


def project_left_to_right(x_left: float, y_left: float, depth: float,
                          fx: float, baseline: float) -> Tuple[float, float]:
    """
    将左图像素坐标投影到右图（极线校准后的简化版本）

    由于图像已经过极线校准（rectified），极线变为水平线，
    同一3D点在左右图的y坐标相同，只有x坐标因视差而不同。

    Args:
        x_left: 左图x坐标（像素）
        y_left: 左图y坐标（像素）
        depth: 深度值（mm）
        fx: 左相机焦距（像素）
        baseline: 基线距离（mm），即左右相机光心之间的距离

    Returns:
        (x_right, y_right): 右图坐标（像素）

    Note:
        视差公式: disparity = (fx * baseline) / depth
        右图x坐标: x_right = x_left - disparity
        右图y坐标: y_right = y_left（极线校准后相同）
    """
    if depth <= 0:
        # 深度无效时返回原坐标
        return x_left, y_left

    # 计算视差
    disparity = (fx * baseline) / depth

    # 计算右图坐标
    x_right = x_left - disparity
    y_right = y_left  # 极线校准后y坐标相同

    return x_right, y_right


def project_keypoints_left_to_right(keypoints: Dict[str, list],
                                    fx: float,
                                    baseline: float) -> Dict[str, Tuple[float, float]]:
    """
    批量将左图关键点投影到右图

    Args:
        keypoints: 关键点字典，格式为 {name: [x, y, depth]}
        fx: 左相机焦距（像素）
        baseline: 基线距离（mm）

    Returns:
        右图关键点字典，格式为 {name: (x_right, y_right)}
    """
    right_keypoints = {}
    for name, kp in keypoints.items():
        x_left, y_left, depth = kp[0], kp[1], kp[2]
        x_right, y_right = project_left_to_right(x_left, y_left, depth, fx, baseline)
        right_keypoints[name] = (x_right, y_right)
    return right_keypoints
