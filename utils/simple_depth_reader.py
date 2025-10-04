#!/usr/bin/env python3
"""
简洁快速的深度图读取器

基于MonSter和MoCha-Stereo的简化版本, 使用YAML文件存储相机参数。
专注于快速读取和预处理深度图用于RGB-D训练。

特点：
1. 使用YAML配置文件管理相机参数
2. 简化的视差到深度转换逻辑
3. 专为RGB-D数据集优化的接口
4. 最小化依赖和代码复杂度

作者: Junfeng Xie
参考: MonSter demo_img.py 和 MoChaOutputs 项目
"""

from typing import Optional, Tuple, Union
import os
import numpy as np
import cv2
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleDepthReader:
    """
    简洁的深度图读取器
    
    使用YAML配置文件管理相机参数，提供快速的视差到深度转换功能。
    """
    
    def __init__(self, camera_config_path: Union[str, Path]):
        """
        初始化深度读取器
        
        Args:
            camera_config_path (Union[str, Path]): 相机参数YAML文件路径
        """
        self.config_path = Path(camera_config_path)
        self.camera_params = self._load_camera_config()
        logger.info(f"深度读取器初始化完成: {self.config_path}")
    
    def _load_camera_config(self) -> dict:
        """
        从YAML文件加载相机参数
        
        Returns:
            dict: 相机参数字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"相机配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取关键参数
            params = {
                'fx': config['camera_matrix_left']['data'][0],  # 左相机fx
                'fy': config['camera_matrix_left']['data'][4],  # 左相机fy
                'cx1': config['camera_matrix_left']['data'][2],  # 左相机cx
                'cy': config['camera_matrix_left']['data'][5],   # 左相机cy
                'cx2': config['camera_matrix_right']['data'][2], # 右相机cx
                'baseline': np.linalg.norm(np.array(config['T']['data']))  # 基线长度(mm)
            }
            
            logger.debug(f"相机参数加载成功: fx={params['fx']:.2f}, baseline={params['baseline']:.2f}mm")
            return params
            
        except Exception as e:
            raise ValueError(f"解析相机配置文件失败: {e}")
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        将视差图转换为深度图
        
        使用简化的双目立体视觉公式, 参考MonSter的快速转换方法。
        
        Args:
            disparity (np.ndarray): 视差图 [H, W]
            
        Returns:
            np.ndarray: 深度图 [H, W], 单位毫米
        """
        # 简化的深度计算（避免复杂的校正）
        fx = self.camera_params['fx']
        baseline = self.camera_params['baseline']
        
        # 防止除零错误
        valid_mask = np.abs(disparity) > 1e-6
        depth = np.zeros_like(disparity, dtype=np.float32)
        
        # 简化公式：depth = fx * baseline / disparity
        depth[valid_mask] = fx * baseline / np.abs(disparity[valid_mask])
        
        return depth  # NOTE: 这里的深度图是毫米单位，z值为相机坐标系前向深度（非世界坐标）
    
    def read_disparity(self, disp_path: Union[str, Path]) -> np.ndarray:
        """
        读取视差图文件
        
        支持多种格式:
        1. .npy格式: 直接加载原始视差数据 (!推荐! 使用模型生成的.npy文件!)
        2. 灰度图格式: 普通的灰度深度图 (不推荐)
        3. 彩色图格式: matplotlib保存的jet colormap视差图 (极其不推荐)
        
        Args:
            disp_path (Union[str, Path]): 视差图文件路径
            
        Returns:
            np.ndarray: 视差图数组, 归一化后的浮点值
        """
        disp_path = Path(disp_path)
        
        if not disp_path.exists():
            raise FileNotFoundError(f"视差图文件不存在: {disp_path}")
        
        if disp_path.suffix.lower() == '.npy':
            # MonSter和MoCha-Stereo的.npy格式 - 原始视差数据 (!推荐! 使用模型生成的.npy文件!)
            disparity = np.load(disp_path)
            if len(disparity.shape) > 2:
                disparity = disparity.squeeze()
            logger.debug(f"读取.npy视差图: {disp_path}, 形状: {disparity.shape}")
            
        else:
            # 图片格式 - 需要判断是灰度图还是彩色图 (不推荐)
            img = cv2.imread(str(disp_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片文件: {disp_path}")
            
            # 检查是否为彩色图（MonSter的jet colormap输出） (极其不推荐)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 彩色图：需要从jet colormap反推原始值
                disparity = self._colormap_to_disparity(img)
                logger.debug(f"读取彩色视差图(jet colormap): {disp_path}, 形状: {disparity.shape}")
            else:
                # 灰度图：直接使用
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                disparity = img.astype(np.float32)
                # 假设图片已经是0-255范围，需要归一化
                disparity = disparity / 255.0 * 100.0  # 假设最大视差为100像素
                logger.debug(f"读取灰度视差图: {disp_path}, 形状: {disparity.shape}")
        
        logger.debug(f"视差图读取完成，范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        return disparity
    
    def _colormap_to_disparity(self, colormap_img: np.ndarray) -> np.ndarray:
        """
        从jet colormap图像反推原始视差值 (极其不推荐)
        
        MonSter使用matplotlib的jet colormap保存视差图，需要反向转换。
        提供两种方法：精确的colormap反推和快速的近似方法。
        
        Args:
            colormap_img (np.ndarray): BGR格式的彩色图像 [H, W, 3]
            
        Returns:
            np.ndarray: 恢复的视差图 [H, W]
        """
        try:
            # 方法1：使用matplotlib精确反推（较慢但准确） (极其不推荐)
            return self._precise_colormap_inversion(colormap_img)
        except ImportError:
            # 方法2：快速近似方法（快速但可能有精度损失） (极其不推荐)
            return self._fast_colormap_approximation(colormap_img)
    
    def _precise_colormap_inversion(self, colormap_img: np.ndarray) -> np.ndarray:
        """精确的colormap反推方法（需要matplotlib）"""
        import matplotlib.cm as cm
        from scipy.spatial.distance import cdist
        
        # 将BGR转换为RGB
        rgb_img = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]
        
        # 获取jet colormap的所有颜色
        jet_cmap = cm.get_cmap('jet')
        lut_size = 256
        
        # 生成colormap的颜色查找表
        colormap_indices = np.linspace(0, 1, lut_size)
        jet_colors = jet_cmap(colormap_indices)[:, :3] * 255  # 转换为0-255范围
        
        # reshape图像为(N, 3)
        pixels = rgb_img.reshape(-1, 3).astype(np.float32)
        
        # 使用scipy的快速距离计算找到最近邻
        distances = cdist(pixels, jet_colors, metric='euclidean')
        closest_indices = np.argmin(distances, axis=1)
        
        # 将colormap索引转换为视差值（假设最大视差为100像素）
        max_disparity = 100.0
        disparity_values = (closest_indices / (lut_size - 1)) * max_disparity
        
        # reshape回原始形状
        disparity = disparity_values.reshape(h, w).astype(np.float32)
        
        logger.debug(f"精确colormap转换完成，视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        return disparity
    
    def _fast_colormap_approximation(self, colormap_img: np.ndarray) -> np.ndarray:
        """
        快速近似方法：基于jet colormap的RGB特征
        
        jet colormap的特点：
        - 蓝色(低值) -> 青色 -> 绿色 -> 黄色 -> 红色(高值)
        - 可以通过RGB比例大致推算数值
        """
        # 将BGR转换为RGB
        rgb_img = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # jet colormap的近似公式（基于经验观察）
        r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
        
        # 基于jet colormap的特征进行近似
        # 蓝色主导 -> 低值，红色主导 -> 高值
        blue_weight = b - (r + g) / 2
        red_weight = r - (g + b) / 2
        green_weight = g - abs(r - b) / 2
        
        # 综合权重计算视差值
        disparity_norm = np.clip(
            0.7 * red_weight + 0.2 * green_weight - 0.1 * blue_weight + 0.5,
            0.0, 1.0
        )
        
        # 假设最大视差为100像素
        max_disparity = 100.0
        disparity = disparity_norm * max_disparity
        
        logger.debug(f"快速近似转换完成，视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
        return disparity
    
    def read_depth(self, disp_path: Union[str, Path], 
                   target_size: Optional[Tuple[int, int]] = None,  
                   normalize: bool = True,
                   max_depth: Optional[float] = None) -> np.ndarray:
        """
        读取视差图并转换为深度图
        
        Args:
            disp_path (Union[str, Path]): 视差图文件路径
            target_size (Optional[Tuple[int, int]]): 目标尺寸 (width, height) # NOTE: 这里的目标尺寸是原始图片的尺寸, 如1440*1080
            normalize (bool): 是否归一化到[0, 1]
            max_depth (Optional[float]): 最大深度值用于归一化
            
        Returns:
            np.ndarray: 深度图 [H, W] 或 [H, W, 1]
        """
        # 读取视差图
        disparity = self.read_disparity(disp_path)
        
        # 转换为深度图
        depth = self.disparity_to_depth(disparity)
        
        # 尺寸调整
        if target_size is not None:
            width, height = target_size
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # 归一化 NOTE: [xjf -> zmt]: 当你测试时，设置成False, 这样生成的深度图的值域为[0, max_depth]，而且返回的是真实的世界坐标系的深度图，coco_cp.py中目前设置的也是False
        if normalize:
            depth = self._normalize_depth(depth, max_depth)
        
        return depth
    
    def _normalize_depth(self, depth: np.ndarray, max_depth: Optional[float] = None) -> np.ndarray:
        """
        归一化深度图
        NOTE: [xjf]: 我发现这里的归一化方式没有问题, 可以使用它, 因为使用了用户可自定义的max_depth
        
        Args:
            depth (np.ndarray): 深度图
            max_depth (Optional[float]): 最大深度值
            
        Returns:
            np.ndarray: 归一化后的深度图 [0, 1]
        """
        valid_mask = depth > 0
        
        if not valid_mask.any():
            return depth
        
        if max_depth is None:
            raise ValueError("max_depth不能为None，请设置max_depth")
        
        normalized = np.zeros_like(depth)
        normalized[valid_mask] = np.clip(depth[valid_mask] / max_depth, 0, 1)
        
        return normalized.astype(np.float32)
    
    def read_for_dataset(self, disp_path: Union[str, Path],
                         target_size: Tuple[int, int] = (256, 256),
                         return_channel_last: bool = True) -> np.ndarray:
        """
        为RGB-D数据集读取深度图
        
        专门优化用于与JointsDataset配合使用的接口。
        
        Args:
            disp_path (Union[str, Path]): 视差图文件路径
            target_size (Tuple[int, int]): 目标尺寸，默认(256, 256)
            return_channel_last (bool): 是否返回[H, W, 1]格式
            
        Returns:
            np.ndarray: 深度图，格式为[H, W]或[H, W, 1]
        """
        depth = self.read_depth(
            disp_path=disp_path,
            target_size=target_size,
            normalize=True
        )
        
        if return_channel_last:
            depth = depth[:, :, np.newaxis]  # [H, W] -> [H, W, 1]
        
        return depth.astype(np.float32)


def create_camera_config_template(output_path: Union[str, Path]) -> None:
    """
    创建相机参数配置文件模板
    
    Args:
        output_path (Union[str, Path]): 输出文件路径
    """
    template = {
        'camera_matrix_left': {
            'cols': 3,
            'rows': 3,
            'dt': 'd',
            'data': [
                1774.02759117584, 0.0, 689.047651992725,  # fx, 0, cx
                0.0, 1774.12812333568, 536.130613432710,  # 0, fy, cy
                0.0, 0.0, 1.0                             # 0, 0, 1
            ]
        },
        'camera_matrix_right': {
            'cols': 3,
            'rows': 3, 
            'dt': 'd',
            'data': [
                1774.02759117584, 0.0, 703.759113246139,  # fx, 0, cx
                0.0, 1774.12812333568, 536.130613432710,  # 0, fy, cy
                0.0, 0.0, 1.0                             # 0, 0, 1
            ]
        },
        'T': {
            'cols': 1,
            'rows': 3,
            'dt': 'd',
            'data': [
                -40.42720433134619,  # baseline (负值表示右相机在左侧)
                0.0,
                0.0
            ]
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)
    
    print(f"相机配置模板已创建: {output_path}")


# 便捷函数
def quick_depth_read(disp_path: Union[str, Path],
                     camera_config: Union[str, Path],
                     target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    快速读取深度图的便捷函数
    
    Args:
        disp_path (Union[str, Path]): 视差图路径
        camera_config (Union[str, Path]): 相机配置文件路径
        target_size (Tuple[int, int]): 目标尺寸
        
    Returns:
        np.ndarray: 深度图 [H, W, 1]
    """
    reader = SimpleDepthReader(camera_config)
    return reader.read_for_dataset(disp_path, target_size)


if __name__ == "__main__":
    """测试简洁深度读取器"""
    import sys
    
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 简洁深度图读取器测试")
    print("=" * 50)
    
    # 创建测试用的相机配置文件
    test_config_path = "/tmp/test_camera_config.yaml"
    if not os.path.exists(test_config_path):
        create_camera_config_template(test_config_path)
        print(f"✅ 创建测试配置文件: {test_config_path}")
    
    # 测试文件路径
    test_disparity_path = "/home/junf/program/MoCha-Stereo-20250612T185702Z-1-001/0703_6_Huguang_128/npy/disparity_0001751539512357.npy"
    
    if not os.path.exists(test_disparity_path):
        print(f"❌ 测试文件不存在: {test_disparity_path}")
        print("📋 请确保视差图文件存在")
        sys.exit(1)
    
    try:
        # 测试简洁读取器
        reader = SimpleDepthReader(test_config_path)
        print(f"✅ 读取器创建成功")
        
        # 快速读取深度图
        depth = reader.read_for_dataset(
            disp_path=test_disparity_path,
            target_size=(256, 256),
            return_channel_last=True
        )
        print(f"✅ 深度图读取成功: {depth.shape}, dtype={depth.dtype}")
        print(f"   范围: [{depth.min():.3f}, {depth.max():.3f}]")
        
        # 测试便捷函数
        depth_quick = quick_depth_read(
            test_disparity_path,
            test_config_path,
            target_size=(128, 128)
        )
        print(f"✅ 便捷函数测试成功: {depth_quick.shape}")
        
        print(f"\n✅ 简洁深度读取器测试通过")
        print(f"💡 使用方法:")
        print(f"   from utils.simple_depth_reader import SimpleDepthReader")
        print(f"   reader = SimpleDepthReader('camera_config.yaml')")
        print(f"   depth = reader.read_for_dataset(disp_path, target_size=(256, 256))")
        
        print(f"\n⚡ 性能对比:")
        print(f"   原版深度读取器: 完整功能，支持复杂滤波和可视化")
        print(f"   简洁深度读取器: 核心功能，专注速度和简洁性")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
