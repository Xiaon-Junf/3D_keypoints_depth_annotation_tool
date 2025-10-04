#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼类关键点3D坐标验证和调整工具
用于加载深度图并验证关键点的深度值，支持手动调整不正确的深度值
NOTE [XJF]:
    非常好的标注工具, 以下是我的建议:
    1. 通过原有json文件的类别字段, 区分不同的鱼, 添加 previous fish && next fish 按钮, 实现逐条鱼的深度修改
    2. 将点云图上的部分点同步加载进本点云中, 逻辑如下:
        1. 从原来的json文件标注数据中提取bbox
        2. 读取节点的 min && max 深度值, 只加载这区间正负50深度的点云
        3. 调整合适后, 用户可以点击刷新button, 重新加载[z_min - 50, z_max + 50]区间的点云到窗口中 (也不一定是加载, 如果使用 "遮挡/mask" 的思想可以实现更快的可视化策略,\
            但我建议是重新加载, 因为坐标轴的过大会影响用户与交互界面的交互手感, 影响标注精度)
        4. 保存当前节点
        5. 如果遇到了点云加载数量过多、可视化麻烦的问题, 可以考虑Open3D, 这也是一个非常优秀的可交互式的3D可视化工具, 我在官网上甚至找到了3D标注的代码教程, \
            API文档见: https://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html \
            其中的球形标注法我认为可以参考, 配合拖条, 应该也可以很好地完成任务, 之前开发SGBM、RaftStereo等深度估计算法时, 也参考过他们官网的文档, 配合参数拖条实现了不错的效果 \
            如果你决定使用Open3D, 请参考你的MonSter_use仓库的可视化方法 \
                推荐原因: 交互非常方便, 按键盘上的1234可以直接切换4个不同深度的颜色热力图, 更加方便标注人员检查深度, 当然你也可以不采纳
    3. 用户拖动拖条调整z的大小时, 可视化界面的关节点也应该同步更新, 避免频繁点击button的检查滞后性 \
        注意, 不要把这一点和2.3点混淆, 2.3点是点云的加载, 这一点是关节点的同步更新
"""

import os
import sys
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
import yaml
import tempfile
import matplotlib
# 尝试使用不同的后端来避免GIL冲突
try:
    matplotlib.use('Qt5Agg')  # 首先尝试Qt5
except ImportError:
    try:
        matplotlib.use('TkAgg')  # 回退到TkAgg
    except ImportError:
        matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import warnings
warnings.filterwarnings("ignore")

# 多进程通信
import multiprocessing as mp
import threading
import time

# 添加项目路径到系统路径
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, lib_path)

try:
    from utils.simple_depth_reader import SimpleDepthReader
    from utils.camera_utils import convert_joints_to_camera_coords
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的环境中运行此脚本")
    sys.exit(1)

# 尝试导入Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: 未安装Open3D，将使用matplotlib进行3D可视化")


class Fish3DKeypointVerifier:
    def __init__(self, annotation_file, depth_root, camera_config):
        """
        初始化鱼类3D关键点验证器
        """
        self.annotation_file = annotation_file
        self.depth_root = depth_root
        self.camera_config = camera_config
        
        # 先初始化图形相关变量
        self.fig = None
        self.ax = None
        self.depth_slider = None
        self.buttons = {}
        
        # 加载相机参数
        self.camera_params = self._load_camera_params()
        
        # 初始化标注数据（将在加载帧时动态加载）
        self.annotation_data = None
        self.annotation_file = annotation_file  # 保存原始标注文件路径作为默认

        # 初始化深度读取器
        try:
            self.depth_reader = SimpleDepthReader(camera_config)
            print(f"成功初始化深度读取器: {camera_config}")
        except Exception as e:
            print(f"初始化深度读取器失败: {e}")
            print("将继续使用模拟深度数据")
            self.depth_reader = None

        # 初始化关键点数据（将在加载帧时动态加载）
        self.fish_keypoints = {}
        self.fish_names = []
        self.current_fish_idx = -1
        self.keypoints = {}
        self.keypoint_names = []
        
        # 当前帧索引
        self.current_frame_idx = 0
        self.frames = self._get_frames()
        print(f"找到 {len(self.frames)} 个帧文件: {self.frames}")
        
        # 当前关键点索引
        self.current_kp_idx = 0 if self.keypoint_names else -1
        
        # 标志位防止递归调用
        self.updating_slider = False
        self.updating_display = False
        
        # 3D可视化相关
        self.vis_window = None
        self.keypoint_meshes = {}
        
        # 全局点云可视化相关
        self.global_vis_window = None
        self.global_keypoint_meshes = {}

        # GUI子进程通信
        self.gui_process = None
        self.gui_pipe = None

        # 点云数据相关
        self.point_cloud_data = None
        self.point_cloud_filtered = None
        self.z_min = 0
        self.z_max = 10000
        
        # 图像数据
        self.image_data = None
        self.color_rectify_data = None  # 用于点云着色的彩色图像
        
        # 立即创建图形界面
        self._create_figure()
        
        # 加载第一帧数据
        if self.frames:
            self._load_current_frame()
        else:
            print("警告: 没有找到帧数据，无法加载深度图")
    
    def _load_camera_params(self):
        """
        加载相机参数
        """
        try:
            with open(self.camera_config, 'r', encoding='utf-8') as file:
                params = yaml.safe_load(file)
            return params
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            return None
        
    def _parse_keypoints_by_group_id(self):
        """
        按照group_id分配逻辑解析关键点
        group_id: 0 表示一条新的鱼
        group_id: null 表示属于当前鱼的关键点
        """
        fish_keypoints = {}
        current_fish_id = None
        fish_count = 0
        
        # 首先按顺序处理所有形状
        for shape in self.annotation_data['shapes']:
            group_id = shape.get('group_id')
            
            # 如果是矩形且group_id为0，表示一条新的鱼
            if shape['shape_type'] == 'rectangle' and shape['label'] == 'fish' and group_id == 0:
                fish_count += 1
                current_fish_id = f"fish_{fish_count}"
                fish_keypoints[current_fish_id] = {}
                print(f"找到新的鱼: {current_fish_id}")
            
            # 如果是关键点且group_id为null，分配给当前鱼
            elif shape['shape_type'] == 'point' and group_id is None:
                if current_fish_id is not None:
                    point = shape['points'][0]
                    x, y = point[0], point[1]
                    # 确保所有关键点初始深度值为0.0
                    fish_keypoints[current_fish_id][shape['label']] = np.array([x, y, 0.0], dtype=np.float32)
                    print(f"将关键点 '{shape['label']}' 分配给 {current_fish_id}")
                else:
                    print(f"警告: 关键点 '{shape['label']}' 没有对应的鱼，将被忽略")
            
            # 其他情况（如group_id不为0或null的矩形）
            else:
                print(f"忽略形状: {shape['label']} (类型: {shape['shape_type']}, group_id: {group_id})")
        
        # 验证分配结果
        if fish_keypoints:
            print(f"成功解析 {len(fish_keypoints)} 条鱼的关键点:")
            for fish_name, kps in fish_keypoints.items():
                print(f"  {fish_name}: {len(kps)} 个关键点 - {list(kps.keys())}")
        else:
            print("警告: 没有找到任何鱼的关键点")
        
        # 初始化所有鱼类的关键点深度值
        for fish_name, keypoints in fish_keypoints.items():
            for kp_name, kp_coords in keypoints.items():
                keypoints[kp_name] = np.array([kp_coords[0], kp_coords[1], 0.0], dtype=np.float32)
        
        return fish_keypoints

    def _get_frames(self):
        """
        获取所有帧的文件名 - 扫描images文件夹中的所有png文件
        只包含那些有对应深度文件和标注文件的图像
        """
        frames = []
        try:
            # 构建各个文件夹路径
            images_root = os.path.join(os.path.dirname(self.depth_root), 'images')
            annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme')

            if os.path.exists(images_root):
                # 扫描images文件夹中的所有png文件
                for file_name in os.listdir(images_root):
                    if file_name.lower().endswith('.png'):
                        base_name = os.path.splitext(file_name)[0]

                        # 检查是否存在对应的深度文件和标注文件
                        depth_file = os.path.join(self.depth_root, f"{base_name}.npy")
                        annotation_file = os.path.join(annotations_root, f"{base_name}.json")

                        if os.path.exists(depth_file) and os.path.exists(annotation_file):
                            frames.append(file_name)
                            print(f"添加有效帧: {file_name} (depth: {os.path.exists(depth_file)}, annotation: {os.path.exists(annotation_file)})")
                        else:
                            print(f"跳过无效帧: {file_name} (depth: {os.path.exists(depth_file)}, annotation: {os.path.exists(annotation_file)})")

                print(f"找到 {len(frames)} 个有效帧: {frames}")
            else:
                print(f"警告: images文件夹不存在: {images_root}")
                print("警告: 无法找到任何图像文件")

        except Exception as e:
            print(f"读取帧数据失败: {e}")

        return sorted(frames)
    
    def _load_current_frame(self):
        """
        加载当前帧的深度数据、图像和标注文件
        根据当前帧的文件名动态构建对应的文件路径
        """
        if not self.frames:
            print("没有可用的帧数据")
            return

        # 获取当前帧的文件名
        current_frame_name = self.frames[self.current_frame_idx]
        # 提取文件名（不含扩展名）
        base_name = os.path.splitext(current_frame_name)[0]

        print(f"正在加载帧: {current_frame_name} (basename: {base_name})")

        # 构建文件路径
        images_root = os.path.join(os.path.dirname(self.depth_root), 'images')
        annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme')

        depth_file_path = os.path.join(self.depth_root, f"{base_name}.npy")
        image_file_path = os.path.join(images_root, current_frame_name)
        annotation_file_path = os.path.join(annotations_root, f"{base_name}.json")

        print(f"深度文件路径: {depth_file_path}")
        print(f"图像文件路径: {image_file_path}")
        print(f"标注文件路径: {annotation_file_path}")

        # 获取图像尺寸
        try:
            with open(self.camera_config, 'r', encoding='utf-8') as f:
                camera_data = yaml.safe_load(f)
                image_size = camera_data.get('image_size', [1440, 1080])
            print(f"图像尺寸: {image_size[0]}x{image_size[1]}")
        except Exception as e:
            print(f"读取相机配置失败，使用默认尺寸: {e}")
            image_size = [1440, 1080]

        # 读取图像
        try:
            if os.path.exists(image_file_path):
                self.image_data = cv2.imread(image_file_path)
                self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
                print(f"成功加载图像: {image_file_path}, 尺寸: {self.image_data.shape}")
            else:
                print(f"图像文件不存在: {image_file_path}")
                self.image_data = None
        except Exception as e:
            print(f"加载图像失败: {e}")
            self.image_data = None

        # 设置color_rectify图像（假设与显示图像相同）
        self.color_rectify_data = self.image_data

        # 读取标注文件
        try:
            if os.path.exists(annotation_file_path):
                with open(annotation_file_path, 'r', encoding='utf-8') as f:
                    self.annotation_data = json.load(f)
                print(f"成功加载标注文件: {annotation_file_path}")
                print(f"标注文件包含 {len(self.annotation_data['shapes'])} 个形状")

                # 解析关键点，按鱼类分组
                self.fish_keypoints = self._parse_keypoints_by_group_id()
                self.fish_names = list(self.fish_keypoints.keys())
                self.current_fish_idx = 0 if self.fish_names else -1

                if self.fish_names:
                    self.keypoints = self.fish_keypoints[self.fish_names[self.current_fish_idx]]
                    self.keypoint_names = list(self.keypoints.keys())
                    print(f"找到 {len(self.fish_names)} 条鱼: {self.fish_names}")
                    print(f"当前鱼类 '{self.fish_names[self.current_fish_idx]}' 包含 {len(self.keypoint_names)} 个关键点: {self.keypoint_names}")
                else:
                    self.keypoints = {}
                    self.keypoint_names = []
                    print("警告: 没有找到任何鱼的关键点")
            else:
                print(f"标注文件不存在: {annotation_file_path}，使用默认标注数据")
                # 如果没有对应的标注文件，尝试使用默认的标注文件
                if os.path.exists(self.annotation_file):
                    with open(self.annotation_file, 'r', encoding='utf-8') as f:
                        self.annotation_data = json.load(f)
                    print(f"使用默认标注文件: {self.annotation_file}")
                    # 解析关键点
                    self.fish_keypoints = self._parse_keypoints_by_group_id()
                    self.fish_names = list(self.fish_keypoints.keys())
                    self.current_fish_idx = 0 if self.fish_names else -1
                    if self.fish_names:
                        self.keypoints = self.fish_keypoints[self.fish_names[self.current_fish_idx]]
                        self.keypoint_names = list(self.keypoints.keys())
                else:
                    print("警告: 默认标注文件也不存在")
                    self.annotation_data = None
                    self.fish_keypoints = {}
                    self.fish_names = []
                    self.current_fish_idx = -1
                    self.keypoints = {}
                    self.keypoint_names = []
        except Exception as e:
            print(f"加载标注文件失败: {e}")
            self.annotation_data = None
            self.fish_keypoints = {}
            self.fish_names = []
            self.current_fish_idx = -1
            self.keypoints = {}
            self.keypoint_names = []

        # 读取深度图 - 使用深度读取器正确转换视差到深度
        try:
            if os.path.exists(depth_file_path):
                # 使用深度读取器正确读取和转换深度数据
                if self.depth_reader is not None:
                    try:
                        # 使用read_depth方法，但不进行归一化以保持真实深度值
                        self.depth_data = self.depth_reader.read_depth(
                            disp_path=depth_file_path,
                            target_size=(image_size[0], image_size[1]),  # (width, height)
                            normalize=False  # 保持真实深度值（毫米）
                        )
                        print(f"成功通过深度读取器加载深度图: {depth_file_path}, 尺寸: {self.depth_data.shape}")
                        print(f"深度范围: [{self.depth_data.min():.2f}, {self.depth_data.max():.2f}] mm")
                    except Exception as e:
                        print(f"深度读取器读取失败，回退到直接读取: {e}")
                        # 回退到直接读取.npy文件（作为视差数据）
                        disparity_data = np.load(depth_file_path)
                        if self.depth_reader is not None:
                            # 手动转换视差到深度
                            self.depth_data = self.depth_reader.disparity_to_depth(disparity_data)
                            # 调整尺寸
                            if self.depth_data.shape != (image_size[1], image_size[0]):
                                self.depth_data = cv2.resize(self.depth_data, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
                            print(f"成功通过视差转换加载深度图: {depth_file_path}, 尺寸: {self.depth_data.shape}")
                            print(f"深度范围: [{self.depth_data.min():.2f}, {self.depth_data.max():.2f}] mm")
                        else:
                            raise Exception("深度读取器不可用")
                else:
                    print("警告: 深度读取器未初始化，使用模拟深度数据")
                    self.depth_data = np.ones((image_size[1], image_size[0])) * 1000  # 默认深度1000mm
            else:
                print(f"深度文件不存在: {depth_file_path}")
                # 创建模拟深度图用于测试
                print("创建模拟深度图用于测试")
                self.depth_data = np.ones((image_size[1], image_size[0])) * 1000  # 默认深度1000mm
        except Exception as e:
            print(f"加载深度图失败: {e}")
            # 尝试创建模拟深度图用于测试
            print("创建模拟深度图用于测试")
            self.depth_data = np.ones((image_size[1], image_size[0])) * 1000  # 默认深度1000mm
            
        # 更新所有鱼类关键点的深度值
        self._update_all_fish_keypoint_depths()
        
        # 加载点云数据
        self._load_point_cloud_data(depth_file_path)
    
    def _load_point_cloud_data(self, frame_path):
        """
        加载点云数据用于3D可视化 - 使用color_rectify图像着色
        """
        try:
            # 从深度图生成点云数据
            if self.depth_data is not None:
                height, width = self.depth_data.shape
                
                # 创建坐标网格
                x_coords = np.arange(width)
                y_coords = np.arange(height)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                
                # 有效深度值掩码 - 排除无效深度
                valid_mask = (self.depth_data > 0) & (self.depth_data < 10000)  # 限制在10米范围内
                
                # 不使用随机采样，保留所有有效点
                valid_indices = np.where(valid_mask)
                y_sample, x_sample = valid_indices
                
                # 提取有效点
                x_points = x_sample
                y_points = y_sample
                z_points = self.depth_data[y_sample, x_sample]

                # 翻转x坐标以匹配图像显示（只影响可视化，不影响深度读取）
                x_points = width - 1 - x_points

                # 翻转z坐标以修复深度方向
                z_points = -z_points

                # 创建点云 - 保持与图像一致的坐标系
                # 图像坐标系：X向右，Y向下，Z向前（深度方向翻转为负值以修正显示）
                self.point_cloud_data = np.column_stack((x_points, y_points, z_points))
                print(f"生成点云数据，包含 {len(self.point_cloud_data)} 个点")
                
                # 初始过滤点云
                self._filter_point_cloud()
        except Exception as e:
            print(f"加载点云数据失败: {e}")
            self.point_cloud_data = None
            self.point_cloud_filtered = None
    
    def _filter_point_cloud(self):
        """
        根据当前关键点的深度范围过滤点云数据
        """
        if self.point_cloud_data is None:
            self.point_cloud_filtered = None
            return
        
        # 计算当前关键点的深度范围
        valid_depths = [kp[2] for kp in self.keypoints.values() if kp[2] > 0]
        if valid_depths:
            # 由于点云z坐标被翻转为负数，需要相应调整过滤范围
            depth_min = max(0, min(valid_depths) - 50)  # 最小深度设为0mm
            depth_max = min(10000, max(valid_depths) + 50)  # 最大深度设为10000mm
            self.z_min = -depth_max  # 转换为负数范围
            self.z_max = -depth_min  # 转换为负数范围
        else:
            self.z_min = -10000  # 负数深度范围
            self.z_max = 0
        
        # 根据深度范围过滤点云（现在z坐标是负数）
        z_coords = self.point_cloud_data[:, 2]
        mask = (z_coords >= self.z_min) & (z_coords <= self.z_max)
        self.point_cloud_filtered = self.point_cloud_data[mask]
        print(f"过滤点云数据，保留深度范围 [{self.z_min}, {self.z_max}] 内的 {len(self.point_cloud_filtered)} 个点")
    
    def _update_keypoint_depths(self):
        """
        根据深度图更新关键点的Z坐标
        """
        if self.depth_data is None:
            print("深度数据为空，跳过关键点深度更新")
            return
            
        height, width = self.depth_data.shape
            
        # 更新当前鱼类的所有关键点深度值
        for name, kp in self.keypoints.items():
            x, y = int(round(kp[0])), int(round(kp[1]))
            
            if 0 <= x < width and 0 <= y < height:
                try:
                    depth_value = self.depth_data[y, x]
                    if depth_value is not None and depth_value > 0:
                        if depth_value <= 10000.0:  # 限制在5米范围内
                            self.keypoints[name][2] = float(depth_value)
                        else:
                            self.keypoints[name][2] = 0.0
                    else:
                        self.keypoints[name][2] = 0.0
                except Exception as e:
                    print(f"读取关键点 {name} 的深度值失败: {e}")
                    self.keypoints[name][2] = 0.0
            else:
                print(f"关键点 {name} 坐标 ({x}, {y}) 超出图像范围 ({width}, {height})")
                self.keypoints[name][2] = 0.0
    
    def _update_all_fish_keypoint_depths(self):
        """
        更新所有鱼类关键点的深度值，但保留用户已调整的值
        """
        # 保存当前鱼类索引和关键点
        current_fish_idx = self.current_fish_idx
        current_keypoints = self.keypoints
        current_keypoint_names = self.keypoint_names
        
        # 保存当前所有鱼类的深度值
        saved_depths = {}
        for fish_name, keypoints in self.fish_keypoints.items():
            saved_depths[fish_name] = {}
            for kp_name, kp_value in keypoints.items():
                saved_depths[fish_name][kp_name] = kp_value[2]  # 保存当前深度值
        
        # 更新每条鱼的关键点深度值
        for fish_idx, fish_name in enumerate(self.fish_names):
            # 切换到当前鱼类
            self.current_fish_idx = fish_idx
            self.keypoints = self.fish_keypoints[fish_name]
            self.keypoint_names = list(self.keypoints.keys())
            
            # 更新该鱼类关键点的深度值
            self._update_keypoint_depths()
            
            # 恢复用户调整的深度值（如果之前调整过）
            for kp_name, kp_value in self.keypoints.items():
                if fish_name in saved_depths and kp_name in saved_depths[fish_name]:
                    # 如果之前有调整过深度值，则保留调整后的值
                    if saved_depths[fish_name][kp_name] != 0.0:  # 0.0表示未调整
                        self.keypoints[kp_name][2] = saved_depths[fish_name][kp_name]
        
        # 恢复当前鱼类索引和关键点
        self.current_fish_idx = current_fish_idx
        if self.fish_names:
            self.keypoints = self.fish_keypoints[self.fish_names[self.current_fish_idx]]
            self.keypoint_names = list(self.keypoints.keys())
        else:
            self.keypoints = current_keypoints
            self.keypoint_names = current_keypoint_names
            
        # 更新点云过滤范围
        self._filter_point_cloud()
    
    def next_frame(self, event=None):
        """
        切换到下一帧
        """
        if not self.frames:
            return
            
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        print(f"切换到帧索引: {self.current_frame_idx}")
        self._load_current_frame()
        self.update_display()
    
    def prev_frame(self, event=None):
        """
        切换到上一帧
        """
        if not self.frames:
            return
            
        self.current_frame_idx = (self.current_frame_idx - 1) % len(self.frames)
        print(f"切换到帧索引: {self.current_frame_idx}")
        self._load_current_frame()
        self.update_display()
    
    def next_fish(self, event=None):
        """
        切换到下一条鱼
        """
        if not self.fish_names:
            return
            
        self.current_fish_idx = (self.current_fish_idx + 1) % len(self.fish_names)
        self._update_current_fish()
        self.update_display()
    
    def prev_fish(self, event=None):
        """
        切换到上一条鱼
        """
        if not self.fish_names:
            return
            
        self.current_fish_idx = (self.current_fish_idx - 1) % len(self.fish_names)
        self._update_current_fish()
        self.update_display()
    
    def _update_current_fish(self):
        """
        更新当前鱼类的关键点
        """
        if not self.fish_names:
            return

        fish_name = self.fish_names[self.current_fish_idx]
        self.keypoints = self.fish_keypoints[fish_name]
        self.keypoint_names = list(self.keypoints.keys())
        self.current_kp_idx = 0 if self.keypoint_names else -1
        print(f"切换到鱼类: {fish_name}, 包含 {len(self.keypoint_names)} 个关键点")

        # 不再更新关键点深度值，保留用户调整的值
        # self._update_keypoint_depths()
        # 更新点云过滤范围
        self._filter_point_cloud()

        # 发送切换鱼类的消息到GUI子进程
        self._send_gui_fish_update()
    
    def adjust_depth(self, val):
        """
        调整当前关键点的深度值
        """
        if self.updating_slider or not self.keypoint_names:
            return
            
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        print(f"调整关键点 {current_kp_name} 的深度值为: {val}")
        self.keypoints[current_kp_name][2] = float(val)
        # 更新点云过滤范围以适应新的深度值
        self._filter_point_cloud()
        self.update_display()
        
        # 实时更新3D可视化窗口中的关键点位置（如果窗口存在）
        self._update_3d_keypoint_position()
        
        # 实时更新全局点云窗口中的关键点位置（如果窗口存在）
        self._update_global_3d_keypoint_position()

        # 发送更新消息到GUI子进程
        self._send_gui_update()
    
    def _update_3d_keypoint_position(self):
        """
        实时更新3D可视化窗口中的关键点位置
        """
        if self.vis_window is None or not self.keypoint_names:
            return
            
        try:
            current_kp_name = self.keypoint_names[self.current_kp_idx]
            if current_kp_name in self.keypoint_meshes:
                kp = self.keypoints[current_kp_name]
                
                # 获取图像尺寸用于坐标翻转
                height, width = self.depth_data.shape
                
                # 计算新的位置（应用坐标翻转）
                x_flipped = width - 1 - kp[0]
                z_flipped = -kp[2]
                new_position = np.array([x_flipped, kp[1], z_flipped])
                
                # 更新现有球体的位置（使用变换矩阵）
                old_sphere = self.keypoint_meshes[current_kp_name]
                
                # 计算位移
                current_center = old_sphere.get_center()
                translation = new_position - current_center
                
                # 应用变换
                old_sphere.translate(translation)
                
                # 更新几何体
                self.vis_window.update_geometry(old_sphere)
                
                # 强制重绘
                self.vis_window.poll_events()
                self.vis_window.update_renderer()
                
                print(f"实时更新关键点 {current_kp_name} 的3D位置: {new_position}")
        except Exception as e:
            print(f"更新3D关键点位置失败: {e}")
            # 如果更新失败，可能是因为窗口已关闭，清理引用
            if "destroyed" in str(e).lower() or "closed" in str(e).lower():
                self.vis_window = None
                self.keypoint_meshes = {}
    
    def _update_global_3d_keypoint_position(self):
        """
        实时更新全局点云窗口中的关键点位置
        """
        if self.global_vis_window is None or not self.keypoint_names:
            return
            
        try:
            current_kp_name = self.keypoint_names[self.current_kp_idx]
            if current_kp_name in self.global_keypoint_meshes:
                kp = self.keypoints[current_kp_name]
                
                # 获取图像尺寸用于坐标翻转
                height, width = self.depth_data.shape
                
                # 计算新的位置（应用坐标翻转）
                x_flipped = width - 1 - kp[0]
                z_flipped = -kp[2]
                new_position = np.array([x_flipped, kp[1], z_flipped])
                
                # 更新现有球体的位置（使用变换矩阵）
                old_sphere = self.global_keypoint_meshes[current_kp_name]
                
                # 计算位移
                current_center = old_sphere.get_center()
                translation = new_position - current_center
                
                # 应用变换
                old_sphere.translate(translation)
                
                # 更新几何体
                self.global_vis_window.update_geometry(old_sphere)
                
                # 强制重绘
                self.global_vis_window.poll_events()
                self.global_vis_window.update_renderer()
                
                print(f"实时更新全局关键点 {current_kp_name} 的3D位置: {new_position}")
        except Exception as e:
            print(f"更新全局3D关键点位置失败: {e}")
            # 如果更新失败，可能是因为窗口已关闭，清理引用
            if "destroyed" in str(e).lower() or "closed" in str(e).lower():
                self.global_vis_window = None
                self.global_keypoint_meshes = {}
    
    def next_keypoint(self, event=None):
        """
        切换到下一个关键点
        """
        if not self.keypoint_names:
            return
            
        self.current_kp_idx = (self.current_kp_idx + 1) % len(self.keypoint_names)
        print(f"切换到关键点索引: {self.current_kp_idx}")
        self.update_display()
    
    def prev_keypoint(self, event=None):
        """
        切换到上一个关键点
        """
        if not self.keypoint_names:
            return
            
        self.current_kp_idx = (self.current_kp_idx - 1) % len(self.keypoint_names)
        print(f"切换到关键点索引: {self.current_kp_idx}")
        self.update_display()
    
    def refresh_point_cloud(self, event=None):
        """
        刷新点云显示范围
        """
        self._filter_point_cloud()
        print(f"刷新点云显示范围为: [{self.z_min}, {self.z_max}]")

        # 发送点云范围更新消息到GUI子进程
        self._send_gui_point_cloud_update()
    
    def _create_figure(self):
        """
        创建图形界面 - 在初始化时立即创建
        """
        print("创建图形界面")
        # 创建图形界面
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.3) 
        
        # 创建按钮
        button_props = {
            'prev_frame': {'rect': [0.05, 0.05, 0.08, 0.04], 'label': 'Prev Frame'},
            'next_frame': {'rect': [0.14, 0.05, 0.08, 0.04], 'label': 'Next Frame'},
            'prev_fish': {'rect': [0.23, 0.05, 0.08, 0.04], 'label': 'Prev Fish'},
            'next_fish': {'rect': [0.32, 0.05, 0.08, 0.04], 'label': 'Next Fish'},
            'prev_kp': {'rect': [0.41, 0.05, 0.08, 0.04], 'label': 'Prev KP'},
            'next_kp': {'rect': [0.50, 0.05, 0.08, 0.04], 'label': 'Next KP'},
            'refresh_pc': {'rect': [0.59, 0.05, 0.08, 0.04], 'label': 'Refresh PC'},
            'save': {'rect': [0.68, 0.05, 0.08, 0.04], 'label': 'Save'},
            '3d_view': {'rect': [0.77, 0.05, 0.08, 0.04], 'label': '3D View'},
            'global_3d': {'rect': [0.86, 0.05, 0.08, 0.04], 'label': 'Global 3D'},
            'gui_3d': {'rect': [0.05, 0.10, 0.12, 0.04], 'label': 'GUI 3D Windows'}
        }
        
        # 创建按钮并连接事件
        for name, props in button_props.items():
            ax_btn = plt.axes(props['rect'])
            btn = Button(ax_btn, props['label'])
            self.buttons[name] = btn
            
            # 连接事件
            if name == 'prev_frame':
                btn.on_clicked(self.prev_frame)
            elif name == 'next_frame':
                btn.on_clicked(self.next_frame)
            elif name == 'prev_fish':
                btn.on_clicked(self.prev_fish)
            elif name == 'next_fish':
                btn.on_clicked(self.next_fish)
            elif name == 'prev_kp':
                btn.on_clicked(self.prev_keypoint)
            elif name == 'next_kp':
                btn.on_clicked(self.next_keypoint)
            elif name == 'refresh_pc':
                btn.on_clicked(self.refresh_point_cloud)
            elif name == 'save':
                btn.on_clicked(self.save_keypoints)
            elif name == '3d_view':
                btn.on_clicked(self.visualize_3d)
            elif name == 'global_3d':
                btn.on_clicked(self.visualize_global_3d)
            elif name == 'gui_3d':
                btn.on_clicked(self.start_gui_3d_windows)
        
        # 创建深度调整滑块
        ax_depth = plt.axes([0.2, 0.12, 0.6, 0.03])
        self.depth_slider = Slider(
            ax_depth, 'Depth (mm)', 0, 10000,  # 限制在10米范围内
            valinit=0, valfmt='%d'
        )
        self.depth_slider.on_changed(self.adjust_depth)
        
        # 设置图形属性
        self.fig.canvas.manager.set_window_title('鱼类关键点3D坐标验证工具')
    
    def update_display(self):
        """
        更新显示
        """
        if self.updating_display:
            return
            
        self.updating_display = True
        try:
            if self.ax is None:
                return
                
            self.ax.clear()
            
            # 优先显示去畸变图像
            if self.image_data is not None:
                try:
                    self.ax.imshow(self.image_data)
                except Exception as e:
                    print(f"显示去畸变图像失败: {e}")
                    # 如果图像显示失败，显示深度图
                    if self.depth_data is not None:
                        depth_display = np.clip(self.depth_data, 0, 10000) / 10000
                        self.ax.imshow(depth_display, cmap='jet')
            else:
                # 如果没有图像数据，显示深度图
                if self.depth_data is not None:
                    depth_display = np.clip(self.depth_data, 0, 10000) / 10000
                    self.ax.imshow(depth_display, cmap='jet')
                else:
                    self.ax.text(0.5, 0.5, '无法加载图像和深度数据', 
                                horizontalalignment='center', 
                                verticalalignment='center',
                                transform=self.ax.transAxes)
                    self.fig.canvas.draw_idle()
                    return
            
            # 绘制关键点
            if self.keypoint_names:
                for i, (name, kp) in enumerate(self.keypoints.items()):
                    color = 'red' if i == self.current_kp_idx else 'blue'
                    self.ax.plot(kp[0], kp[1], 'o', color=color, markersize=8, markeredgewidth=2)
                    self.ax.text(kp[0]+10, kp[1]+10, f"{name}\n{kp[2]:.0f}mm", 
                                color=color, fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                # 高亮当前关键点
                if self.current_kp_idx < len(self.keypoint_names):
                    current_kp_name = self.keypoint_names[self.current_kp_idx]
                    current_kp = self.keypoints[current_kp_name]
                    self.ax.plot(current_kp[0], current_kp[1], 'o', color='yellow', 
                               markersize=12, markeredgewidth=3, markeredgecolor='red')
                
                # 更新标题
                frame_name = self.frames[self.current_frame_idx] if self.frames else "未知"
                fish_name = self.fish_names[self.current_fish_idx] if self.fish_names else "未知"
                current_kp_name = self.keypoint_names[self.current_kp_idx] if self.keypoint_names else "未知"
                current_depth = self.keypoints[current_kp_name][2] if self.keypoint_names else 0
                
                self.ax.set_title(f"Frame: {frame_name} | Fish: {fish_name} | Keypoint: {current_kp_name} (Depth: {current_depth:.2f}mm)",
                                 fontsize=14, weight='bold')
                
                # 更新滑块
                if self.keypoint_names:
                    self.updating_slider = True
                    self.depth_slider.set_val(current_depth)
                    self.updating_slider = False
            
            self.ax.set_xlabel('X (像素坐标)')
            self.ax.set_ylabel('Y (像素坐标)')
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"更新显示失败: {e}")
        finally:
            self.updating_display = False
    
    def visualize_3d(self, event=None):
        """
        可视化3D点云，使用color_rectify图像为点云着色
        """
        # 保存当前matplotlib窗口状态
        current_fig = plt.gcf()

        try:
            # 临时关闭matplotlib的交互模式，避免GIL冲突
            plt.ioff()

            if OPEN3D_AVAILABLE:
                self._visualize_3d_open3d_with_color_rectify()
            else:
                self._visualize_3d_matplotlib()
        finally:
            # 恢复matplotlib的交互模式
            plt.ion()
            # 重新激活原始窗口
            plt.figure(current_fig.number)
    
    def _visualize_3d_open3d_with_color_rectify(self):
        """
        使用Open3D进行3D可视化，使用color_rectify图像为点云着色
        """
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            
            if self.point_cloud_filtered is not None and len(self.point_cloud_filtered) > 0:
                # 使用与图像完全一致的坐标系：
                # X向右，Y向下，Z向前（深度方向）
                points = self.point_cloud_filtered.copy()
                
                # 为点云着色 - 使用color_rectify图像的真实颜色
                if self.color_rectify_data is not None:
                    height, width = self.color_rectify_data.shape[:2]
                    colors = []
                    valid_points = []
                    
                    for point in points:
                        x, y, z = int(round(point[0])), int(round(point[1])), point[2]
                        # 由于点云x坐标被翻转，在获取图像颜色时需要翻转回原始坐标
                        x_original = width - 1 - x
                        if 0 <= x_original < width and 0 <= y < height:
                            # 获取color_rectify图像颜色（RGB格式）
                            color = self.color_rectify_data[y, x_original] / 255.0
                            colors.append(color)
                            valid_points.append(point)
                    
                    if valid_points:
                        points = np.array(valid_points)
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        print(f"使用color_rectify图像着色点云，包含 {len(points)} 个点")
                    else:
                        # 如果没有有效的颜色点，使用所有点但不着色
                        pcd.points = o3d.utility.Vector3dVector(points)
                        print(f"无法使用color_rectify图像着色，使用默认颜色，包含 {len(points)} 个点")
                else:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    print(f"没有color_rectify图像，使用默认颜色，包含 {len(points)} 个点")
            
            # 创建关键点几何体
            keypoint_mesh_list = []

            # 获取图像尺寸用于坐标翻转
            height, width = self.depth_data.shape

            for name, kp in self.keypoints.items():
                if kp[2] > 0:
                    # 创建球体表示关键点
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)  # 10mm半径
                    # 翻转x坐标以匹配点云显示
                    x_flipped = width - 1 - kp[0]
                    # 翻转z坐标以匹配点云的z轴方向
                    z_flipped = -kp[2]
                    sphere.translate([x_flipped, kp[1], z_flipped])  # 使用翻转后的坐标系匹配点云
                    sphere.paint_uniform_color([0, 1, 0])  # 绿色
                    keypoint_mesh_list.append(sphere)
            
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="3D点云可视化", 
                             width=1200, height=800)
            
            # 保存可视化器引用以便后续更新
            self.vis_window = vis
            
            # 添加几何体
            if len(pcd.points) > 0:
                vis.add_geometry(pcd)
            
            for mesh in keypoint_mesh_list:
                vis.add_geometry(mesh)
            
            # 保存关键点网格引用以便后续更新
            self.keypoint_meshes = {}
            for i, (name, mesh) in enumerate(zip(self.keypoint_names, keypoint_mesh_list)):
                self.keypoint_meshes[name] = mesh
            
            # 设置渲染选项以获得最佳效果
            render_option = vis.get_render_option()
            render_option.point_size = 2.0  # 点大小
            render_option.background_color = np.array([0.05, 0.05, 0.05])  # 深色背景
            render_option.light_on = True
            
            # 设置合适的视角
            view_control = vis.get_view_control()
            
            # 计算点云边界框
            if len(pcd.points) > 0:
                bbox = pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                # 设置视角（修正为与图像一致的视角）
                view_control.set_front([0, -0.5, 1])  # 从Z轴正方向看（匹配翻转后的Z轴）
                view_control.set_lookat(center)
                view_control.set_up([0, 1, 0])  # Y轴向下，匹配图像坐标系
                view_control.set_zoom(0.8)
            else:
                # 默认视角（修正后的视角）
                view_control.set_front([0, -0.5, 1])
                view_control.set_up([0, 1, 0])
            
            print("3D点云可视化说明:")
            print("- 点云使用color_rectify图像的真实颜色着色")
            print("- 绿色球体表示关键点")
            print("- 视角已修正，与原始图像一致: X向右, Y向下, Z向前")
            print("- 交互: 左键旋转, 中键（滚轮）平移, 滚轮缩放")
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
            # 清理引用
            self.vis_window = None
            self.keypoint_meshes = {}
            
        except Exception as e:
            print(f"使用Open3D进行3D可视化失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到matplotlib
            self._visualize_3d_matplotlib()
    
    def visualize_global_3d(self, event=None):
        """
        可视化全局3D点云（0-5米范围），用于对照检查
        """
        # 保存当前matplotlib窗口状态
        current_fig = plt.gcf()

        try:
            # 临时关闭matplotlib的交互模式，避免GIL冲突
            plt.ioff()

            if OPEN3D_AVAILABLE:
                self._visualize_global_3d_open3d()
            else:
                print("Open3D不可用，无法显示全局3D点云")
        finally:
            # 恢复matplotlib的交互模式
            plt.ion()
            # 重新激活原始窗口
            plt.figure(current_fig.number)
    
    def _visualize_global_3d_open3d(self):
        """
        使用Open3D进行全局3D可视化（0-10米范围）
        """
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            
            if self.point_cloud_data is not None and len(self.point_cloud_data) > 0:
                # 过滤0-5米范围的全局点云
                z_coords = self.point_cloud_data[:, 2]
                global_mask = (z_coords >= -10000) & (z_coords <= 0)  # 0-10米范围（z为负值）
                global_points = self.point_cloud_data[global_mask]
                
                if len(global_points) > 0:
                    # 使用与图像完全一致的坐标系
                    points = global_points.copy()
                    
                    # 为点云着色 - 使用color_rectify图像的真实颜色
                    if self.color_rectify_data is not None:
                        height, width = self.color_rectify_data.shape[:2]
                        colors = []
                        valid_points = []
                        
                        for point in points:
                            x, y, z = int(round(point[0])), int(round(point[1])), point[2]
                            # 由于点云x坐标被翻转，在获取图像颜色时需要翻转回原始坐标
                            x_original = width - 1 - x
                            if 0 <= x_original < width and 0 <= y < height:
                                # 获取color_rectify图像颜色（RGB格式）
                                color = self.color_rectify_data[y, x_original] / 255.0
                                colors.append(color)
                                valid_points.append(point)
                        
                        if valid_points:
                            points = np.array(valid_points)
                            pcd.points = o3d.utility.Vector3dVector(points)
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                            print(f"使用color_rectify图像着色全局点云，包含 {len(points)} 个点")
                        else:
                            # 如果没有有效的颜色点，使用所有点但不着色
                            pcd.points = o3d.utility.Vector3dVector(points)
                            print(f"无法使用color_rectify图像着色，使用默认颜色，包含 {len(points)} 个点")
                    else:
                        pcd.points = o3d.utility.Vector3dVector(points)
                        print(f"没有color_rectify图像，使用默认颜色，包含 {len(points)} 个点")
                else:
                    print("全局点云数据为空")
            
            # 创建关键点几何体
            global_keypoint_mesh_list = []

            # 获取图像尺寸用于坐标翻转
            height, width = self.depth_data.shape

            for name, kp in self.keypoints.items():
                if kp[2] > 0:
                    # 创建球体表示关键点
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=15)  # 稍大的半径以便在全局视图中看到
                    # 翻转x坐标以匹配点云显示
                    x_flipped = width - 1 - kp[0]
                    # 翻转z坐标以匹配点云的z轴方向
                    z_flipped = -kp[2]
                    sphere.translate([x_flipped, kp[1], z_flipped])  # 使用翻转后的坐标系匹配点云
                    sphere.paint_uniform_color([1, 0, 0])  # 红色，区别于局部视图
                    global_keypoint_mesh_list.append(sphere)
            
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="全局3D点云可视化 (0-5米)", 
                             width=1200, height=800)
            
            # 保存可视化器引用以便后续更新
            self.global_vis_window = vis
            
            # 添加几何体
            if len(pcd.points) > 0:
                vis.add_geometry(pcd)
            
            for mesh in global_keypoint_mesh_list:
                vis.add_geometry(mesh)
            
            # 保存关键点网格引用以便后续更新
            self.global_keypoint_meshes = {}
            for i, (name, mesh) in enumerate(zip(self.keypoint_names, global_keypoint_mesh_list)):
                self.global_keypoint_meshes[name] = mesh
            
            # 设置渲染选项以获得最佳效果
            render_option = vis.get_render_option()
            render_option.point_size = 1.5  # 稍小的点大小以适应全局视图
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 稍亮的背景
            render_option.light_on = True
            
            # 设置合适的视角
            view_control = vis.get_view_control()
            
            # 计算点云边界框
            if len(pcd.points) > 0:
                bbox = pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                # 设置视角（全局视图，修正为与图像一致的视角）
                view_control.set_front([0, -0.3, 1])  # 从Z轴正方向看（匹配翻转后的Z轴）
                view_control.set_lookat(center)
                view_control.set_up([0, -1, 0])  # Y轴向下，匹配图像坐标系
                view_control.set_zoom(0.3)  # 更小的缩放以看到全局
            else:
                # 默认视角（修正后的视角）
                view_control.set_front([0, -0.3, 1])
                view_control.set_up([0, -1, 0])
                view_control.set_zoom(0.3)
            
            print("全局3D点云可视化说明:")
            print("- 点云使用color_rectify图像的真实颜色着色")
            print("- 红色球体表示关键点（区别于局部视图的绿色）")
            print("- 显示0-10米深度范围内的所有点云")
            print("- 视角已修正，与原始图像一致: X向右, Y向下, Z向前")
            print("- 交互: 左键旋转, 中键（滚轮）平移, 滚轮缩放")
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
            # 清理引用
            self.global_vis_window = None
            self.global_keypoint_meshes = {}
            
        except Exception as e:
            print(f"使用Open3D进行全局3D可视化失败: {e}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"使用matplotlib进行3D可视化失败: {e}")

    def start_gui_3d_windows(self, event=None):
        """
        启动GUI 3D窗口子进程
        """
        if self.gui_process is not None and self.gui_process.is_alive():
            print("GUI 3D窗口已在运行")
            return

        try:
            # 创建通信管道
            parent_conn, child_conn = mp.Pipe()
            self.gui_pipe = parent_conn

            # 准备子进程参数
            gui_args = {
                'depth_file_path': os.path.join(self.depth_root, "left_disp_0.npy"),
                'image_path': "fish_dataset/images/0001751539512357.png",
                'camera_config': self.camera_config,
                'pipe_conn': child_conn,
                'depth_data': self.depth_data,
                'color_rectify_data': self.color_rectify_data,
                'keypoints': self.keypoints,
                'keypoint_names': self.keypoint_names,
                'point_cloud_data': self.point_cloud_data,
                'z_min': self.z_min,
                'z_max': self.z_max
            }

            # 启动子进程
            self.gui_process = mp.Process(
                target=run_gui_visualization_worker,
                args=(gui_args,)
            )
            self.gui_process.start()

            print("GUI 3D窗口子进程已启动")

            # 等待子进程完全初始化
            time.sleep(1.0)  # 增加等待时间
            print("等待子进程初始化完成...")

            # 发送初始化完成信号
            if self.gui_pipe is not None:
                try:
                    self.gui_pipe.send({'type': 'init_complete'})
                    print("已发送初始化完成信号到GUI子进程")
                except Exception as e:
                    print(f"发送初始化信号失败: {e}")

            # 等待子进程响应
            time.sleep(0.5)

            # 发送当前状态
            self._send_gui_fish_update()
            print("已发送初始状态到GUI子进程")

        except Exception as e:
            print(f"启动GUI 3D窗口失败: {e}")
            self.gui_process = None
            self.gui_pipe = None

    def _send_gui_update(self):
        """
        发送更新消息到GUI子进程
        """
        if self.gui_pipe is None:
            return

        try:
            current_kp_name = self.keypoint_names[self.current_kp_idx] if self.keypoint_names else None
            if current_kp_name and current_kp_name in self.keypoints:
                kp = self.keypoints[current_kp_name]

                # 发送关键点更新消息
                message = {
                    'type': 'update_keypoint',
                    'kp_name': current_kp_name,
                    'x': float(kp[0]),
                    'y': float(kp[1]),
                    'z': float(kp[2])
                }

                if self.gui_pipe.poll():
                    # 清空管道中的旧消息
                    while self.gui_pipe.poll():
                        self.gui_pipe.recv()

                self.gui_pipe.send(message)

        except Exception as e:
            print(f"发送GUI更新消息失败: {e}")

    def _send_gui_fish_update(self):
        """
        发送切换鱼类的消息到GUI子进程
        """
        if self.gui_pipe is None:
            return

        try:
            # 发送切换鱼类的消息
            message = {
                'type': 'switch_fish',
                'keypoints': {name: [float(x), float(y), float(z)] for name, (x, y, z) in self.keypoints.items()},
                'keypoint_names': [str(name) for name in self.keypoint_names],
                'z_min': float(self.z_min),
                'z_max': float(self.z_max)
            }

            if self.gui_pipe.poll():
                # 清空管道中的旧消息
                while self.gui_pipe.poll():
                    self.gui_pipe.recv()

            self.gui_pipe.send(message)
            print(f"已发送切换鱼类消息: {len(self.keypoint_names)} 个关键点, 范围 [{self.z_min}, {self.z_max}]")
            print(f"关键点名称: {self.keypoint_names}")
            print(f"第一个关键点: {list(self.keypoints.items())[0] if self.keypoints else '无'}")

        except Exception as e:
            print(f"发送GUI鱼类更新消息失败: {e}")
            import traceback
            traceback.print_exc()

    def _send_gui_point_cloud_update(self):
        """
        发送点云范围更新消息到GUI子进程
        """
        if self.gui_pipe is None:
            return

        try:
            # 发送点云范围更新的消息
            message = {
                'type': 'update_point_cloud_range',
                'z_min': self.z_min,
                'z_max': self.z_max
            }

            if self.gui_pipe.poll():
                # 清空管道中的旧消息
                while self.gui_pipe.poll():
                    self.gui_pipe.recv()

            self.gui_pipe.send(message)

        except Exception as e:
            print(f"发送GUI点云更新消息失败: {e}")

    def save_keypoints(self, event=None):
        """
        保存关键点 - 将backup文件放入backup文件夹，原名存储
        """
        if not self.annotation_data:
            print("没有加载标注数据，无法保存")
            return

        # 获取当前帧对应的标注文件路径
        if not self.frames:
            print("没有当前帧信息，无法确定保存路径")
            return

        current_frame_name = self.frames[self.current_frame_idx]
        base_name = os.path.splitext(current_frame_name)[0]

        # 构建保存路径
        annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme')
        backup_root = os.path.join(annotations_root, 'backup')
        current_annotation_file = os.path.join(annotations_root, f"{base_name}.json")
        backup_annotation_file = os.path.join(backup_root, f"{base_name}.json")

        try:
            # 确保backup文件夹存在
            os.makedirs(backup_root, exist_ok=True)

            # 创建备份文件（如果原文件存在）
            if os.path.exists(current_annotation_file):
                import shutil
                shutil.copy2(current_annotation_file, backup_annotation_file)
                print(f"已创建备份文件: {backup_annotation_file}")
            else:
                print(f"原标注文件不存在: {current_annotation_file}，将直接保存")

            # 更新标注数据
            for shape in self.annotation_data['shapes']:
                if shape['shape_type'] == 'point':
                    name = shape['label']
                    # 查找对应的关键点（可能在任何鱼类中）
                    for fish_keypoints in self.fish_keypoints.values():
                        if name in fish_keypoints:
                            depth_value = fish_keypoints[name][2]
                            # 在描述中添加深度信息
                            if 'description' in shape:
                                shape['description'] = f"depth: {depth_value:.2f}mm"
                            else:
                                shape['description'] = f"depth: {depth_value:.2f}mm"
                            break

            # 保存文件到当前标注文件路径
            with open(current_annotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_data, f, indent=4, ensure_ascii=False)

            print(f"成功保存关键点深度值到: {current_annotation_file}")

        except Exception as e:
            print(f"保存标注文件失败: {e}")
    
    def run(self):
        """
        运行交互式验证工具
        """
        print("启动交互式验证工具")
        
        # 初始显示
        self.update_display()
        
        # 显示图形界面
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"显示图形界面失败: {e}")
        finally:
            # 清理GUI子进程
            self._cleanup_gui_process()

    def _cleanup_gui_process(self):
        """
        清理GUI子进程
        """
        if self.gui_process is not None and self.gui_process.is_alive():
            try:
                # 发送关闭消息
                if self.gui_pipe is not None:
                    self.gui_pipe.send({'type': 'shutdown'})

                # 等待子进程结束
                self.gui_process.join(timeout=2.0)

                if self.gui_process.is_alive():
                    print("强制终止GUI子进程")
                    self.gui_process.terminate()
                    self.gui_process.join()

                print("GUI子进程已清理")

            except Exception as e:
                print(f"清理GUI子进程失败: {e}")
            finally:
                self.gui_process = None
                self.gui_pipe = None


def main():
    parser = argparse.ArgumentParser(description='鱼类关键点3D坐标验证和调整工具')
    parser.add_argument('--dataset_root', type=str,
                       default='fish_dataset',
                       help='数据集根目录')
    parser.add_argument('--camera_config', type=str,
                       default='fish_dataset/camera_configs/mocha_stereo_params.yaml',
                       help='相机配置文件路径')
    parser.add_argument('--default_annotation', type=str,
                       default='fish_dataset/annotations/labelme/fishdata.json',
                       help='默认标注文件路径（当对应帧的标注文件不存在时使用）')

    args = parser.parse_args()

    # 检查数据集根目录
    if not os.path.exists(args.dataset_root):
        print(f"数据集根目录不存在: {args.dataset_root}")
        return False

    # 构建各个子目录路径
    depth_root = os.path.join(args.dataset_root, 'depths')
    images_root = os.path.join(args.dataset_root, 'images')
    annotations_root = os.path.join(args.dataset_root, 'annotations', 'labelme')

    # 检查必要的目录
    if not os.path.exists(depth_root):
        print(f"警告: 深度图目录不存在: {depth_root}")
    else:
        print(f"深度图目录: {depth_root}")

    if not os.path.exists(images_root):
        print(f"图像目录不存在: {images_root}")
        return False
    else:
        print(f"图像目录: {images_root}")

    if not os.path.exists(annotations_root):
        print(f"标注目录不存在: {annotations_root}，将创建")
        os.makedirs(annotations_root, exist_ok=True)

    if not os.path.exists(args.camera_config):
        print(f"相机配置文件不存在: {args.camera_config}")
        return False

    # 检查默认标注文件
    if not os.path.exists(args.default_annotation):
        print(f"警告: 默认标注文件不存在: {args.default_annotation}")

    try:
        verifier = Fish3DKeypointVerifier(
            args.default_annotation,  # 传递默认标注文件路径
            depth_root,
            args.camera_config
        )
        verifier.run()
        return True
    except Exception as e:
        print(f"运行工具时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_gui_visualization_worker(args):
    """
    GUI可视化子进程工作函数（使用传统Open3D可视化器）
    """
    try:
        import open3d as o3d
        import threading
        import time

        # 设置Open3D环境变量以避免GUI上下文问题
        os.environ['OPEN3D_DISABLE_GUI'] = '0'  # 启用GUI
        os.environ['OPEN3D_HEADLESS'] = '0'     # 非无头模式
        os.environ['DISPLAY'] = ':0'            # 设置显示（Windows不需要但保持兼容性）

        # 获取参数
        depth_data = args['depth_data']
        color_rectify_data = args['color_rectify_data']
        point_cloud_data = args['point_cloud_data']
        keypoints = args['keypoints']
        keypoint_names = args['keypoint_names']
        pipe_conn = args['pipe_conn']
        z_min = args['z_min']
        z_max = args['z_max']

        height, width = depth_data.shape

        # 全局变量用于存储当前状态
        current_keypoints = keypoints.copy()
        current_keypoint_names = keypoint_names.copy()
        current_z_min = z_min
        current_z_max = z_max

        # 在消息处理线程中使用的变量（使用列表包装以便修改）
        thread_current_keypoints = [current_keypoints.copy()]
        thread_current_keypoint_names = [current_keypoint_names.copy()]
        thread_current_z_min = [current_z_min]
        thread_current_z_max = [current_z_max]

        # 存储几何体引用
        local_pcd_geometry = None

        # 创建点云函数（复用主进程逻辑）
        def create_filtered_point_cloud(point_cloud_data, z_range=None):
            """根据z范围过滤点云"""
            if point_cloud_data is None:
                return None

            z_coords = point_cloud_data[:, 2]
            if z_range is not None:
                z_min_filter, z_max_filter = z_range
                mask = (z_coords >= z_min_filter) & (z_coords <= z_max_filter)
                filtered_points = point_cloud_data[mask]
            else:
                filtered_points = point_cloud_data

            # 为点云着色
            colors = []
            if color_rectify_data is not None:
                for point in filtered_points:
                    x, y, z = int(round(point[0])), int(round(point[1])), point[2]
                    x_original = width - 1 - x
                    if 0 <= x_original < width and 0 <= y < height:
                        color = color_rectify_data[y, x_original] / 255.0
                        colors.append(color)
                    else:
                        colors.append([0.5, 0.5, 0.5])
            else:
                colors = [[0.5, 0.5, 0.5]] * len(filtered_points)

            # 创建Open3D点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            return pcd

        # 创建全局点云（0-10米范围，固定）
        global_pcd = create_filtered_point_cloud(point_cloud_data, z_range=(-10000, 0))  # 0-10米深度范围

        # 关键点球体管理
        local_keypoints = {}
        global_keypoints = {}
        local_pcd_geometry = None

        # 保存用户视角设置（在GUI子进程全局作用域中定义）
        saved_local_view_params = [None]  # 使用列表包装以便修改
        saved_global_view_params = [None]  # 使用列表包装以便修改

        def update_keypoint(local_vis, global_vis, kp_name, x, y, z):

            try:
                # 计算翻转后的坐标
                x_flipped = width - 1 - x
                z_flipped = -z
                position = np.array([x_flipped, y, z_flipped])

                print(f"更新关键点 {kp_name}: 原始坐标({x:.1f}, {y:.1f}, {z:.1f}) -> 3D坐标({x_flipped:.1f}, {y:.1f}, {z_flipped:.1f})")

                # 更新局部窗口 - 直接修改现有几何体的变换，避免重置视角
                if local_vis is not None:
                    try:
                        if kp_name in local_keypoints:
                            # 直接更新现有球体的位置
                            sphere = local_keypoints[kp_name]
                            current_center = sphere.get_center()
                            translation = position - current_center
                            sphere.translate(translation)
                            local_vis.update_geometry(sphere)
                            print(f"更新局部关键点 {kp_name} 位置成功")
                        else:
                            # 如果不存在，创建新的
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                            sphere.translate(position)
                            sphere.paint_uniform_color([0, 1, 0])  # 绿色
                            local_keypoints[kp_name] = sphere
                            local_vis.add_geometry(sphere)
                            print(f"创建新的局部关键点 {kp_name}")
                    except Exception as e:
                        print(f"更新局部窗口关键点 {kp_name} 失败: {e}")

                # 更新全局窗口 - 直接修改现有几何体的变换，避免重置视角
                if global_vis is not None:
                    try:
                        if kp_name in global_keypoints:
                            # 直接更新现有球体的位置
                            sphere = global_keypoints[kp_name]
                            current_center = sphere.get_center()
                            translation = position - current_center
                            sphere.translate(translation)
                            global_vis.update_geometry(sphere)
                            print(f"更新全局关键点 {kp_name} 位置成功")
                        else:
                            # 如果不存在，创建新的
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=15)
                            sphere.translate(position)
                            sphere.paint_uniform_color([1, 0, 0])  # 红色
                            global_keypoints[kp_name] = sphere
                            global_vis.add_geometry(sphere)
                            print(f"创建新的全局关键点 {kp_name}")
                    except Exception as e:
                        print(f"更新全局窗口关键点 {kp_name} 失败: {e}")

            except Exception as e:
                print(f"更新关键点 {kp_name} 时发生异常: {e}")
                import traceback
                traceback.print_exc()

        def update_local_point_cloud(local_vis):
            """更新局部点云显示"""
            nonlocal local_pcd_geometry

            # 移除旧的局部点云
            if local_pcd_geometry is not None:
                try:
                    local_vis.remove_geometry(local_pcd_geometry)
                except:
                    pass  # 几何体可能已经被移除

            # 创建新的过滤点云
            local_pcd = create_filtered_point_cloud(point_cloud_data, z_range=(thread_current_z_min[0], thread_current_z_max[0]))
            if local_pcd is not None and len(local_pcd.points) > 0:
                local_vis.add_geometry(local_pcd)
                local_pcd_geometry = local_pcd
                print(f"更新局部点云: {len(local_pcd.points)} 个点")
            else:
                local_pcd_geometry = None
                print("局部点云为空")

        def update_all_keypoints(local_vis, global_vis):
            """更新所有关键点"""
            try:
                print(f"开始更新关键点，清空 {len(local_keypoints)} 个旧关键点")

                # 先清空所有现有的关键点
                old_local_kps = list(local_keypoints.keys())
                old_global_kps = list(global_keypoints.keys())
                print(f"需要移除的局部关键点: {old_local_kps}")
                print(f"需要移除的全局关键点: {old_global_kps}")

                # 移除所有旧的关键点
                for kp_name in old_local_kps:
                    try:
                        if kp_name in local_keypoints:
                            local_vis.remove_geometry(local_keypoints[kp_name])
                            print(f"成功移除局部关键点 {kp_name}")
                    except Exception as e:
                        print(f"移除局部关键点 {kp_name} 失败: {e}")

                for kp_name in old_global_kps:
                    try:
                        if kp_name in global_keypoints:
                            global_vis.remove_geometry(global_keypoints[kp_name])
                            print(f"成功移除全局关键点 {kp_name}")
                    except Exception as e:
                        print(f"移除全局关键点 {kp_name} 失败: {e}")

                # 清空字典
                local_keypoints.clear()
                global_keypoints.clear()
                print("关键点字典已清空")

                # 添加新的关键点
                valid_kps = 0
                current_names = list(thread_current_keypoints[0].keys())
                print(f"准备添加 {len(current_names)} 个新关键点: {current_names}")

                for i, kp_name in enumerate(current_names):
                    if kp_name in thread_current_keypoints[0]:
                        kp = thread_current_keypoints[0][kp_name]
                        if kp[2] > 0:
                            print(f"准备添加关键点 {i+1}/{len(current_names)}: {kp_name} at ({kp[0]:.1f}, {kp[1]:.1f}, {kp[2]:.1f})")
                            try:
                                update_keypoint(local_vis, global_vis, kp_name, kp[0], kp[1], kp[2])
                                valid_kps += 1
                                print(f"成功添加关键点 {kp_name}")
                            except Exception as e:
                                print(f"添加关键点 {kp_name} 失败: {e}")
                        else:
                            print(f"跳过深度为0的关键点: {kp_name} at ({kp[0]:.1f}, {kp[1]:.1f}, {kp[2]:.1f})")
                    else:
                        print(f"关键点名称 {kp_name} 不在关键点字典中")

                print(f"成功添加了 {valid_kps} 个新关键点 (总共 {len(current_names)} 个关键点)")
                print(f"最终local_keypoints数量: {len(local_keypoints)}")
                print(f"最终global_keypoints数量: {len(global_keypoints)}")

                # 注意：视角重置现在在消息处理线程中进行，以保持用户的视角设置

            except Exception as e:
                print(f"update_all_keypoints 发生异常: {e}")
                import traceback
                traceback.print_exc()

        # 消息处理线程
        def message_handler():
            # 写入调试文件
            debug_log_path = os.path.join(tempfile.gettempdir(), 'gui_debug.log')
            with open(debug_log_path, 'a') as f:
                f.write("GUI子进程消息处理线程已启动\n")
            print("GUI子进程消息处理线程已启动")
            message_count = 0
            while True:
                if pipe_conn.poll():
                    try:
                        message = pipe_conn.recv()
                        message_count += 1
                        debug_msg = f"GUI子进程接收到消息 #{message_count}: {message['type']}\n"
                        with open(debug_log_path, 'a') as f:
                            f.write(debug_msg)
                        print(f"GUI子进程接收到消息 #{message_count}: {message['type']}")
                        if message['type'] == 'update_keypoint':
                            update_keypoint(local_vis, global_vis,
                                message['kp_name'],
                                message['x'],
                                message['y'],
                                message['z']
                            )
                        elif message['type'] == 'init_complete':
                            print("GUI子进程接收到初始化完成信号")
                        elif message['type'] == 'switch_fish':
                            print("GUI子进程开始处理switch_fish消息")

                            # 在切换鱼类前，保存用户的当前视角设置
                            try:
                                # 使用Open3D的相机参数保存方法
                                saved_local_view_params[0] = local_vis.get_view_control().convert_to_pinhole_camera_parameters()
                                saved_global_view_params[0] = global_vis.get_view_control().convert_to_pinhole_camera_parameters()
                                print("已保存用户的视角设置")
                            except Exception as e:
                                print(f"保存用户视角失败: {e}")
                                saved_local_view_params[0] = None
                                saved_global_view_params[0] = None

                            # 更新线程变量（修改列表内容而不是重新赋值）
                            import numpy as np
                            thread_current_keypoints[0] = {name: np.array([x, y, z]) for name, (x, y, z) in message['keypoints'].items()}
                            thread_current_keypoint_names[0] = [str(name) for name in message['keypoint_names']]
                            thread_current_z_min[0] = message['z_min']
                            thread_current_z_max[0] = message['z_max']

                            print(f"GUI子进程接收切换鱼类消息: keypoints={len(thread_current_keypoints[0])}, names={len(thread_current_keypoint_names[0])}")
                            print(f"新关键点列表: {thread_current_keypoint_names[0]}")
                            print(f"第一个关键点: {list(thread_current_keypoints[0].items())[0] if thread_current_keypoints[0] else '无'}")
                            print("调用update_all_keypoints...")
                            update_all_keypoints(local_vis, global_vis)
                            print("关键点更新完成")
                            print("开始更新局部点云...")
                            update_local_point_cloud(local_vis)
                            print("局部点云更新完成")

                            # 切换鱼类完成后，恢复用户的视角设置
                            try:
                                if saved_local_view_params[0] is not None:
                                    # 恢复局部视角到用户之前的设置
                                    local_vis.get_view_control().convert_from_pinhole_camera_parameters(saved_local_view_params[0])
                                    print("已恢复局部视角设置")

                                if saved_global_view_params[0] is not None:
                                    # 恢复全局视角到用户之前的设置
                                    global_vis.get_view_control().convert_from_pinhole_camera_parameters(saved_global_view_params[0])
                                    print("已恢复全局视角设置")

                            except Exception as e:
                                print(f"恢复用户视角失败: {e}")

                            print(f"GUI子进程切换完成")
                        elif message['type'] == 'update_point_cloud_range':
                            thread_current_z_min[0] = message['z_min']
                            thread_current_z_max[0] = message['z_max']
                            update_local_point_cloud(local_vis)
                            print(f"GUI子进程更新点云范围: [{thread_current_z_min[0]}, {thread_current_z_max[0]}]")
                        elif message['type'] == 'shutdown':
                            break
                    except Exception as e:
                        print(f"GUI子进程处理消息失败: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                time.sleep(0.05)  # 减少睡眠时间，提高响应性

        # 创建可视化器（使用VisualizerWithKeyCallback以避免GUI上下文问题）
        local_vis = o3d.visualization.VisualizerWithKeyCallback()
        local_vis.create_window("局部3D点云", 800, 600)

        global_vis = o3d.visualization.VisualizerWithKeyCallback()
        global_vis.create_window("全局3D点云 (0-10米)", 800, 600)

        # 设置非阻塞渲染模式
        local_vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        global_vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        global_vis.add_geometry(global_pcd)

        # 初始化局部点云
        update_local_point_cloud(local_vis)
        print(f"GUI子进程初始化完成，局部点云: {len(local_pcd_geometry.points) if local_pcd_geometry else 0} 个点")

        # 设置渲染选项（只在初始化时设置一次）
        for vis in [local_vis, global_vis]:
            render_option = vis.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.array([0.05, 0.05, 0.05])
            render_option.light_on = True

        # 只在初始化时设置视角一次，并保存初始设置
        # 由于我们翻转了Z坐标系，调整视角以匹配原始图像视角
        local_view_control = local_vis.get_view_control()
        local_view_control.set_front([0, -0.5, 1])  # 从Z轴正方向看（匹配翻转后的Z轴）
        local_view_control.set_lookat([0, 0, 0])
        local_view_control.set_up([0, -1, 0])  # Y轴向下，匹配图像坐标系
        local_view_control.set_zoom(0.8)

        global_view_control = global_vis.get_view_control()
        global_view_control.set_front([0, -0.3, 1])  # 从Z轴正方向看（匹配翻转后的Z轴）
        global_view_control.set_lookat([0, 0, 0])
        global_view_control.set_up([0, -1, 0])  # Y轴向下，匹配图像坐标系
        global_view_control.set_zoom(0.3)

        # 保存初始视角参数（更新为修正后的视角）
        saved_local_view_params = {
            'front': [0, -0.5, 1],
            'lookat': [0, 0, 0],
            'up': [0, -1, 0],
            'zoom': 0.8
        }
        saved_global_view_params = {
            'front': [0, -0.3, 1],
            'lookat': [0, 0, 0],
            'up': [0, -1, 0],
            'zoom': 0.3
        }

        # 初始化关键点
        update_all_keypoints(local_vis, global_vis)

        # 可视化器创建完成后，再启动消息处理线程
        msg_thread = threading.Thread(target=message_handler, daemon=True)
        msg_thread.start()
        print("GUI子进程消息处理线程启动完成")

        print("GUI 3D窗口子进程已启动")
        print("两个窗口现在都可以独立交互，支持切换鱼类和点云范围更新")

        # 运行可视化器
        try:
            while True:
                local_vis.poll_events()
                local_vis.update_renderer()

                global_vis.poll_events()
                global_vis.update_renderer()

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            local_vis.destroy_window()
            global_vis.destroy_window()

    except Exception as e:
        print(f"GUI子进程运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("hello world")
    success = main()
    sys.exit(0 if success else 1)