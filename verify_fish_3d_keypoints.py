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
    from utils.camera_utils import convert_joints_to_camera_coords, project_left_to_right, project_keypoints_left_to_right
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

        # 新增：维护关键点到鱼类的映射关系，用于正确保存
        self.keypoint_to_fish_map = {}  # (x, y) -> fish_name
        
        # 新增：维护被reset标记的关键点，这些关键点不应该被自动填充深度值
        self.reset_keypoints = {}  # fish_name -> set(kp_name)
        
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
        self.right_image_data = None  # 右图数据
        self.color_rectify_data = None  # 用于点云着色的彩色图像

        # 标签显示控制
        self.show_labels = True  # 默认显示标签

        # 同步深度调整模式
        self.sync_depth_mode = False  # 同步调整当前鱼所有关键点
        self.last_depth_value = 0.0   # 记录上次深度值用于计算delta

        # 立体视觉重叠区域（SGBM算法计算的硬编码值）
        self.OVERLAP_X_MIN = 256
        self.OVERLAP_X_MAX = 1439

        # 右图拖动状态
        self.dragging_right = False  # 是否正在拖动右图关键点
        self.drag_kp_name = None     # 正在拖动的关键点名称
        self.drag_start_x = None     # 拖动起始x坐标
        self.drag_mode_global = False  # 拖动模式：False=单个（仅当前点），True=全局（任意点）

        # 右图关键点 group_id 状态
        # 格式: {fish_id: {kp_name: None or 1}}
        # None = 可见（默认），1 = 关键点不存在（SGBM筛除或用户手动设置）
        self.right_kp_group_ids = {}

        # 右图整鱼抛弃状态：True = 用户手动抛弃该鱼的右图构建（Delete Right）
        # 用户拖动深度滑块赋值后自动清除，下次 Save 时恢复构建
        self.right_fish_abandoned = {}  # {fish_id: bool}

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
    
    def _parse_depth_from_description(self, description: str) -> tuple:
        """
        从description字段解析深度值
        
        Args:
            - description (str): 描述字段，格式如 "depth: 524.99mm" 或 "depth: 0.00mm (reset - invalid depth)"
            
        Returns:
            - (tuple) (深度值（毫米）, 是否为重置标记)
                - 深度值: 浮点数，解析失败返回0.0
                - 是否为重置标记: bool，True表示用户主动重置为无效深度
        
        Notes:
            - 只搜索label为关键点的深度，如果label为fish（或者group_id为0），则表示当前字段为鱼的框，description字段自然是空的
            - 如果description包含"reset"标记，表示用户主动标记为无效深度，不应该被自动填充
        """
        if not description:
            return 0.0, False
        
        try:
            # 检查是否包含reset标记
            import re
            is_reset = 'reset' in description.lower()
            
            # 匹配格式: "depth: XXX.XXmm" 或 "depth: XXXmm"
            match = re.search(r'depth:\s*([\d.]+)\s*mm', description, re.IGNORECASE)
            if match:
                depth_value = float(match.group(1))
                return depth_value, is_reset
            else:
                return 0.0, is_reset
        except Exception as e:
            print(f"解析description失败: {description}, 错误: {e}")
            return 0.0, False
        
    def _parse_keypoints_by_group_id(self):
        """
        按照group_id分配逻辑解析关键点，并建立关键点到鱼的映射
        group_id: 0 表示一条新的鱼
        group_id: null 表示属于当前鱼的关键点
        """
        fish_keypoints = {}
        keypoint_to_fish_map = {}  # (x, y) -> fish_name
        reset_keypoints = {}  # fish_name -> set(kp_name)
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
                reset_keypoints[current_fish_id] = set()  # 初始化该鱼的reset集合
                print(f"找到新的鱼: {current_fish_id}")

            # 如果是关键点且group_id为null，分配给当前鱼
            elif shape['shape_type'] == 'point' and group_id is None:
                if current_fish_id is not None:
                    point = shape['points'][0]
                    x, y = point[0], point[1]
                    
                    # 尝试从description字段解析深度值
                    description = shape.get('description', '')
                    depth_z, is_reset = self._parse_depth_from_description(description)
                    
                    # 如果description中没有深度值（返回0.0）且不是reset标记，后续会从深度图获取
                    # 如果是reset标记，则保持0.0不变
                    fish_keypoints[current_fish_id][shape['label']] = np.array([x, y, depth_z], dtype=np.float32)
                    
                    # 建立坐标到鱼的映射
                    keypoint_to_fish_map[(x, y)] = current_fish_id
                    
                    # 如果是reset标记，记录到reset_keypoints集合中
                    if is_reset:
                        reset_keypoints[current_fish_id].add(shape['label'])
                        print(f"将关键点 '{shape['label']}' 分配给 {current_fish_id}, 深度: {depth_z:.2f}mm (已重置为无效)")
                    else:
                        print(f"将关键点 '{shape['label']}' 分配给 {current_fish_id}, 深度: {depth_z:.2f}mm")
                else:
                    print(f"警告: 关键点 '{shape['label']}' 没有对应的鱼，将被忽略")

            # 其他情况（如group_id不为0或null的矩形）
            else:
                print(f"忽略形状: {shape['label']} (类型: {shape['shape_type']}, group_id: {group_id})")

        # 验证分配结果
        if fish_keypoints:
            print(f"成功解析 {len(fish_keypoints)} 条鱼的关键点:")
            for fish_name, kps in fish_keypoints.items():
                reset_count = len(reset_keypoints.get(fish_name, set()))
                print(f"  {fish_name}: {len(kps)} 个关键点 - {list(kps.keys())} (其中 {reset_count} 个已重置)")
        else:
            print("警告: 没有找到任何鱼的关键点")

        return fish_keypoints, keypoint_to_fish_map, reset_keypoints

    def _get_frames(self):
        """
        获取所有帧的文件名 - 扫描images文件夹中的所有png文件
        只包含那些有对应深度文件和标注文件的图像
        """
        frames = []
        try:
            # 构建各个文件夹路径
            images_root = os.path.join(os.path.dirname(self.depth_root), 'images', 'left')
            annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme', 'left')

            if os.path.exists(images_root):
                # 扫描images/left文件夹中的所有png文件
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
        images_root = os.path.join(os.path.dirname(self.depth_root), 'images', 'left')
        annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme', 'left')

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

        # 读取右图
        try:
            right_image_path = os.path.join(os.path.dirname(self.depth_root), 'images', 'right', current_frame_name)
            if os.path.exists(right_image_path):
                self.right_image_data = cv2.imread(right_image_path)
                self.right_image_data = cv2.cvtColor(self.right_image_data, cv2.COLOR_BGR2RGB)
                print(f"成功加载右图: {right_image_path}, 尺寸: {self.right_image_data.shape}")
            else:
                print(f"右图文件不存在: {right_image_path}")
                self.right_image_data = None
        except Exception as e:
            print(f"加载右图失败: {e}")
            self.right_image_data = None

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
                self.fish_keypoints, self.keypoint_to_fish_map, self.reset_keypoints = self._parse_keypoints_by_group_id()
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
                    self.fish_keypoints, self.keypoint_to_fish_map, self.reset_keypoints = self._parse_keypoints_by_group_id()
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
                    self.reset_keypoints = {}
        except Exception as e:
            print(f"加载标注文件失败: {e}")
            self.annotation_data = None
            self.fish_keypoints = {}
            self.fish_names = []
            self.current_fish_idx = -1
            self.keypoints = {}
            self.keypoint_names = []
            self.reset_keypoints = {}

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

        # 初始化/加载右图关键点 group_id 状态
        self._load_right_kp_group_ids(base_name)

        # 加载点云数据
        self._load_point_cloud_data(depth_file_path)

        # 更新显示（确保初始化时绘制竖线等UI元素）
        self.update_display()

    def _load_right_kp_group_ids(self, base_name: str):
        """
        初始化或从已有右图 JSON 中加载每条鱼的关键点 group_id 状态和整鱼抛弃状态。

        若右图 JSON 存在：
          - 从中读取各关键点的 group_id
          - 若左图中某条鱼在右图 JSON 中没有对应 bbox，推断为被抛弃，设 right_fish_abandoned=True
        若右图 JSON 不存在：全部初始化为默认值（group_id=None，abandoned=False）
        """
        self.right_kp_group_ids = {}
        self.right_fish_abandoned = {}

        right_json_path = os.path.join(
            os.path.dirname(self.depth_root), 'annotations', 'labelme', 'right', f"{base_name}.json"
        )

        if os.path.exists(right_json_path):
            try:
                with open(right_json_path, 'r', encoding='utf-8') as f:
                    right_data = json.load(f)

                # 解析右图 JSON 中出现的所有 fish_id（按 bbox 顺序编号）
                fish_count = 0
                current_fish_id = None
                right_fish_ids_in_json = set()

                for shape in right_data.get('shapes', []):
                    gid = shape.get('group_id')
                    if shape['shape_type'] == 'rectangle' and shape['label'] == 'fish' and gid == 0:
                        fish_count += 1
                        current_fish_id = f"fish_{fish_count}"
                        right_fish_ids_in_json.add(current_fish_id)
                        self.right_kp_group_ids[current_fish_id] = {}
                    elif shape['shape_type'] == 'point' and current_fish_id is not None:
                        kp_name = shape['label']
                        self.right_kp_group_ids[current_fish_id][kp_name] = gid

                # 推断 right_fish_abandoned：左图中有但右图 JSON 中没有出现的鱼 = 曾被抛弃
                for fish_id in self.fish_keypoints:
                    if fish_id in right_fish_ids_in_json:
                        self.right_fish_abandoned[fish_id] = False
                    else:
                        self.right_fish_abandoned[fish_id] = True
                        print(f"{fish_id} 未出现在右图JSON中，推断为已抛弃（Delete Right）")

                print(f"从右图JSON加载 group_id 状态: {right_json_path}")
            except Exception as e:
                print(f"读取右图JSON失败，使用默认值: {e}")
                self._init_right_kp_group_ids_default()
        else:
            self._init_right_kp_group_ids_default()

    def _init_right_kp_group_ids_default(self):
        """将所有鱼的所有关键点 group_id 初始化为 None（可见），right_fish_abandoned 全部初始化为 False"""
        self.right_kp_group_ids = {}
        self.right_fish_abandoned = {}
        for fish_id, kps in self.fish_keypoints.items():
            self.right_kp_group_ids[fish_id] = {kp_name: None for kp_name in kps}
            self.right_fish_abandoned[fish_id] = False

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
                    # 检查是否是被reset标记的关键点
                    is_reset_kp = fish_name in self.reset_keypoints and kp_name in self.reset_keypoints[fish_name]
                    
                    if is_reset_kp:
                        # 如果是reset标记的关键点，强制保持为0，不从深度图读取
                        self.keypoints[kp_name][2] = 0.0
                    elif saved_depths[fish_name][kp_name] != 0.0:
                        # 如果之前有标注过深度值（不为0），则保留标注后的值
                        self.keypoints[kp_name][2] = saved_depths[fish_name][kp_name]
                    # 否则使用从深度图读取的新值（已在_update_keypoint_depths中更新）
        
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

        # 更新同步模式的基准深度值
        if self.keypoint_names:
            current_kp_name = self.keypoint_names[self.current_kp_idx]
            self.last_depth_value = self.keypoints[current_kp_name][2]

        # 不再更新关键点深度值，保留用户调整的值
        # self._update_keypoint_depths()
        # 更新点云过滤范围
        self._filter_point_cloud()

        # 发送切换鱼类的消息到GUI子进程
        self._send_gui_fish_update()

        # 实时更新3D可视化窗口中的所有关键点位置（如果窗口存在）
        self._update_all_3d_keypoint_positions()

        # 实时更新全局点云窗口中的所有关键点位置（如果窗口存在）
        self._update_all_global_3d_keypoint_positions()
    
    def adjust_depth(self, val):
        """
        调整当前关键点的深度值。
        同步模式下会同时调整当前鱼的所有关键点。
        若调整后有任意关键点深度 > 0，自动清除该鱼的右图抛弃标记。
        """
        if self.updating_slider or not self.keypoint_names:
            return

        current_kp_name = self.keypoint_names[self.current_kp_idx]
        current_fish_name = self.fish_names[self.current_fish_idx] if self.fish_names else None

        if self.sync_depth_mode:
            # 同步模式：计算深度变化量并应用到所有关键点
            delta = float(val) - self.last_depth_value
            print(f"同步调整所有关键点深度，delta: {delta:.1f}mm")
            for kp_name in self.keypoint_names:
                old_depth = self.keypoints[kp_name][2]
                new_depth = max(0.0, min(10000.0, old_depth + delta))
                self.keypoints[kp_name][2] = new_depth
            # 更新基准值
            self.last_depth_value = float(val)
        else:
            # 单点模式：只调整当前关键点
            print(f"调整关键点 {current_kp_name} 的深度值为: {val}")
            self.keypoints[current_kp_name][2] = float(val)

        # 若当前鱼有任意关键点深度 > 0，自动解除右图抛弃标记
        if current_fish_name and self.right_fish_abandoned.get(current_fish_name, False):
            has_valid = any(kp[2] > 0 for kp in self.keypoints.values())
            if has_valid:
                self.right_fish_abandoned.pop(current_fish_name, None)
                print(f"鱼类 '{current_fish_name}' 右图抛弃标记已解除（检测到有效深度）")

        # 更新点云过滤范围以适应新的深度值
        self._filter_point_cloud()
        self.update_display()

        # 实时更新3D可视化窗口中的关键点位置（如果窗口存在）
        self._update_3d_keypoint_position()

        # 实时更新全局点云窗口中的关键点位置（如果窗口存在）
        self._update_global_3d_keypoint_position()

        # 发送更新消息到GUI子进程
        self._send_gui_update()
    
    def increase_depth(self, event=None):
        """
        增加当前关键点的深度值1mm
        """
        if not self.keypoint_names:
            return
        
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        current_depth = self.keypoints[current_kp_name][2]
        new_depth = min(10000.0, current_depth + 1.0)  # 限制最大值为10000mm
        
        # 更新滑块值（会触发adjust_depth）
        self.depth_slider.set_val(new_depth)
    
    def decrease_depth(self, event=None):
        """
        减少当前关键点的深度值1mm
        """
        if not self.keypoint_names:
            return
        
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        current_depth = self.keypoints[current_kp_name][2]
        new_depth = max(0.0, current_depth - 1.0)  # 限制最小值为0mm
        
        # 更新滑块值（会触发adjust_depth）
        self.depth_slider.set_val(new_depth)
    
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

    def _update_all_3d_keypoint_positions(self):
        """
        实时更新3D可视化窗口中的所有关键点位置（切换鱼类时使用）
        """
        if self.vis_window is None or not self.keypoint_names:
            return

        try:
            # 获取图像尺寸用于坐标翻转
            height, width = self.depth_data.shape

            # 遍历所有关键点，更新它们的位置
            for kp_name in self.keypoint_names:
                if kp_name in self.keypoint_meshes and kp_name in self.keypoints:
                    kp = self.keypoints[kp_name]

                    # 计算新的位置（应用坐标翻转）
                    x_flipped = width - 1 - kp[0]
                    z_flipped = -kp[2]
                    new_position = np.array([x_flipped, kp[1], z_flipped])

                    # 更新现有球体的位置（使用变换矩阵）
                    old_sphere = self.keypoint_meshes[kp_name]

                    # 计算位移
                    current_center = old_sphere.get_center()
                    translation = new_position - current_center

                    # 应用变换
                    old_sphere.translate(translation)

            # 更新关键点几何体
            for mesh in self.keypoint_meshes.values():
                self.vis_window.update_geometry(mesh)

            # 强制重绘
            self.vis_window.poll_events()
            self.vis_window.update_renderer()

            print(f"实时更新所有3D关键点位置，共 {len(self.keypoint_names)} 个关键点")
        except Exception as e:
            print(f"更新所有3D关键点位置失败: {e}")
            # 如果更新失败，可能是因为窗口已关闭，清理引用
            if "destroyed" in str(e).lower() or "closed" in str(e).lower():
                self.vis_window = None
                self.keypoint_meshes = {}

    def _update_all_global_3d_keypoint_positions(self):
        """
        实时更新全局点云窗口中的所有关键点位置（切换鱼类时使用）
        """
        if self.global_vis_window is None or not self.keypoint_names:
            return

        try:
            # 获取图像尺寸用于坐标翻转
            height, width = self.depth_data.shape

            # 遍历所有关键点，更新它们的位置
            for kp_name in self.keypoint_names:
                if kp_name in self.global_keypoint_meshes and kp_name in self.keypoints:
                    kp = self.keypoints[kp_name]

                    # 计算新的位置（应用坐标翻转）
                    x_flipped = width - 1 - kp[0]
                    z_flipped = -kp[2]
                    new_position = np.array([x_flipped, kp[1], z_flipped])

                    # 更新现有球体的位置（使用变换矩阵）
                    old_sphere = self.global_keypoint_meshes[kp_name]

                    # 计算位移
                    current_center = old_sphere.get_center()
                    translation = new_position - current_center

                    # 应用变换
                    old_sphere.translate(translation)

            # 更新关键点几何体
            for mesh in self.global_keypoint_meshes.values():
                self.global_vis_window.update_geometry(mesh)

            # 强制重绘
            self.global_vis_window.poll_events()
            self.global_vis_window.update_renderer()

            print(f"实时更新所有全局3D关键点位置，共 {len(self.keypoint_names)} 个关键点")
        except Exception as e:
            print(f"更新所有全局3D关键点位置失败: {e}")
            # 如果更新失败，可能是因为窗口已关闭，清理引用
            if "destroyed" in str(e).lower() or "closed" in str(e).lower():
                self.global_vis_window = None
                self.global_keypoint_meshes = {}

    def _refresh_3d_view_point_cloud(self):
        """
        重新加载3D View窗口中的点云（使用新的过滤范围）
        """
        if self.vis_window is None:
            return

        try:
            # 创建新的点云对象
            pcd = o3d.geometry.PointCloud()

            if self.point_cloud_filtered is not None and len(self.point_cloud_filtered) > 0:
                # 使用与图像完全一致的坐标系
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
                        print(f"重新加载3D View点云，包含 {len(points)} 个点")
                    else:
                        print("警告: 没有有效的点云颜色数据")
                        pcd.points = o3d.utility.Vector3dVector(points)
                else:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    print(f"重新加载3D View点云（无颜色），包含 {len(points)} 个点")

            # 清除所有几何体，然后重新添加
            self.vis_window.clear_geometries()

            # 重新添加点云
            if len(pcd.points) > 0:
                self.vis_window.add_geometry(pcd)

            # 重新添加所有关键点
            for kp_name, sphere in self.keypoint_meshes.items():
                self.vis_window.add_geometry(sphere)

            # 强制重绘
            self.vis_window.poll_events()
            self.vis_window.update_renderer()

            print("成功重新加载3D View窗口中的点云")

        except Exception as e:
            print(f"重新加载3D View点云失败: {e}")
            import traceback
            traceback.print_exc()


    def next_keypoint(self, event=None):
        """
        切换到下一个关键点
        """
        if not self.keypoint_names:
            return

        self.current_kp_idx = (self.current_kp_idx + 1) % len(self.keypoint_names)
        print(f"切换到关键点索引: {self.current_kp_idx}")
        # 更新同步模式的基准深度值
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        self.last_depth_value = self.keypoints[current_kp_name][2]
        self.update_display()

    def prev_keypoint(self, event=None):
        """
        切换到上一个关键点
        """
        if not self.keypoint_names:
            return

        self.current_kp_idx = (self.current_kp_idx - 1) % len(self.keypoint_names)
        print(f"切换到关键点索引: {self.current_kp_idx}")
        # 更新同步模式的基准深度值
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        self.last_depth_value = self.keypoints[current_kp_name][2]
        self.update_display()
    
    def refresh_point_cloud(self, event=None):
        """
        刷新点云显示范围
        """
        self._filter_point_cloud()
        print(f"刷新点云显示范围为: [{self.z_min}, {self.z_max}]")

        # 重新加载3D View窗口中的点云（如果窗口存在）
        self._refresh_3d_view_point_cloud()

        # 重新加载Global 3D窗口中的点云（如果窗口存在）
        # 暂时不需要全局点云刷新功能

        # 发送点云范围更新消息到GUI子进程
        self._send_gui_point_cloud_update()

    def reset_current_fish_depths(self, event=None):
        """
        将当前鱼的所有关键点的深度重置为0mm，同时标记该鱼在右图中被抛弃（Delete Right）。
        用户可通过拖动 Depth 滑块赋值来解除抛弃状态，下次 Save 时恢复右图构建。
        """
        if not self.fish_names or self.current_fish_idx < 0:
            print("没有当前鱼类，无法重置深度")
            return

        current_fish_name = self.fish_names[self.current_fish_idx]

        # 将当前鱼的所有关键点深度设置为0mm
        reset_count = 0
        for kp_name in self.keypoints.keys():
            self.keypoints[kp_name][2] = 0.0
            reset_count += 1

        # 更新所有鱼类数据中的对应关键点
        if current_fish_name in self.fish_keypoints:
            for kp_name in self.fish_keypoints[current_fish_name].keys():
                self.fish_keypoints[current_fish_name][kp_name][2] = 0.0

        # 将当前鱼的所有关键点添加到reset_keypoints集合中
        if current_fish_name not in self.reset_keypoints:
            self.reset_keypoints[current_fish_name] = set()
        self.reset_keypoints[current_fish_name].update(self.keypoints.keys())

        # 更新标注文件中的description字段 - 只更新属于当前鱼的关键点
        if self.annotation_data and 'shapes' in self.annotation_data:
            for shape in self.annotation_data['shapes']:
                if shape.get('shape_type') == 'point':
                    point = shape['points'][0]
                    x, y = point[0], point[1]

                    if (x, y) in self.keypoint_to_fish_map:
                        fish_name = self.keypoint_to_fish_map[(x, y)]
                        if fish_name == current_fish_name:
                            shape['description'] = "depth: 0.00mm (reset - invalid depth)"
                            print(f"重置鱼类 '{current_fish_name}' 关键点 '{shape['label']}' 的深度为0mm")

        print(f"已将鱼类 '{current_fish_name}' 的 {reset_count} 个关键点深度重置为0mm")

        # 标记该鱼在右图中被抛弃，不再参与右图 JSON 构建
        self.right_fish_abandoned[current_fish_name] = True
        print(f"鱼类 '{current_fish_name}' 已标记为右图抛弃（Delete Right）；拖动 Depth 滑块赋值可解除")

        # 更新点云过滤范围
        self._filter_point_cloud()

        # 更新显示
        self.update_display()

        # 发送更新消息到GUI子进程
        self._send_gui_fish_update()

    def _create_figure(self):
        """
        创建图形界面 - 在初始化时立即创建
        """
        print("创建图形界面")
        # 创建双子图布局（左图 + 右图）
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(18, 8))
        self.ax = self.ax_left  # 保持向后兼容
        plt.subplots_adjust(bottom=0.37, wspace=0.05)  # 调整底部空间和子图间距

        # 创建按钮 - 优化的多行布局，避免重叠
        button_props = {
            # 第1行 (y=0.04): 基础控制与保存
            'prev_frame': {'rect': [0.08, 0.04, 0.10, 0.04], 'label': 'Prev Frame'},
            'next_frame': {'rect': [0.19, 0.04, 0.10, 0.04], 'label': 'Next Frame'},
            'save':       {'rect': [0.30, 0.04, 0.10, 0.04], 'label': 'Save'},
            'refresh_pc': {'rect': [0.41, 0.04, 0.10, 0.04], 'label': 'Refresh PC'},
            'reset_depth':{'rect': [0.52, 0.04, 0.10, 0.04], 'label': 'Delete Right'},

            # 第2行 (y=0.10): 鱼类与关键点导航
            'prev_fish':  {'rect': [0.08, 0.10, 0.10, 0.04], 'label': 'Prev Fish'},
            'next_fish':  {'rect': [0.19, 0.10, 0.10, 0.04], 'label': 'Next Fish'},
            'prev_kp':    {'rect': [0.30, 0.10, 0.10, 0.04], 'label': 'Prev KP'},
            'next_kp':    {'rect': [0.41, 0.10, 0.10, 0.04], 'label': 'Next KP'},

            # 第3行 (y=0.16): 视图切换与模式设置
            '3d_view':      {'rect': [0.08, 0.16, 0.10, 0.04], 'label': '3D View'},
            'global_3d':    {'rect': [0.19, 0.16, 0.10, 0.04], 'label': 'Global 3D'},
            'toggle_labels':{'rect': [0.30, 0.16, 0.10, 0.04], 'label': 'Labels: ON'},
            'sync_fish':    {'rect': [0.41, 0.16, 0.10, 0.04], 'label': 'Sync: OFF'},
            'drag_mode':    {'rect': [0.52, 0.16, 0.10, 0.04], 'label': 'Drag: Single'},

            # 第4行 (y=0.22): 右图关键点 group_id 调整（v=可见, a=不存在）
            'right_kp_visible': {'rect': [0.08, 0.22, 0.14, 0.04], 'label': 'Right KP: Visible [v]'},
            'right_kp_absent':  {'rect': [0.23, 0.22, 0.14, 0.04], 'label': 'Right KP: Absent [a]'},
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
            elif name == 'reset_depth':
                btn.on_clicked(self.reset_current_fish_depths)
            elif name == 'save':
                btn.on_clicked(self.save_keypoints)
            elif name == '3d_view':
                btn.on_clicked(self.visualize_3d)
            elif name == 'global_3d':
                btn.on_clicked(self.visualize_global_3d)
            elif name == 'toggle_labels':
                btn.on_clicked(self.toggle_labels)
            elif name == 'sync_fish':
                btn.on_clicked(self.toggle_sync_mode)
            elif name == 'drag_mode':
                btn.on_clicked(self.toggle_drag_mode)
            elif name == 'right_kp_visible':
                btn.on_clicked(self.set_right_kp_visible)
            elif name == 'right_kp_absent':
                btn.on_clicked(self.set_right_kp_absent)
            # elif name == 'gui_3d':
            #     btn.on_clicked(self.start_gui_3d_windows)

        # 创建深度调整滑块及增减按钮
        # 减少按钮（滑块左侧）
        ax_depth_minus = plt.axes([0.08, 0.28, 0.03, 0.03])
        btn_depth_minus = Button(ax_depth_minus, '-')
        btn_depth_minus.on_clicked(self.decrease_depth)
        self.buttons['depth_minus'] = btn_depth_minus

        # 深度滑块（中间）
        ax_depth = plt.axes([0.12, 0.28, 0.70, 0.03])
        self.depth_slider = Slider(
            ax_depth, 'Depth (mm)', 0, 10000,  # 限制在10米范围内
            valinit=0, valfmt='%d',
            valstep=1.0  # 添加1mm步长，提高精度
        )
        self.depth_slider.on_changed(self.adjust_depth)
        
        # 增加按钮（滑块右侧）
        ax_depth_plus = plt.axes([0.83, 0.28, 0.03, 0.03])
        btn_depth_plus = Button(ax_depth_plus, '+')
        btn_depth_plus.on_clicked(self.increase_depth)
        self.buttons['depth_plus'] = btn_depth_plus

        # 设置图形属性
        self.fig.canvas.manager.set_window_title('鱼类关键点3D坐标验证工具')

        # 连接右图鼠标事件用于拖动
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # 连接键盘快捷键：v=可见, a=不存在
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def update_display(self):
        """
        更新显示 - 左右图并排显示，带参考竖线
        """
        if self.updating_display:
            return

        self.updating_display = True
        try:
            if self.ax_left is None:
                return

            # 保存当前视图范围（用于放大镜工具保持）
            left_xlim = self.ax_left.get_xlim() if self.ax_left.get_images() else None
            left_ylim = self.ax_left.get_ylim() if self.ax_left.get_images() else None
            right_xlim = self.ax_right.get_xlim() if self.ax_right.get_images() else None
            right_ylim = self.ax_right.get_ylim() if self.ax_right.get_images() else None

            # 清空左右子图
            self.ax_left.clear()
            self.ax_right.clear()

            # 获取相机参数用于投影计算
            fx, baseline = self._get_stereo_params()

            # ========== 左图显示 ==========
            if self.image_data is not None:
                try:
                    self.ax_left.imshow(self.image_data)
                except Exception as e:
                    print(f"显示左图失败: {e}")
                    if self.depth_data is not None:
                        depth_display = np.clip(self.depth_data, 0, 10000) / 10000
                        self.ax_left.imshow(depth_display, cmap='jet')
            else:
                if self.depth_data is not None:
                    depth_display = np.clip(self.depth_data, 0, 10000) / 10000
                    self.ax_left.imshow(depth_display, cmap='jet')
                else:
                    self.ax_left.text(0.5, 0.5, '无法加载左图',
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform=self.ax_left.transAxes)

            # ========== 右图显示 ==========
            if self.right_image_data is not None:
                try:
                    self.ax_right.imshow(self.right_image_data)
                except Exception as e:
                    print(f"显示右图失败: {e}")
                    self.ax_right.text(0.5, 0.5, '右图显示失败',
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      transform=self.ax_right.transAxes)
            else:
                self.ax_right.text(0.5, 0.5, '右图不可用',
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  transform=self.ax_right.transAxes)

            # ========== 绘制关键点和参考线 ==========
            if self.keypoint_names:
                # 获取图像高度用于绘制竖线
                img_height = self.image_data.shape[0] if self.image_data is not None else 1080
                img_width = self.image_data.shape[1] if self.image_data is not None else 1440

                # 当前鱼的 fish_id（用于查询右图 group_id，提出循环外避免重复计算）
                cur_fish_id_for_loop = self.fish_names[self.current_fish_idx] if self.fish_names else None

                for i, (name, kp) in enumerate(self.keypoints.items()):
                    is_current = (i == self.current_kp_idx)
                    color = 'red' if is_current else 'blue'
                    x_left, y_left, depth = kp[0], kp[1], kp[2]

                    # 计算右图映射点
                    x_right, y_right = project_left_to_right(x_left, y_left, depth, fx, baseline)

                    # 获取该关键点在右图的 group_id
                    right_gid = self.right_kp_group_ids.get(cur_fish_id_for_loop, {}).get(name, None) if cur_fish_id_for_loop else None

                    # 左图：绘制关键点
                    self.ax_left.plot(x_left, y_left, 'o', color=color, markersize=8, markeredgewidth=2)
                    if self.show_labels:
                        self.ax_left.text(x_left+10, y_left+10, f"{name}\n{depth:.0f}mm",
                                         color=color, fontsize=9, weight='bold',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

                    # 右图：绘制映射点（仅在重叠区域内）
                    if self.OVERLAP_X_MIN <= x_left <= self.OVERLAP_X_MAX:
                        if right_gid == 1:
                            # 关键点不存在：画灰色 × 叉号
                            self.ax_right.plot(x_right, y_right, 'x', color='gray',
                                              markersize=12, markeredgewidth=2.5)
                            if self.show_labels:
                                self.ax_right.text(x_right+10, y_right+10, f"{name} [X]",
                                                  color='gray', fontsize=9, weight='bold',
                                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                        else:
                            # 关键点可见：画空心圆
                            self.ax_right.plot(x_right, y_right, 'o', color=color, markersize=8,
                                              markeredgewidth=2, fillstyle='none')
                            if self.show_labels:
                                self.ax_right.text(x_right+10, y_right+10, f"{name}",
                                                  color=color, fontsize=9, weight='bold',
                                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

                    # 绘制参考竖线（仅对当前关键点）
                    if is_current:
                        self._draw_reference_lines(self.ax_left, x_left, img_height, img_width, is_main=True)
                        self._draw_reference_lines(self.ax_right, x_right, img_height, img_width, is_main=True)

                # 高亮当前关键点
                if self.current_kp_idx < len(self.keypoint_names):
                    current_kp_name = self.keypoint_names[self.current_kp_idx]
                    current_kp = self.keypoints[current_kp_name]
                    x_left, y_left, depth = current_kp[0], current_kp[1], current_kp[2]
                    x_right, y_right = project_left_to_right(x_left, y_left, depth, fx, baseline)

                    # 获取当前关键点右图 group_id（复用 cur_fish_id_for_loop）
                    cur_right_gid = self.right_kp_group_ids.get(cur_fish_id_for_loop, {}).get(current_kp_name, None) if cur_fish_id_for_loop else None

                    # 左图高亮
                    self.ax_left.plot(x_left, y_left, 'o', color='yellow',
                                     markersize=12, markeredgewidth=3, markeredgecolor='red')
                    # 右图高亮（仅在重叠区域内）
                    if self.OVERLAP_X_MIN <= x_left <= self.OVERLAP_X_MAX:
                        if cur_right_gid == 1:
                            # 不存在：高亮叉号（橙色）
                            self.ax_right.plot(x_right, y_right, 'x', color='orange',
                                              markersize=16, markeredgewidth=3.5)
                        else:
                            self.ax_right.plot(x_right, y_right, 'o', color='yellow',
                                              markersize=12, markeredgewidth=3, markeredgecolor='red',
                                              fillstyle='none')

                # 更新标题
                frame_name = self.frames[self.current_frame_idx] if self.frames else "未知"
                fish_name = self.fish_names[self.current_fish_idx] if self.fish_names else "未知"
                current_kp_name = self.keypoint_names[self.current_kp_idx] if self.keypoint_names else "未知"
                current_depth = self.keypoints[current_kp_name][2] if self.keypoint_names else 0

                # 获取当前关键点右图状态标签（复用 cur_fish_id_for_loop，避免重复计算）
                cur_right_gid_title = self.right_kp_group_ids.get(cur_fish_id_for_loop, {}).get(current_kp_name, None) if cur_fish_id_for_loop else None
                right_kp_status = "Absent[1]" if cur_right_gid_title == 1 else "Visible[null]"

                # 计算当前关键点的视差
                disparity = (fx * baseline) / current_depth if current_depth > 0 else 0

                self.fig.suptitle(
                    f"Frame: {frame_name} | Fish: {fish_name} | KP: {current_kp_name} | "
                    f"Depth: {current_depth:.0f}mm | Disp: {disparity:.1f}px | Right: {right_kp_status}",
                    fontsize=11, weight='bold'
                )

                # 更新滑块
                if self.keypoint_names:
                    self.updating_slider = True
                    self.depth_slider.set_val(current_depth)
                    self.updating_slider = False

            # 设置子图标签
            self.ax_left.set_title('Left', fontsize=11)
            self.ax_left.set_xlabel('X (pixels)')
            self.ax_left.set_ylabel('Y (pixels)')

            self.ax_right.set_title('Right - Projected Points', fontsize=11)
            self.ax_right.set_xlabel('X (pixels)')
            self.ax_right.set_ylabel('Y (pixels)')

            # 恢复视图范围（保持放大镜工具状态）
            if left_xlim is not None:
                self.ax_left.set_xlim(left_xlim)
                self.ax_left.set_ylim(left_ylim)
            if right_xlim is not None:
                self.ax_right.set_xlim(right_xlim)
                self.ax_right.set_ylim(right_ylim)

            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"更新显示失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.updating_display = False

    def _draw_reference_lines(self, ax, x_center, img_height, img_width, is_main=True, interval=50):
        """
        绘制参考竖线（铺满全图）

        Args:
            ax: matplotlib轴对象
            x_center: 中心x坐标（关键点/映射点位置）
            img_height: 图像高度
            img_width: 图像宽度
            is_main: 是否为主线（关键点位置）
            interval: 参考线间隔（像素）
        """
        # 动态计算需要的线条数量以铺满全图
        num_lines = img_width // interval + 1

        # 主线（关键点/映射点位置）- 粗线
        if 0 <= x_center < img_width:
            ax.axvline(x=x_center, color='lime', linewidth=2, alpha=0.8, linestyle='-')

        # 辅助参考线 - 细线
        for i in range(1, num_lines + 1):
            # 左侧参考线
            x_left = x_center - i * interval
            if 0 <= x_left < img_width:
                ax.axvline(x=x_left, color='cyan', linewidth=0.5, alpha=0.4, linestyle='--')

            # 右侧参考线
            x_right = x_center + i * interval
            if 0 <= x_right < img_width:
                ax.axvline(x=x_right, color='cyan', linewidth=0.5, alpha=0.4, linestyle='--')

    def toggle_labels(self, event=None):
        """切换标签显示状态"""
        self.show_labels = not self.show_labels
        # 更新按钮文字
        btn = self.buttons.get('toggle_labels')
        if btn:
            btn.label.set_text('Labels: ON' if self.show_labels else 'Labels: OFF')
        self.update_display()

    def toggle_sync_mode(self, event=None):
        """切换同步深度调整模式"""
        self.sync_depth_mode = not self.sync_depth_mode
        # 更新按钮文字
        btn = self.buttons.get('sync_fish')
        if btn:
            btn.label.set_text('Sync: ON' if self.sync_depth_mode else 'Sync: OFF')
        # 记录当前深度值作为基准
        if self.sync_depth_mode and self.keypoint_names:
            current_kp_name = self.keypoint_names[self.current_kp_idx]
            self.last_depth_value = self.keypoints[current_kp_name][2]
        self.fig.canvas.draw_idle()

    def toggle_drag_mode(self, event=None):
        """切换右图拖动模式"""
        self.drag_mode_global = not self.drag_mode_global
        # 更新按钮文字
        btn = self.buttons.get('drag_mode')
        if btn:
            btn.label.set_text('Drag: Global' if self.drag_mode_global else 'Drag: Single')
        self.fig.canvas.draw_idle()

    def set_right_kp_visible(self, event=None):
        """将当前关键点在右图中标记为可见（group_id = None）"""
        if not self.fish_names or self.current_fish_idx < 0:
            return
        if not self.keypoint_names or self.current_kp_idx >= len(self.keypoint_names):
            return
        current_fish = self.fish_names[self.current_fish_idx]
        current_kp = self.keypoint_names[self.current_kp_idx]
        if current_fish not in self.right_kp_group_ids:
            self.right_kp_group_ids[current_fish] = {}
        self.right_kp_group_ids[current_fish][current_kp] = None
        print(f"右图关键点 [{current_fish}] '{current_kp}' 设置为可见 (group_id=null)")
        self.update_display()

    def set_right_kp_absent(self, event=None):
        """将当前关键点在右图中标记为不存在（group_id = 1）"""
        if not self.fish_names or self.current_fish_idx < 0:
            return
        if not self.keypoint_names or self.current_kp_idx >= len(self.keypoint_names):
            return
        current_fish = self.fish_names[self.current_fish_idx]
        current_kp = self.keypoint_names[self.current_kp_idx]
        if current_fish not in self.right_kp_group_ids:
            self.right_kp_group_ids[current_fish] = {}
        self.right_kp_group_ids[current_fish][current_kp] = 1
        print(f"右图关键点 [{current_fish}] '{current_kp}' 设置为不存在 (group_id=1)")
        self.update_display()

    def _get_stereo_params(self):
        """获取立体视觉参数（fx和baseline）"""
        fx = self.camera_params.get('camera_matrix_left', {}).get('data', [2351.0])[0] if self.camera_params else 2351.0
        baseline = abs(self.camera_params.get('T', {}).get('data', [-40.39])[0]) if self.camera_params else 40.39
        return fx, baseline

    def _check_keypoint_click(self, kp_name, event_x, event_y, fx, baseline, threshold=20):
        """
        检查是否点击在指定关键点附近

        Returns:
            bool: 是否点击在关键点附近
        """
        from utils.camera_utils import project_left_to_right

        x_left, y_left, depth = self.keypoints[kp_name]

        # 检查是否在重叠区域
        if not (self.OVERLAP_X_MIN <= x_left <= self.OVERLAP_X_MAX):
            return False

        # 计算右图坐标
        x_right, y_right = project_left_to_right(x_left, y_left, depth, fx, baseline)

        # 检查距离
        dist = ((event_x - x_right)**2 + (event_y - y_right)**2)**0.5

        return dist < threshold

    def _on_mouse_press(self, event):
        """鼠标按下事件 - 检测是否点击右图关键点"""
        if event.inaxes != self.ax_right or event.button != 1:
            return

        if not self.keypoint_names:
            return

        # 获取相机参数
        fx, baseline = self._get_stereo_params()
        click_threshold = 20

        # 确定要检查的关键点列表
        if self.drag_mode_global:
            kp_names_to_check = self.keypoint_names
        else:
            kp_names_to_check = [self.keypoint_names[self.current_kp_idx]]

        # 检查是否点击在关键点附近
        for kp_name in kp_names_to_check:
            if self._check_keypoint_click(kp_name, event.xdata, event.ydata, fx, baseline, click_threshold):
                self.dragging_right = True
                self.drag_kp_name = kp_name
                self.drag_start_x = event.xdata
                print(f"开始拖动关键点: {kp_name}")
                break

    def _on_mouse_move(self, event):
        """鼠标移动事件 - 拖动时更新深度"""
        if not self.dragging_right or event.inaxes != self.ax_right:
            return

        if self.drag_kp_name is None:
            return

        # 获取当前关键点的左图坐标
        x_left, _, _ = self.keypoints[self.drag_kp_name]

        # 新的右图x坐标
        x_right_new = event.xdata

        # 反向计算深度
        fx, baseline = self._get_stereo_params()

        disparity = x_left - x_right_new

        # 防止除零和负深度
        if disparity <= 0:
            return

        new_depth = (fx * baseline) / disparity

        # 限制深度范围
        new_depth = max(0.0, min(10000.0, new_depth))

        # 更新深度
        self.keypoints[self.drag_kp_name][2] = new_depth

        # 更新滑块（如果当前关键点是正在编辑的）
        current_kp_name = self.keypoint_names[self.current_kp_idx]
        if self.drag_kp_name == current_kp_name and self.depth_slider:
            self.updating_slider = True
            self.depth_slider.set_val(new_depth)
            self.updating_slider = False

        # 更新显示
        self.update_display()

    def _on_mouse_release(self, event):
        """鼠标释放事件 - 结束拖动"""
        if self.dragging_right:
            print(f"结束拖动关键点: {self.drag_kp_name}")
            self.dragging_right = False
            self.drag_kp_name = None
            self.drag_start_x = None

    def _on_key_press(self, event):
        """
        键盘快捷键处理
        v: 将当前右图关键点设置为可见（group_id=null）
        a: 将当前右图关键点设置为不存在（group_id=1）
        """
        if event.key == 'v':
            self.set_right_kp_visible()
        elif event.key == 'a':
            self.set_right_kp_absent()

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

    def _run_gui_in_main_process(self):
        """
        在主进程中运行GUI（Windows兼容模式）
        """
        try:
            import open3d as o3d
            import threading
            import platform

            # 设置Open3D环境变量
            os.environ['OPEN3D_DISABLE_GUI'] = '0'
            os.environ['OPEN3D_HEADLESS'] = '0'
            if platform.system() == 'Windows':
                os.environ['DISPLAY'] = ''
                os.environ['OPEN3D_USE_NATIVE_WINDOWS_OPENGL'] = '1'

            print("正在创建Open3D可视化窗口...")

            # 创建可视化器
            local_vis = o3d.visualization.VisualizerWithKeyCallback()
            local_vis.create_window("局部3D点云", 800, 600)

            global_vis = o3d.visualization.VisualizerWithKeyCallback()
            global_vis.create_window("全局3D点云 (0-10米)", 800, 600)

            # 设置渲染选项
            local_vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
            global_vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])

            for vis in [local_vis, global_vis]:
                render_option = vis.get_render_option()
                render_option.point_size = 2.0
                render_option.background_color = np.array([0.05, 0.05, 0.05])
                render_option.light_on = True

            # 创建和添加点云
            if self.point_cloud_data is not None:
                # 创建全局点云（0-10米范围）
                global_pcd = o3d.geometry.PointCloud()
                global_pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data)
                global_pcd.colors = o3d.utility.Vector3dVector(self.color_rectify_data.reshape(-1, 3) / 255.0)
                global_vis.add_geometry(global_pcd)

                # 创建局部点云（过滤范围）
                local_points = self.point_cloud_data[
                    (self.point_cloud_data[:, 2] >= self.z_min) &
                    (self.point_cloud_data[:, 2] <= self.z_max)
                ]
                local_colors = self.color_rectify_data[
                    (self.point_cloud_data[:, 2] >= self.z_min) &
                    (self.point_cloud_data[:, 2] <= self.z_max)
                ].reshape(-1, 3) / 255.0

                if len(local_points) > 0:
                    local_pcd = o3d.geometry.PointCloud()
                    local_pcd.points = o3d.utility.Vector3dVector(local_points)
                    local_pcd.colors = o3d.utility.Vector3dVector(local_colors)
                    local_vis.add_geometry(local_pcd)

            # 添加关键点
            if self.keypoints:
                for kp_name, (x, y, z) in self.keypoints.items():
                    # 翻转X坐标以匹配图像坐标系
                    height, width = self.depth_data.shape
                    x_flipped = width - 1 - x

                    # 创建关键点球体
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    sphere.translate([x_flipped, y, -z])  # 翻转Z轴
                    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色关键点

                    local_vis.add_geometry(sphere)
                    global_vis.add_geometry(sphere)

            # 设置视角
            local_view_control = local_vis.get_view_control()
            local_view_control.set_front([0, -0.5, 1])
            local_view_control.set_lookat([0, 0, 0])
            local_view_control.set_up([0, -1, 0])
            local_view_control.set_zoom(0.8)

            global_view_control = global_vis.get_view_control()
            global_view_control.set_front([0, -0.3, 1])
            global_view_control.set_lookat([0, 0, 0])
            global_view_control.set_up([0, -1, 0])
            global_view_control.set_zoom(0.3)

            print("Open3D可视化窗口已创建，Windows用户请手动关闭窗口以继续")

            # 运行可视化器（阻塞模式）
            local_vis.run()
            local_vis.destroy_window()
            global_vis.run()
            global_vis.destroy_window()

            print("GUI可视化完成")

        except Exception as e:
            print(f"Windows兼容模式GUI运行失败: {e}")
            import traceback
            traceback.print_exc()

    def start_gui_3d_windows(self, event=None):
        """
        启动GUI 3D窗口子进程
        """
        import platform

        if self.gui_process is not None and self.gui_process.is_alive():
            print("GUI 3D窗口已在运行")
            return

        # Windows兼容性处理
        if platform.system() == 'Windows':
            print("检测到Windows系统，使用兼容模式启动GUI...")
            self._run_gui_in_main_process()
            return

        try:
            # 设置进程启动方式（Linux/macOS使用fork）
            mp.set_start_method('fork', force=True)

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

    def _build_right_json(self, base_name: str) -> dict:
        """
        为当前帧的所有鱼生成右图 labelme JSON 数据。

        过滤规则（同时满足才写入右图）：
        1. 整鱼抛弃：right_fish_abandoned[fish_id]==True → 跳过整条鱼
        2. 单点过滤：depth==0 或 x_right 不在 [OVERLAP_X_MIN, OVERLAP_X_MAX] → 跳过该点
        3. 整鱼自动跳过：一条鱼所有关键点都被过滤 → 连 bbox 也不建立
        4. imagePath 仅写文件名，labelme 在 JSON 同目录下查找图片

        Returns:
            dict: labelme 格式的 JSON 字典，可直接 json.dump
        """
        from utils.camera_utils import project_bbox_left_to_right, project_left_to_right

        fx, baseline = self._get_stereo_params()

        img_height = self.image_data.shape[0] if self.image_data is not None else 1080
        img_width = self.image_data.shape[1] if self.image_data is not None else 1440

        version = self.annotation_data.get('version', '5.2.1')
        flags = self.annotation_data.get('flags', {})

        right_shapes = []

        # 第一步：建立 fish_id -> bbox shape 映射
        fish_bbox_map = {}
        fish_count = 0
        for shape in self.annotation_data['shapes']:
            if shape['shape_type'] == 'rectangle' and shape['label'] == 'fish' and shape.get('group_id') == 0:
                fish_count += 1
                fish_bbox_map[f"fish_{fish_count}"] = shape

        # 第二步：遍历每条鱼
        for fish_id, kps in self.fish_keypoints.items():

            # 规则1：用户手动抛弃 → 整条跳过
            if self.right_fish_abandoned.get(fish_id, False):
                print(f"跳过 {fish_id}（用户手动 Delete Right）")
                continue

            fish_right_gids = self.right_kp_group_ids.get(fish_id, {})

            # 规则2+3：逐点过滤，收集有效关键点
            valid_kp_shapes = []
            for kp_name, kp in kps.items():
                x_left, y_left, depth = float(kp[0]), float(kp[1]), float(kp[2])

                # depth==0 → 跳过（SGBM 无法计算视差 或 用户 Reset）
                if depth <= 0:
                    print(f"  跳过 {fish_id}/{kp_name}：depth={depth:.1f}mm")
                    continue

                x_right, y_right = project_left_to_right(x_left, y_left, depth, fx, baseline)

                # 不在双目重叠区域 → 跳过
                if x_right < self.OVERLAP_X_MIN or x_right > self.OVERLAP_X_MAX:
                    print(f"  跳过 {fish_id}/{kp_name}：x_right={x_right:.1f} 不在重叠区域 [{self.OVERLAP_X_MIN}, {self.OVERLAP_X_MAX}]")
                    continue

                right_gid = fish_right_gids.get(kp_name, None)

                # 从左图 shapes 中找对应 description
                description = ''
                for s in self.annotation_data['shapes']:
                    if (s['shape_type'] == 'point' and s['label'] == kp_name
                            and s.get('group_id') is None):
                        sp = s['points'][0]
                        if abs(sp[0] - x_left) < 1.0 and abs(sp[1] - y_left) < 1.0:
                            description = s.get('description', '')
                            break

                valid_kp_shapes.append({
                    "label": kp_name,
                    "points": [[x_right, y_right]],
                    "group_id": right_gid,
                    "description": description,
                    "shape_type": "point",
                    "flags": {}
                })

            # 规则3：无有效关键点 → 整条鱼（含 bbox）跳过
            if not valid_kp_shapes:
                print(f"跳过 {fish_id}：所有关键点均在重叠区域外或 depth==0")
                continue

            # 构建 bbox（用有效关键点的平均深度投影）
            bbox_shape = fish_bbox_map.get(fish_id)
            if bbox_shape is not None:
                pts = bbox_shape['points']
                lx1, ly1 = pts[0][0], pts[0][1]
                lx2, ly2 = pts[1][0], pts[1][1]
                valid_depths = [kp[2] for kp in kps.values() if kp[2] > 0]
                avg_depth = float(np.mean(valid_depths))
                rx1, ry1, rx2, ry2 = project_bbox_left_to_right(lx1, ly1, lx2, ly2, avg_depth, fx, baseline)
                right_shapes.append({
                    "label": "fish",
                    "points": [[rx1, ry1], [rx2, ry2]],
                    "group_id": 0,
                    "description": f"projected from left, avg_depth: {avg_depth:.1f}mm",
                    "shape_type": "rectangle",
                    "flags": {}
                })

            right_shapes.extend(valid_kp_shapes)

        right_json = {
            "version": version,
            "flags": flags,
            "shapes": right_shapes,
            "imagePath": f"{base_name}.png",
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        return right_json

    def save_keypoints(self, event=None):
        """
        保存关键点 - 全量保存当前帧所有鱼的深度值（左图JSON）和右图JSON。
        类似IDE的全局保存：不局限于当前鱼，所有鱼的当前状态都写入文件。
        """
        if not self.annotation_data:
            print("没有加载标注数据，无法保存")
            return

        if not self.frames:
            print("没有当前帧信息，无法确定保存路径")
            return

        current_frame_name = self.frames[self.current_frame_idx]
        base_name = os.path.splitext(current_frame_name)[0]

        # 构建保存路径
        annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme', 'left')
        backup_root = os.path.join(annotations_root, 'backup')
        current_annotation_file = os.path.join(annotations_root, f"{base_name}.json")
        backup_annotation_file = os.path.join(backup_root, f"{base_name}.json")

        try:
            os.makedirs(backup_root, exist_ok=True)

            # 备份原文件
            if os.path.exists(current_annotation_file):
                import shutil
                shutil.copy2(current_annotation_file, backup_annotation_file)
                print(f"已创建备份文件: {backup_annotation_file}")
            else:
                print(f"原标注文件不存在: {current_annotation_file}，将直接保存")

            if not self.fish_names:
                print("没有鱼类信息，无法保存")
                return

            # ========== 全量更新左图 JSON：所有鱼的所有关键点深度 ==========
            updated_count = 0
            not_found_count = 0
            for shape in self.annotation_data['shapes']:
                if shape['shape_type'] == 'point':
                    point = shape['points'][0]
                    x, y = point[0], point[1]

                    if (x, y) in self.keypoint_to_fish_map:
                        fish_name = self.keypoint_to_fish_map[(x, y)]
                        kp_name = shape['label']

                        # 更新所有鱼（不过滤 current_fish_name）
                        if fish_name in self.fish_keypoints and kp_name in self.fish_keypoints[fish_name]:
                            depth_value = self.fish_keypoints[fish_name][kp_name][2]
                            shape['description'] = f"depth: {depth_value:.2f}mm"
                            updated_count += 1
                        else:
                            print(f"警告: 找不到 {fish_name} 中关键点 {kp_name} 的深度数据")
                            not_found_count += 1
                    else:
                        not_found_count += 1

            print(f"全量更新: {updated_count} 个关键点深度已写入（{len(self.fish_names)} 条鱼）"
                  + (f"，{not_found_count} 个未匹配" if not_found_count else ""))

            # 保存左图 JSON
            with open(current_annotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotation_data, f, indent=4, ensure_ascii=False)
            print(f"成功保存左图JSON: {current_annotation_file}")

            # ========== 保存右图 JSON ==========
            try:
                right_annotations_root = os.path.join(os.path.dirname(self.depth_root), 'annotations', 'labelme', 'right')
                right_backup_root = os.path.join(right_annotations_root, 'backup')
                right_annotation_file = os.path.join(right_annotations_root, f"{base_name}.json")
                right_backup_file = os.path.join(right_backup_root, f"{base_name}.json")

                os.makedirs(right_annotations_root, exist_ok=True)
                os.makedirs(right_backup_root, exist_ok=True)

                if os.path.exists(right_annotation_file):
                    import shutil
                    shutil.copy2(right_annotation_file, right_backup_file)
                    print(f"已备份右图JSON: {right_backup_file}")

                right_json = self._build_right_json(base_name)
                with open(right_annotation_file, 'w', encoding='utf-8') as f:
                    json.dump(right_json, f, indent=4, ensure_ascii=False)
                print(f"成功保存右图JSON: {right_annotation_file}")

            except Exception as e_right:
                print(f"保存右图JSON失败: {e_right}")

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
                       default='fish_dataset/annotations/labelme/left/fishdata.json',
                       help='默认标注文件路径（当对应帧的标注文件不存在时使用）')

    args = parser.parse_args()

    # 检查数据集根目录
    if not os.path.exists(args.dataset_root):
        print(f"数据集根目录不存在: {args.dataset_root}")
        return False

    # 构建各个子目录路径
    depth_root = os.path.join(args.dataset_root, 'depths')
    images_root = os.path.join(args.dataset_root, 'images', 'left')
    annotations_root = os.path.join(args.dataset_root, 'annotations', 'labelme', 'left')

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
        import platform

        # 设置Open3D环境变量以避免GUI上下文问题
        os.environ['OPEN3D_DISABLE_GUI'] = '0'  # 启用GUI
        os.environ['OPEN3D_HEADLESS'] = '0'     # 非无头模式

        # Windows特定设置
        if platform.system() == 'Windows':
            os.environ['DISPLAY'] = ''  # Windows不需要DISPLAY变量
            # 设置OpenGL上下文选项
            os.environ['OPEN3D_USE_NATIVE_WINDOWS_OPENGL'] = '1'
        else:
            os.environ['DISPLAY'] = ':0'    # Linux/macOS需要

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
                                    try:
                                        local_vis.get_view_control().convert_from_pinhole_camera_parameters(saved_local_view_params[0])
                                        print("已恢复局部视角设置")
                                    except Exception as view_e:
                                        print(f"恢复局部视角失败（可能是窗口尺寸不匹配），使用默认视角: {view_e}")
                                        # 如果恢复失败，使用默认视角设置
                                        local_view_control = local_vis.get_view_control()
                                        local_view_control.set_front([0, -0.5, 1])
                                        local_view_control.set_lookat([0, 0, 0])
                                        local_view_control.set_up([0, -1, 0])
                                        local_view_control.set_zoom(0.8)

                                if saved_global_view_params[0] is not None:
                                    # 恢复全局视角到用户之前的设置
                                    try:
                                        global_vis.get_view_control().convert_from_pinhole_camera_parameters(saved_global_view_params[0])
                                        print("已恢复全局视角设置")
                                    except Exception as view_e:
                                        print(f"恢复全局视角失败（可能是窗口尺寸不匹配），使用默认视角: {view_e}")
                                        # 如果恢复失败，使用默认视角设置
                                        global_view_control = global_vis.get_view_control()
                                        global_view_control.set_front([0, -0.3, 1])
                                        global_view_control.set_lookat([0, 0, 0])
                                        global_view_control.set_up([0, -1, 0])
                                        global_view_control.set_zoom(0.3)

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