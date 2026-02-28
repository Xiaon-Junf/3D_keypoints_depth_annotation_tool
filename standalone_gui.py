#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立运行的3D可视化GUI进程
通过读取状态文件来与主进程同步状态，避免进程间直接通信
"""

import os
import sys
import json
import time
import argparse
import threading
import tempfile
from pathlib import Path

import numpy as np
import cv2
import yaml

# 尝试导入Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("错误: 未安装Open3D，无法运行GUI")
    sys.exit(1)

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


class Standalone3DGUI:
    """独立的3D可视化GUI进程"""

    def __init__(self, state_file_path, camera_config_path):
        """
        初始化独立GUI进程
        """
        self.state_file_path = state_file_path
        self.camera_config_path = camera_config_path

        # 状态管理
        self.current_state = {}
        self.last_state_update = 0
        self.state_check_interval = 0.2  # 200ms检查一次状态文件
        self.state_update_thread = None
        self.running = True

        # 可视化相关
        self.local_vis = None
        self.global_vis = None
        self.local_pcd_geometry = None
        self.global_pcd_geometry = None
        self.keypoint_spheres_local = {}  # 本地窗口的关键点球体
        self.keypoint_spheres_global = {}  # 全局窗口的关键点球体

        # 数据缓存
        self.depth_data = None
        self.color_rectify_data = None
        self.point_cloud_data = None
        self.fish_keypoints = {}
        self.fish_bbox = {}  # 新增：bbox信息

        # 相机参数
        self.camera_params = self._load_camera_config()

        # 初始化数据
        self._initialize_data()

        # 数据路径
        self.depth_root = None  # 需要从状态中推断
        self.images_root = None
        self.current_frame = None

        print(f"独立GUI进程已初始化，状态文件: {state_file_path}")

    def _load_frame_data(self, frame_name):
        """根据帧名加载深度数据和图像数据"""
        try:
            if not frame_name:
                return

            # 推断数据路径
            camera_config_dir = os.path.dirname(self.camera_config_path)
            dataset_root = os.path.dirname(camera_config_dir)

            depth_root = os.path.join(dataset_root, 'depths')
            images_root = os.path.join(dataset_root, 'images')

            # 构建文件路径
            base_name = os.path.splitext(frame_name)[0]
            depth_file_path = os.path.join(depth_root, f"{base_name}.npy")
            image_file_path = os.path.join(images_root, frame_name)

            print(f"GUI加载帧数据: {frame_name}")

            # 加载深度数据
            if os.path.exists(depth_file_path):
                # 使用深度读取器
                try:
                    from utils.simple_depth_reader import SimpleDepthReader
                    depth_reader = SimpleDepthReader(self.camera_config_path)
                    image_size = self.camera_params.get('image_size', [1440, 1080])
                    self.depth_data = depth_reader.read_depth(
                        disp_path=depth_file_path,
                        target_size=(image_size[0], image_size[1]),
                        normalize=False
                    )
                    print(f"GUI成功加载深度图: {depth_file_path}, 尺寸: {self.depth_data.shape}")
                except Exception as e:
                    print(f"GUI深度读取器失败，回退到直接读取: {e}")
                    # 回退到直接读取
                    disparity_data = np.load(depth_file_path)
                    self.depth_data = disparity_data  # 假设已经是深度数据
                    print(f"GUI直接加载深度图: {depth_file_path}, 尺寸: {self.depth_data.shape}")
            else:
                print(f"GUI深度文件不存在: {depth_file_path}")
                self.depth_data = np.ones((1080, 1440)) * 1000  # 默认深度

            # 加载图像数据
            if os.path.exists(image_file_path):
                self.color_rectify_data = cv2.imread(image_file_path)
                self.color_rectify_data = cv2.cvtColor(self.color_rectify_data, cv2.COLOR_BGR2RGB)
                print(f"GUI成功加载图像: {image_file_path}, 尺寸: {self.color_rectify_data.shape}")
            else:
                print(f"GUI图像文件不存在: {image_file_path}")
                self.color_rectify_data = np.full((1080, 1440, 3), [128, 128, 128], dtype=np.uint8)

            # 生成点云数据
            self._generate_point_cloud_data()

        except Exception as e:
            print(f"GUI加载帧数据失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_point_cloud_data(self):
        """生成点云数据用于3D可视化"""
        try:
            if self.depth_data is None:
                return

            height, width = self.depth_data.shape

            # 创建坐标网格
            x_coords = np.arange(width)
            y_coords = np.arange(height)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            # 有效深度值掩码
            valid_mask = (self.depth_data > 0) & (self.depth_data < 10000)

            # 保留所有有效点
            valid_indices = np.where(valid_mask)
            y_sample, x_sample = valid_indices

            # 提取有效点
            x_points = x_sample
            y_points = y_sample
            z_points = self.depth_data[y_sample, x_sample]

            # 翻转x坐标以匹配图像显示
            x_points = width - 1 - x_points

            # 翻转z坐标以修复深度方向
            z_points = -z_points

            # 创建点云
            self.point_cloud_data = np.column_stack((x_points, y_points, z_points))
            print(f"GUI生成点云数据，包含 {len(self.point_cloud_data)} 个点")

        except Exception as e:
            print(f"GUI生成点云数据失败: {e}")
            self.point_cloud_data = np.random.rand(1000, 3) * 1000  # 创建模拟数据

    def _load_camera_config(self):
        """加载相机配置"""
        try:
            with open(self.camera_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载相机配置失败: {e}")
            return None

    def _initialize_data(self):
        """初始化基本数据"""
        try:
            # 加载相机参数获取图像尺寸
            image_size = self.camera_params.get('image_size', [1440, 1080])

            # 创建模拟数据用于初始化
            self.depth_data = np.ones((image_size[1], image_size[0])) * 1000  # 默认深度1000mm
            self.color_rectify_data = np.full((image_size[1], image_size[0], 3), [128, 128, 128], dtype=np.uint8)

            # 创建基本的点云数据用于初始化
            height, width = self.depth_data.shape
            x_coords = np.arange(width)
            y_coords = np.arange(height)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            valid_mask = (self.depth_data > 0) & (self.depth_data < 10000)
            valid_indices = np.where(valid_mask)
            y_sample, x_sample = valid_indices

            x_points = x_sample
            y_points = y_sample
            z_points = self.depth_data[y_sample, x_sample]

            # 翻转坐标系以匹配显示要求
            x_points = width - 1 - x_points
            z_points = -z_points

            self.point_cloud_data = np.column_stack((x_points, y_points, z_points))

            print(f"初始化数据完成，点云包含 {len(self.point_cloud_data)} 个点")

        except Exception as e:
            print(f"初始化数据失败: {e}")
            self.point_cloud_data = np.random.rand(1000, 3) * 1000  # 创建模拟数据

    def _load_state_file(self):
        """读取状态文件"""
        try:
            if not os.path.exists(self.state_file_path):
                return None

            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # 检查时间戳确保状态是最新的
            if 'timestamp' in state:
                # 如果状态太旧（超过5秒），忽略
                if time.time() - state['timestamp'] > 5.0:
                    return None

            return state

        except Exception as e:
            print(f"读取状态文件失败: {e}")
            return None

    def _should_update_visualization(self, new_state):
        """判断是否需要更新可视化"""
        if not self.current_state:
            return True

        # 检查关键状态变化
        state_keys_to_check = ['current_fish_idx', 'fish_keypoints', 'point_cloud_range']

        for key in state_keys_to_check:
            if key not in self.current_state or key not in new_state:
                return True
            if self.current_state[key] != new_state[key]:
                return True

        return False

    def _update_data_from_state(self, state):
        """从状态更新内部数据"""
        try:
            # 更新鱼类关键点数据
            if 'fish_keypoints' in state:
                self.fish_keypoints = {}
                for fish_name, keypoints in state['fish_keypoints'].items():
                    self.fish_keypoints[fish_name] = {}
                    for kp_name, kp_coords in keypoints.items():
                        self.fish_keypoints[fish_name][kp_name] = np.array(kp_coords, dtype=np.float32)

            # 更新点云范围
            if 'point_cloud_range' in state:
                self.z_min = state['point_cloud_range']['z_min']
                self.z_max = state['point_cloud_range']['z_max']

        except Exception as e:
            print(f"更新数据失败: {e}")

    def _create_visualizations(self):
        """创建可视化窗口"""
        try:
            # 创建本地可视化窗口（局部点云）
            self.local_vis = o3d.visualization.Visualizer()
            self.local_vis.create_window("局部3D点云", 800, 600)

            # 创建全局可视化窗口（全局点云）
            self.global_vis = o3d.visualization.Visualizer()
            self.global_vis.create_window("全局3D点云 (0-10米)", 800, 600)

            # 设置渲染选项
            for vis in [self.local_vis, self.global_vis]:
                vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
                vis.get_render_option().point_size = 2.0

            print("可视化窗口创建完成")

        except Exception as e:
            print(f"创建可视化窗口失败: {e}")
            sys.exit(1)

    def _update_visualizations(self):
        """更新可视化内容"""
        try:
            # 获取当前鱼类的关键点
            current_fish_name = f"fish_{self.current_state.get('current_fish_idx', 0) + 1}"
            current_keypoints = self.fish_keypoints.get(current_fish_name, {})

            # 更新局部点云（过滤范围）
            self._update_local_point_cloud(current_keypoints)

            # 更新全局点云（0-10米范围）
            self._update_global_point_cloud()

            # 更新关键点显示
            self._update_keypoints(current_keypoints)

        except Exception as e:
            print(f"更新可视化失败: {e}")

    def _update_local_point_cloud(self, current_keypoints):
        """更新局部点云显示"""
        try:
            # 移除旧的点云
            if self.local_pcd_geometry is not None:
                self.local_vis.remove_geometry(self.local_pcd_geometry)

            # 计算局部点云范围
            if current_keypoints:
                valid_depths = [kp[2] for kp in current_keypoints.values() if kp[2] > 0]
                if valid_depths:
                    depth_min = max(0, min(valid_depths) - 50)
                    depth_max = min(10000, max(valid_depths) + 50)
                    z_min_filter = -depth_max
                    z_max_filter = -depth_min
                else:
                    z_min_filter = -10000
                    z_max_filter = 0
            else:
                z_min_filter = self.z_min if hasattr(self, 'z_min') else -10000
                z_max_filter = self.z_max if hasattr(self, 'z_max') else 0

            # 过滤点云
            z_coords = self.point_cloud_data[:, 2]
            mask = (z_coords >= z_min_filter) & (z_coords <= z_max_filter)
            local_points = self.point_cloud_data[mask]

            if len(local_points) > 0:
                # 为点云着色
                colors = self._create_point_cloud_colors(local_points)

                # 创建Open3D点云
                local_pcd = o3d.geometry.PointCloud()
                local_pcd.points = o3d.utility.Vector3dVector(local_points)
                local_pcd.colors = o3d.utility.Vector3dVector(colors)

                # 添加到可视化
                self.local_vis.add_geometry(local_pcd)
                self.local_pcd_geometry = local_pcd

                print(f"更新局部点云: {len(local_points)} 个点, 范围: [{z_min_filter:.0f}, {z_max_filter:.0f}]")

        except Exception as e:
            print(f"更新局部点云失败: {e}")

    def _update_global_point_cloud(self):
        """更新全局点云显示"""
        try:
            # 移除旧的点云
            if self.global_pcd_geometry is not None:
                self.global_vis.remove_geometry(self.global_pcd_geometry)

            # 全局点云：0-10米范围
            z_coords = self.point_cloud_data[:, 2]
            global_mask = (z_coords >= -10000) & (z_coords <= 0)
            global_points = self.point_cloud_data[global_mask]

            if len(global_points) > 0:
                # 为点云着色
                colors = self._create_point_cloud_colors(global_points)

                # 创建Open3D点云
                global_pcd = o3d.geometry.PointCloud()
                global_pcd.points = o3d.utility.Vector3dVector(global_points)
                global_pcd.colors = o3d.utility.Vector3dVector(colors)

                # 添加到可视化
                self.global_vis.add_geometry(global_pcd)
                self.global_pcd_geometry = global_pcd

                print(f"更新全局点云: {len(global_points)} 个点")

        except Exception as e:
            print(f"更新全局点云失败: {e}")

    def _create_point_cloud_colors(self, points):
        """为点云创建颜色"""
        try:
            if self.color_rectify_data is None:
                return np.full((len(points), 3), [0.5, 0.5, 0.5])

            height, width = self.color_rectify_data.shape[:2]
            colors = []

            for point in points:
                x, y, z = int(round(point[0])), int(round(point[1])), point[2]
                # 由于点云x坐标被翻转，在获取图像颜色时需要翻转回原始坐标
                x_original = width - 1 - x
                if 0 <= x_original < width and 0 <= y < height:
                    color = self.color_rectify_data[y, x_original] / 255.0
                    colors.append(color)
                else:
                    colors.append([0.5, 0.5, 0.5])

            return np.array(colors)

        except Exception as e:
            print(f"创建点云颜色失败: {e}")
            return np.full((len(points), 3), [0.5, 0.5, 0.5])

    def _update_keypoints(self, current_keypoints):
        """更新关键点显示"""
        try:
            height, width = self.depth_data.shape

            # 清除旧的关键点
            for sphere in self.keypoint_spheres_local.values():
                try:
                    self.local_vis.remove_geometry(sphere)
                except:
                    pass
            for sphere in self.keypoint_spheres_global.values():
                try:
                    self.global_vis.remove_geometry(sphere)
                except:
                    pass

            self.keypoint_spheres_local.clear()
            self.keypoint_spheres_global.clear()

            # 添加新的关键点
            for kp_name, kp in current_keypoints.items():
                if kp[2] > 0:  # 只显示有深度的关键点
                    # 计算显示坐标（翻转x和z轴）
                    x_flipped = width - 1 - kp[0]
                    z_flipped = -kp[2]
                    position = [x_flipped, kp[1], z_flipped]

                    # 创建局部窗口关键点（绿色，稍大）
                    local_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                    local_sphere.translate(position)
                    local_sphere.paint_uniform_color([0, 1, 0])  # 绿色
                    self.local_vis.add_geometry(local_sphere)
                    self.keypoint_spheres_local[kp_name] = local_sphere

                    # 创建全局窗口关键点（红色，稍小）
                    global_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8)
                    global_sphere.translate(position)
                    global_sphere.paint_uniform_color([1, 0, 0])  # 红色
                    self.global_vis.add_geometry(global_sphere)
                    self.keypoint_spheres_global[kp_name] = global_sphere

            print(f"更新关键点: {len(current_keypoints)} 个关键点")

        except Exception as e:
            print(f"更新关键点失败: {e}")

    def _run_visualization_loop(self):
        """运行可视化主循环"""
        try:
            # 设置视角
            self._setup_view_controls()

            print("开始可视化主循环...")
            print("按 Ctrl+C 退出GUI")

            # 主循环
            while self.running:
                # 检查状态文件更新
                current_time = time.time()
                if current_time - self.last_state_update >= self.state_check_interval:
                    new_state = self._load_state_file()
                    if new_state and self._should_update_visualization(new_state):
                        print(f"检测到状态变化，更新可视化 (timestamp: {new_state.get('timestamp', 0):.1f})")
                        self._update_data_from_state(new_state)
                        self._update_visualizations()
                        self.current_state = new_state.copy()

                    self.last_state_update = current_time

                # 更新可视化窗口
                if self.local_vis:
                    self.local_vis.poll_events()
                    self.local_vis.update_renderer()

                if self.global_vis:
                    self.global_vis.poll_events()
                    self.global_vis.update_renderer()

                time.sleep(0.05)  # 20 FPS

        except KeyboardInterrupt:
            print("收到退出信号，正在关闭GUI...")
        except Exception as e:
            print(f"可视化循环出错: {e}")
        finally:
            self._cleanup()

    def _should_update_visualization(self, new_state):
        """判断是否需要更新可视化"""
        if not self.current_state:
            return True

        # 检查关键状态变化
        state_keys_to_check = ['current_fish_idx', 'fish_keypoints', 'point_cloud_range']

        for key in state_keys_to_check:
            if key not in self.current_state or key not in new_state:
                return True
            if self.current_state[key] != new_state[key]:
                return True

        return False

    def _update_data_from_state(self, state):
        """从状态更新内部数据"""
        try:
            # 检查是否需要加载新的帧数据
            current_frame = state.get('current_frame', '')
            if current_frame != self.current_frame and current_frame:
                print(f"GUI检测到帧变化: {self.current_frame} -> {current_frame}")
                self._load_frame_data(current_frame)
                self.current_frame = current_frame

            # 更新鱼类关键点数据
            if 'fish_keypoints' in state:
                self.fish_keypoints = {}
                for fish_name, keypoints in state['fish_keypoints'].items():
                    self.fish_keypoints[fish_name] = {}
                    for kp_name, kp_coords in keypoints.items():
                        self.fish_keypoints[fish_name][kp_name] = np.array(kp_coords, dtype=np.float32)

            # 更新鱼类bbox数据
            if 'fish_bbox' in state:
                self.fish_bbox = {}
                for fish_name, bbox_coords in state['fish_bbox'].items():
                    self.fish_bbox[fish_name] = bbox_coords

            # 更新点云范围
            if 'point_cloud_range' in state:
                self.z_min = state['point_cloud_range']['z_min']
                self.z_max = state['point_cloud_range']['z_max']

        except Exception as e:
            print(f"更新数据失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_visualizations(self):
        """更新可视化内容"""
        try:
            # 获取当前鱼类的关键点
            current_fish_name = f"fish_{self.current_state.get('current_fish_idx', 0) + 1}"
            current_keypoints = self.fish_keypoints.get(current_fish_name, {})

            # 更新局部点云（过滤范围）
            self._update_local_point_cloud(current_keypoints)

            # 更新全局点云（0-10米范围）
            self._update_global_point_cloud()

            # 更新关键点显示
            self._update_keypoints(current_keypoints)

        except Exception as e:
            print(f"更新可视化失败: {e}")

    def _update_local_point_cloud(self, current_keypoints):
        """更新局部点云显示"""
        try:
            # 移除旧的点云
            if self.local_pcd_geometry is not None:
                self.local_vis.remove_geometry(self.local_pcd_geometry)

            # 获取当前鱼类的bbox
            current_fish_name = f"fish_{self.current_state.get('current_fish_idx', 0) + 1}"
            current_bbox = self.fish_bbox.get(current_fish_name)

            if current_bbox and self.depth_data is not None:
                # 从bbox区域读取深度值来确定深度范围
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = current_bbox
                height, width = self.depth_data.shape

                # 确保bbox坐标在图像范围内
                bbox_x1 = max(0, int(bbox_x1))
                bbox_y1 = max(0, int(bbox_y1))
                bbox_x2 = min(width, int(bbox_x2))
                bbox_y2 = min(height, int(bbox_y2))

                # 从bbox区域提取深度值
                bbox_depths = self.depth_data[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
                valid_bbox_depths = bbox_depths[(bbox_depths > 0) & (bbox_depths < 10000)]

                if len(valid_bbox_depths) > 0:
                    # 根据bbox区域的深度范围确定点云加载范围
                    bbox_depth_min = np.min(valid_bbox_depths)
                    bbox_depth_max = np.max(valid_bbox_depths)

                    # 扩展范围：正负50mm
                    depth_min = max(0, bbox_depth_min - 50)
                    depth_max = min(10000, bbox_depth_max + 50)

                    # 转换为点云坐标系（z轴翻转）
                    z_min_filter = -depth_max
                    z_max_filter = -depth_min

                    print(f"GUI根据bbox [{bbox_x1}, {bbox_y1}, {bbox_x2}, {bbox_y2}] 确定深度范围: bbox深度 [{bbox_depth_min:.1f}, {bbox_depth_max:.1f}] -> 加载范围 [{depth_min:.1f}, {depth_max:.1f}]")
                else:
                    # bbox区域没有有效深度，使用状态文件中的范围
                    z_min_filter = getattr(self, 'z_min', -10000)
                    z_max_filter = getattr(self, 'z_max', 0)
                    print("GUI bbox区域没有有效深度，使用状态文件中的范围")
            else:
                # 没有bbox信息，使用状态文件中的范围
                z_min_filter = getattr(self, 'z_min', -10000)
                z_max_filter = getattr(self, 'z_max', 0)
                print("GUI没有bbox信息，使用状态文件中的点云范围")

            # 过滤点云
            z_coords = self.point_cloud_data[:, 2]
            mask = (z_coords >= z_min_filter) & (z_coords <= z_max_filter)
            local_points = self.point_cloud_data[mask]

            if len(local_points) > 0:
                # 为点云着色
                colors = self._create_point_cloud_colors(local_points)

                # 创建Open3D点云
                local_pcd = o3d.geometry.PointCloud()
                local_pcd.points = o3d.utility.Vector3dVector(local_points)
                local_pcd.colors = o3d.utility.Vector3dVector(colors)

                # 添加到可视化
                self.local_vis.add_geometry(local_pcd)
                self.local_pcd_geometry = local_pcd

                print(f"GUI更新局部点云: {len(local_points)} 个点, 范围: [{z_min_filter:.1f}, {z_max_filter:.1f}]")

        except Exception as e:
            print(f"GUI更新局部点云失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_global_point_cloud(self):
        """更新全局点云显示"""
        try:
            # 移除旧的点云
            if self.global_pcd_geometry is not None:
                self.global_vis.remove_geometry(self.global_pcd_geometry)

            # 全局点云：0-10米范围
            z_coords = self.point_cloud_data[:, 2]
            global_mask = (z_coords >= -10000) & (z_coords <= 0)
            global_points = self.point_cloud_data[global_mask]

            if len(global_points) > 0:
                # 为点云着色
                colors = self._create_point_cloud_colors(global_points)

                # 创建Open3D点云
                global_pcd = o3d.geometry.PointCloud()
                global_pcd.points = o3d.utility.Vector3dVector(global_points)
                global_pcd.colors = o3d.utility.Vector3dVector(colors)

                # 添加到可视化
                self.global_vis.add_geometry(global_pcd)
                self.global_pcd_geometry = global_pcd

                print(f"更新全局点云: {len(global_points)} 个点")

        except Exception as e:
            print(f"更新全局点云失败: {e}")

    def _update_keypoints(self, current_keypoints):
        """更新关键点显示"""
        try:
            height, width = self.depth_data.shape

            # 清除旧的关键点
            for sphere in self.keypoint_spheres_local.values():
                try:
                    self.local_vis.remove_geometry(sphere)
                except:
                    pass
            for sphere in self.keypoint_spheres_global.values():
                try:
                    self.global_vis.remove_geometry(sphere)
                except:
                    pass

            self.keypoint_spheres_local.clear()
            self.keypoint_spheres_global.clear()

            # 添加新的关键点
            for kp_name, kp in current_keypoints.items():
                if kp[2] > 0:  # 只显示有深度的关键点
                    # 计算显示坐标（翻转x和z轴）
                    x_flipped = width - 1 - kp[0]
                    z_flipped = -kp[2]
                    position = [x_flipped, kp[1], z_flipped]

                    # 创建局部窗口关键点（绿色，稍大）
                    local_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                    local_sphere.translate(position)
                    local_sphere.paint_uniform_color([0, 1, 0])  # 绿色
                    self.local_vis.add_geometry(local_sphere)
                    self.keypoint_spheres_local[kp_name] = local_sphere

                    # 创建全局窗口关键点（红色，稍小）
                    global_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8)
                    global_sphere.translate(position)
                    global_sphere.paint_uniform_color([1, 0, 0])  # 红色
                    self.global_vis.add_geometry(global_sphere)
                    self.keypoint_spheres_global[kp_name] = global_sphere

            print(f"更新关键点: {len(current_keypoints)} 个关键点")

        except Exception as e:
            print(f"更新关键点失败: {e}")

    def _setup_view_controls(self):
        """设置视角控制"""
        try:
            # 局部视图设置
            if self.local_vis:
                local_view = self.local_vis.get_view_control()
                local_view.set_front([0, -0.5, 1])
                local_view.set_lookat([0, 0, 0])
                local_view.set_up([0, -1, 0])
                local_view.set_zoom(0.8)

            # 全局视图设置
            if self.global_vis:
                global_view = self.global_vis.get_view_control()
                global_view.set_front([0, -0.3, 1])
                global_view.set_lookat([0, 0, 0])
                global_view.set_up([0, -1, 0])
                global_view.set_zoom(0.3)

        except Exception as e:
            print(f"设置视角失败: {e}")

    def _cleanup(self):
        """清理资源"""
        try:
            if self.local_vis:
                self.local_vis.destroy_window()
            if self.global_vis:
                self.global_vis.destroy_window()
            print("GUI资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def stop(self):
        """停止GUI进程"""
        print("正在停止GUI进程...")
        self.running = False

    def run(self):
        """运行GUI进程"""
        try:
            # 创建可视化窗口
            self._create_visualizations()

            # 初始化显示
            self._update_visualizations()

            # 运行主循环
            self._run_visualization_loop()

        except Exception as e:
            print(f"运行GUI进程失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()


def main():
    parser = argparse.ArgumentParser(description='独立的3D可视化GUI进程')
    parser.add_argument('--state-file', type=str, required=True,
                       help='状态文件路径')
    parser.add_argument('--camera-config', type=str, required=True,
                       help='相机配置文件路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.camera_config):
        print(f"相机配置文件不存在: {args.camera_config}")
        sys.exit(1)

    # 创建GUI实例
    gui = Standalone3DGUI(args.state_file, args.camera_config)

    # 处理键盘中断
    def signal_handler(signum, frame):
        print("收到中断信号，正在关闭GUI...")
        gui.stop()

    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行GUI
    try:
        gui.run()
    except KeyboardInterrupt:
        print("GUI进程被用户中断")
    except Exception as e:
        print(f"GUI进程运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gui.stop()


if __name__ == "__main__":
    main()
