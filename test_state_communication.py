#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试状态文件通信机制
"""

import os
import sys
import json
import numpy as np
import time
import tempfile
import threading

def test_state_file_communication():
    """测试状态文件通信机制"""
    print("测试状态文件通信机制...")
    try:
        # 创建临时状态文件目录
        temp_dir = os.path.join(tempfile.gettempdir(), 'fish_keypoints_gui_test')
        os.makedirs(temp_dir, exist_ok=True)
        state_file_path = os.path.join(temp_dir, 'test_state.json')

        # 创建测试状态数据
        test_state = {
            'timestamp': time.time(),
            'current_frame': 'test_frame.png',
            'current_fish_idx': 0,
            'current_kp_idx': 1,
            'fish_keypoints': {
                'fish_1': {
                    'head': [100.0, 200.0, 1000.0],
                    'tail': [150.0, 250.0, 1100.0]
                }
            },
            'point_cloud_range': {
                'z_min': -1200.0,
                'z_max': -900.0
            }
        }

        # 写入状态文件
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_state, f, indent=2)
        print(f"[OK] 状态文件写入成功: {state_file_path}")

        # 读取状态文件
        with open(state_file_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)

        # 验证数据完整性
        assert loaded_state['current_frame'] == 'test_frame.png'
        assert loaded_state['current_fish_idx'] == 0
        assert len(loaded_state['fish_keypoints']['fish_1']) == 2
        assert loaded_state['point_cloud_range']['z_min'] == -1200.0

        print("[OK] 状态文件读取和验证成功")

        # 测试状态更新
        test_state['timestamp'] = time.time()
        test_state['current_fish_idx'] = 1

        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_state, f, indent=2)

        with open(state_file_path, 'r', encoding='utf-8') as f:
            updated_state = json.load(f)

        assert updated_state['current_fish_idx'] == 1
        print("[OK] 状态文件更新测试成功")

        # 清理测试文件
        os.remove(state_file_path)
        os.rmdir(temp_dir)
        print("[OK] 测试文件清理完成")

        return True

    except Exception as e:
        print(f"[FAIL] 状态文件通信测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standalone_gui_import():
    """测试独立GUI模块导入"""
    print("\n测试独立GUI模块导入...")
    try:
        import standalone_gui
        print("[OK] standalone_gui模块导入成功")

        # 测试类实例化（使用模拟路径）
        gui = standalone_gui.Standalone3DGUI(
            state_file_path="dummy_path.json",
            camera_config_path="fish_dataset/camera_configs/mocha_stereo_params.yaml"
        )
        print("[OK] Standalone3DGUI类实例化成功")

        return True

    except Exception as e:
        print(f"[FAIL] 独立GUI模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_process_initialization():
    """测试主进程初始化"""
    print("\n测试主进程初始化...")
    try:
        # 添加项目路径
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        sys.path.insert(0, lib_path)

        from verify_fish_3d_keypoints import Fish3DKeypointVerifier

        # 创建验证器实例（不启动GUI）
        verifier = Fish3DKeypointVerifier(
            'fish_dataset/annotations/labelme/fishdata.json',
            'fish_dataset/depths',
            'fish_dataset/camera_configs/mocha_stereo_params.yaml'
        )

        print("[OK] 主进程初始化成功")
        print(f"  - 找到 {len(verifier.frames)} 个帧")
        print(f"  - 状态文件路径: {verifier.state_file_path}")
        print(f"  - 鱼类数量: {len(verifier.fish_names)}")

        # 验证状态文件存在
        if verifier.state_file_path and os.path.exists(verifier.state_file_path):
            print("[OK] 状态文件已创建")
        else:
            print("[FAIL] 状态文件创建失败")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] 主进程初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试状态文件通信和独立GUI系统...")
    print("=" * 60)

    success_count = 0
    total_tests = 3

    if test_state_file_communication():
        success_count += 1

    if test_standalone_gui_import():
        success_count += 1

    if test_main_process_initialization():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")

    if success_count == total_tests:
        print("[SUCCESS] 所有测试通过！状态文件通信系统工作正常。")
        print("\n现在您可以运行主程序:")
        print("python verify_fish_3d_keypoints.py --dataset_root fish_dataset --camera_config fish_dataset/camera_configs/mocha_stereo_params.yaml")
        print("\n然后点击 'GUI 3D Windows' 按钮启动独立的3D可视化窗口。")
    else:
        print("[ERROR] 部分测试失败，请检查错误信息。")
