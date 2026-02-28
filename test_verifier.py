#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from verify_fish_3d_keypoints import Fish3DKeypointVerifier

# 创建验证器实例
verifier = Fish3DKeypointVerifier(
    'fish_dataset/annotations/labelme/fishdata.json',
    'fish_dataset/depths',
    'fish_dataset/camera_configs/mocha_stereo_params.yaml'
)

print(f'找到帧数: {len(verifier.frames)}')
print(f'帧列表: {verifier.frames}')
print(f'当前帧索引: {verifier.current_frame_idx}')

if verifier.frames:
    print(f'第一帧: {verifier.frames[0]}')
    # 尝试加载当前帧
    try:
        verifier._load_current_frame()
        print('成功加载当前帧')
        print(f'鱼类数量: {len(verifier.fish_names)}')
        print(f'鱼类列表: {verifier.fish_names}')
    except Exception as e:
        print(f'加载帧失败: {e}')
        import traceback
        traceback.print_exc()

