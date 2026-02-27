"""
复制脚本：从两个源目录复制同名PNG文件到目标目录
源目录1: E:\图片\0509_0_redTilmpa\clahe_color\processing\rectified_R
源目录2: E:\图片\0703_6_Huguang\color_clahe\processing\rectified_R
参考目录: I:\fish_dataset_3\images
目标目录: J:\fish_dataset_3\images\right
"""

import os
import shutil
from pathlib import Path


def copy_images():
    """从两个源目录复制同名PNG文件到目标目录"""
    source_dir1 = Path(r"E:\图片\0509_0_redTilmpa\clahe_color\processing\rectified_R")
    source_dir2 = Path(r"E:\图片\0703_6_Huguang\color_clahe\processing\rectified_R")
    reference_dir = Path(r"I:\fish_dataset_3\images")
    target_dir = Path(r"J:\fish_dataset_3\images\right")
    
    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir}")
    
    # 获取参考目录中的所有PNG文件
    png_files = list(reference_dir.glob("*.png"))
    
    copied_count = 0
    not_found_count = 0
    
    for png_file in png_files:
        file_name = png_file.name
        src1 = source_dir1 / file_name
        src2 = source_dir2 / file_name
        dst = target_dir / file_name
        
        if src1.exists():
            shutil.copy2(src1, dst)
            print(f"已复制: {file_name} (来源: source1)")
            copied_count += 1
        elif src2.exists():
            shutil.copy2(src2, dst)
            print(f"已复制: {file_name} (来源: source2)")
            copied_count += 1
        else:
            print(f"未找到: {file_name}")
            not_found_count += 1
    
    print(f"\n汇总:")
    print(f"  已复制: {copied_count} 个文件")
    print(f"  未找到: {not_found_count} 个文件")


if __name__ == "__main__":
    copy_images()
