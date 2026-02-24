#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练集目录按「同名左图」匹配，将 labelme JSON 复制到各 fish_dataset_N/annotations/labelme/left。

用法:
    python copy_labelme_from_training.py [--source-dir SOURCE_JSON_DIR] [--base-dir BASE] [--dry-run]
"""

import argparse
import os
import shutil
from pathlib import Path


# 默认路径（训练集在 7z 内时的实际路径）
DEFAULT_SOURCE_JSON = "/Volumes/Junf/0和其他.7z/训练集/json"
DEFAULT_BASE_DIR = "/Volumes/Junf"
DATASET_RANGE = (0, 17)  # fish_dataset_0 .. fish_dataset_16
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def copy_labelme_from_training(
    source_json_dir: str,
    base_dir: str,
    dataset_start: int = 0,
    dataset_end: int = 17,
    dry_run: bool = False,
) -> None:
    """
    对每个 fish_dataset_N，根据 rectified_L 下左图文件名在 source_json_dir 中找同名 .json，
    复制到 fish_dataset_N/annotations/labelme/left/。

    Args:
        source_json_dir: 训练集 json 目录（内含与左图同名的 .json）。
        base_dir: fish_dataset_* 所在根目录。
        dataset_start: 起始编号（含）。
        dataset_end: 结束编号（不含），如 17 表示 0..16。
        dry_run: 若为 True 仅打印将要复制的文件，不写入。
    """
    source = Path(source_json_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"源 json 目录不存在: {source}")

    base = Path(base_dir)
    total_copied = 0
    for n in range(dataset_start, dataset_end):
        rectified_L_dir = base / f"fish_dataset_{n}" / "rectified_L"
        dest_dir = base / f"fish_dataset_{n}" / "annotations" / "labelme" / "left"
        if not rectified_L_dir.is_dir():
            print(f"[{n}] 跳过: 无 rectified_L 目录 {rectified_L_dir}")
            continue

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for p in rectified_L_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = p.stem
            src_json = source / f"{stem}.json"
            if not src_json.is_file():
                continue
            dst_json = dest_dir / f"{stem}.json"
            if dry_run:
                print(f"  [dry-run] {src_json} -> {dst_json}")
            else:
                shutil.copy2(src_json, dst_json)
            copied += 1

        total_copied += copied
        print(f"[{n}] fish_dataset_{n}: 复制 {copied} 个 json -> annotations/labelme/left")

    print(f"合计复制: {total_copied} 个 json")
    if dry_run:
        print("(dry-run 未实际写入)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按左图同名将训练集 json 复制到各 fish_dataset_N/annotations/labelme/left"
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_JSON,
        help=f"训练集 json 目录，默认: {DEFAULT_SOURCE_JSON}",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"fish_dataset_* 根目录，默认: {DEFAULT_BASE_DIR}",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DATASET_RANGE[0],
        help="起始 fish_dataset 编号（含）",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DATASET_RANGE[1],
        help="结束 fish_dataset 编号（不含），如 17 表示 0..16",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要复制的文件，不实际复制",
    )
    args = parser.parse_args()

    copy_labelme_from_training(
        source_json_dir=args.source_dir,
        base_dir=args.base_dir,
        dataset_start=args.start,
        dataset_end=args.end,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
