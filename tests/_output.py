"""测试输出产物管理（AGENTS.md 4.5 规则）。

- 所有测试生成的文件统一写入本目录下 `output/` 子目录
- 每次运行前调用 `clean_output_dir()` 清空旧产物，避免堆积
- 不直接向 `tests/` 根目录写任何文件
"""
from __future__ import annotations

import shutil
from pathlib import Path

# tests/ 目录（本文件所在处）
_TESTS_DIR = Path(__file__).resolve().parent
# 统一输出目录：tests/output/
OUTPUT_DIR = _TESTS_DIR / "output"


def clean_output_dir() -> Path:
    """清空并重建输出目录，返回该目录 Path。

    供测试脚本在 main() 开头调用，保证「每次运行前清空旧产物」。
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def output_path(name: str) -> Path:
    """返回输出目录下的目标路径（不保证目录存在，调用前先 clean_output_dir()）。"""
    return OUTPUT_DIR / name
