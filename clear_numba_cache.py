"""自动清除 Numba和Python 缓存脚本。

Numba 的 ``@njit(cache=True)`` 会把编译产物写到各模块 ``__pycache__`` 目录下，
文件名为 ``<module>.<funcname>-<lineno>.py<ver>.<n>.nbc`` / ``.nbi``。
改动签名、常量或升级 Numba 版本后，旧缓存会导致「缓存与代码不一致」的诡异报错，
本脚本用于一键清空这些缓存（可选的 .pyc 一并清除）。

用法::

    uv run python clear_numba_cache.py            # 预览并交互确认后删除
    uv run python clear_numba_cache.py -n         # 只预览，不删除
    uv run python clear_numba_cache.py -y         # 跳过确认直接删除
    uv run python clear_numba_cache.py --all      # 连同 .pyc 一起清（含整个 __pycache__）

退出码: 0 = 成功; 1 = 有文件删除失败或发生异常。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 项目根目录（脚本所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# Numba 缓存文件后缀（.nbi 是索引，.nbc 是编译产物）
NUMBA_SUFFIXES = {".nbc", ".nbi"}
# 旧版本 Numba 曾以目录形式缓存（模块同目录下的 *.nbc 目录），一并兜底
NUMBA_DIR_SUFFIX = ".nbc"

logger = logging.getLogger("clear_numba_cache")


def _setup_logger(verbose: bool = False) -> None:
    """配置模块级 logger（无 handler 时初始化一次）。"""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-5s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def scan_roots(roots: list[str]) -> list[Path]:
    """解析用户传入的扫描根目录（相对项目根），校验存在性。"""
    dirs: list[Path] = []
    for raw in roots:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.resolve()
        if not p.is_dir():
            logger.warning("扫描根目录不存在，已跳过: %s", p)
            continue
        dirs.append(p)
    return dirs


def find_numba_caches(roots: list[Path]) -> tuple[list[Path], list[Path]]:
    """收集 Numba 缓存文件与缓存目录。

    Returns:
        (文件列表, 目录列表)。目录是旧版 Numba 的 ``*.nbc`` 目录。
    """
    files: list[Path] = []
    dirs: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in NUMBA_SUFFIXES:
                    files.append(path)
                elif path.is_dir() and path.name.lower().endswith(NUMBA_DIR_SUFFIX):
                    dirs.append(path)
            except OSError:
                continue
    return files, dirs


def find_pyc_dirs(roots: list[Path]) -> list[Path]:
    """收集所有 __pycache__ 目录（--all 时连同 .pyc 一起清除）。"""
    dirs: list[Path] = []
    for root in roots:
        for path in root.rglob("__pycache__"):
            if path.is_dir():
                dirs.append(path)
    return dirs


def _delete(path: Path) -> bool:
    """删除单个文件/目录，失败时记录并返回 False。"""
    try:
        if path.is_dir():
            for child in path.iterdir():
                _delete(child)
            path.rmdir()
        else:
            path.unlink()
        return True
    except OSError as exc:
        # Windows 上若有 Python 进程占用缓存文件，删除会失败
        logger.error("删除失败 %s: %s", path, exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="自动清除 Numba 缓存（.nbc / .nbi），可选连同 .pyc 一起清空。"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="只列出将删除的缓存，不实际删除。",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="跳过交互确认，直接删除。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="连同各模块 __pycache__ 里的 .pyc 一并清除。",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["Airis"],
        help="要扫描的根目录（相对项目根，可多个）。默认: Airis",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出 DEBUG 日志。",
    )
    args = parser.parse_args()

    _setup_logger(args.verbose)

    roots = scan_roots(args.roots)
    if not roots:
        logger.error("没有可扫描的根目录，退出。")
        return 1

    numba_files, numba_dirs = find_numba_caches(roots)
    targets: list[Path] = list(numba_files) + list(numba_dirs)

    if args.all:
        targets += find_pyc_dirs(roots)

    if not targets:
        logger.info("未发现任何 Numba 缓存，无需清理。")
        return 0

    total_bytes = sum(p.stat().st_size for p in targets if p.is_file())
    logger.info(
        "待清理: %d 个文件 / %d 个目录，共 %.2f MB",
        len([t for t in targets if t.is_file()]),
        len([t for t in targets if t.is_dir()]),
        total_bytes / 1024 / 1024,
    )
    for t in sorted(targets, key=lambda p: str(p).lower()):
        logger.info("  - %s", t.relative_to(PROJECT_ROOT))

    if args.dry_run:
        logger.info("（dry-run）未删除任何文件。")
        return 0

    if not args.yes:
        answer = input("确认删除以上缓存？[y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            logger.info("已取消，未删除任何文件。")
            return 0

    failed = sum(0 if _delete(t) else 1 for t in targets)
    if failed:
        logger.error("有 %d 个条目删除失败（可能被运行中的进程占用），请关闭相关 Python 进程后重试。", failed)
        return 1

    logger.info("Numba 缓存已清除完毕 🐱")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
