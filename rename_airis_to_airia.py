#!/usr/bin/env python
"""一次性迁移脚本：`Airis` → `Airia`（2026-09-11）。

背景
----
设定正文用的是「Airia 指挥系统 / Airia 模块」（`story/.../ACT猫娘脚本.md:2`、
`第一版.md:181`），而代码包名是 `Airis` —— 名实不符。本脚本一次性对齐，
**全部改为 `Airia`**（含设定稿、历史文档、宣发稿）。

用法
----
    python rename_airis_to_airia.py            # dry-run：只统计，不写文件
    python rename_airis_to_airia.py --apply    # 备份 zip → 内容替换 → 目录改名
    python rename_airis_to_airia.py --verify   # 残留检查 + 全子包导入冒烟

安全设计
--------
- 显式 `encoding="utf-8"` + `newline=""`：保换行、杜绝 GBK（AGENTS.md 4.2）
- 跳过 `.venv/.git/.uv-cache/.numba-cache/.pytest_cache/__pycache__/tests/output` 与二进制后缀
- 内容替换**先于**目录改名；替换前先写备份 zip
- 本脚本自身不参与替换（否则 OLD 常量会被改写）；`_backup_*` 目录与备份 zip 也跳过

范围说明
--------
仓库**外**的 VS Code Copilot memory 进度日志（`dev-progress.md` 1 处 +
`dev-progress-archive.md` 18 处）不在本脚本扫描范围，已单独处理：
见 `docs/story-architecture-review.md` §八「引擎改名轮」。
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = "Airis"
NEW = "Airia"

SKIP_DIRS = {
    ".venv", ".git", ".uv-cache", ".numba-cache", ".pytest_cache",
    "__pycache__", "node_modules",
}
SKIP_REL = {Path("tests") / "output"}
SKIP_SUFFIX = {
    ".pyc", ".pyo", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico",
    ".npy", ".npz", ".nbc", ".nbi", ".db", ".sqlite", ".exe", ".dll", ".so",
    ".pyd", ".pdf", ".woff", ".woff2", ".ttf",
}
BACKUP_NAME = "_backup_airis_to_airia_20260911.zip"


# ---------------------------------------------------------------------------
# 文件枚举
# ---------------------------------------------------------------------------


def iter_files() -> list[Path]:
    """待处理文件（跳过排除目录 / 二进制后缀 / 本脚本自身 / 备份文件）。"""
    me = Path(__file__).resolve()
    out: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.resolve() == me:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if any(part.startswith("_backup") for part in rel.parts):
            continue
        if any(rel.is_relative_to(s) for s in SKIP_REL):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        if p.name == BACKUP_NAME:
            continue
        out.append(p)
    return out


def scan() -> list[tuple[Path, int]]:
    """[(文件, 命中数)]，仅可 UTF-8 解码的文本文件。"""
    hits: list[tuple[Path, int]] = []
    for p in iter_files():
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                txt = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        n = txt.count(OLD)
        if n:
            hits.append((p, n))
    return hits


# ---------------------------------------------------------------------------
# dry-run / apply
# ---------------------------------------------------------------------------


def dry_run() -> int:
    hits = scan()
    total = sum(n for _p, n in hits)
    print(f"[dry-run] 命中 {OLD}: {total} 处 / {len(hits)} 个文件")
    by_ext: dict[str, int] = {}
    for p, n in hits:
        by_ext[p.suffix or "(no ext)"] = by_ext.get(p.suffix or "(no ext)", 0) + n
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1]):
        print(f"    {ext:<10} {n:5} 处")
    print("\n前 20 个文件：")
    for p, n in sorted(hits, key=lambda kv: -kv[1])[:20]:
        print(f"    {n:5}  {p.relative_to(ROOT)}")
    pkg = ROOT / OLD
    print(f"\n待改名目录: {pkg}  (存在={pkg.is_dir()}, 目标已存在={(ROOT / NEW).exists()})")
    return 0


def backup() -> Path:
    dst = ROOT / BACKUP_NAME
    files = iter_files()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"[backup] {dst.name}: {len(files)} 个文件, {dst.stat().st_size / 1024:.0f} KB")
    return dst


def apply() -> int:
    hits = scan()
    total = sum(n for _p, n in hits)
    if not total:
        print(f"[apply] 无 {OLD} 命中，无需替换")
    backup()
    changed_files = 0
    changed_occ = 0
    for p, _n in hits:
        with open(p, "r", encoding="utf-8", newline="") as f:
            txt = f.read()
        new_txt = txt.replace(OLD, NEW)
        if new_txt == txt:
            continue
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(new_txt)
        changed_files += 1
        changed_occ += txt.count(OLD)
    print(f"[replace] {changed_files} 个文件 / {changed_occ} 处 {OLD} -> {NEW}")

    # 目录改名（内容替换后）
    src, dst = ROOT / OLD, ROOT / NEW
    if src.is_dir():
        if dst.exists():
            print(f"[rename] 目标已存在，跳过: {dst}")
        else:
            src.rename(dst)
            print(f"[rename] {OLD}/ -> {NEW}/")
    else:
        print(f"[rename] 源目录不存在（可能已改名）: {src}")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify() -> int:
    rc = 0
    hits = scan()
    # 区分「代码/数据残留」（必须为 0）与「文档提及」（改名说明里必然会写到旧名）
    code_hits = [(p, n) for p, n in hits if p.suffix.lower() != ".md"]
    doc_hits = [(p, n) for p, n in hits if p.suffix.lower() == ".md"]
    print(f"[verify] 代码/数据残留 {OLD}: {sum(n for _p, n in code_hits)} 处 / {len(code_hits)} 文件")
    for p, n in sorted(code_hits, key=lambda kv: -kv[1])[:20]:
        print(f"    {n:5}  {p.relative_to(ROOT)}")
    if code_hits:
        rc = 1
    print(f"[verify] 文档提及 {OLD}: {sum(n for _p, n in doc_hits)} 处 / {len(doc_hits)} 文件（改名记录，允许）")
    for p, n in sorted(doc_hits, key=lambda kv: -kv[1])[:10]:
        print(f"    {n:5}  {p.relative_to(ROOT)}")

    pkg_dir = ROOT / NEW
    if not pkg_dir.is_dir():
        print(f"[verify] FAIL 包目录不存在: {NEW}/")
        return 1

    root_pkg = importlib.import_module(NEW)
    n_ok, failed = 1, []
    for info in pkgutil.walk_packages(root_pkg.__path__, prefix=f"{NEW}."):
        try:
            importlib.import_module(info.name)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((info.name, type(exc).__name__, str(exc)[:90]))
    print(f"[verify] 导入成功 {n_ok} 个模块；失败 {len(failed)}")
    for name, kind, msg in failed[:15]:
        print(f"    FAIL {name}: {kind}: {msg}")
    if failed:
        rc = 1
    print("[verify] " + ("ALL GREEN" if rc == 0 else "存在问题"))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=f"{OLD} -> {NEW} 一次性迁移")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="执行：备份 + 替换 + 改名")
    g.add_argument("--verify", action="store_true", help="残留检查 + 导入冒烟")
    args = ap.parse_args()
    if args.apply:
        return apply()
    if args.verify:
        return verify()
    return dry_run()


if __name__ == "__main__":
    sys.exit(main())
