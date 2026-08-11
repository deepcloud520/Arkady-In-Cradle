"""列式高度场（span）核心 —— recast rcHeightfield 的 NumPy/Numba 重写。

算法思路参考 recastnavigation (c) 2009 Mikko Mononen，zlib License：
    https://github.com/recastnavigation/recastnavigation
本实现为独立重写（预分配 SoA 数组，非 C++ 移植），仅保留算法思想。

对应 AGENTS.md 3.2：以「每列 span 实心段」统一表示 地面/桥/隧道/建筑，
替代 heightmap + layer_map/layer_alt 的双层存储。不同"图层"由 span 在
列内的位置自然决定，无需 LAYER_MAP/ENTRY_MASK 特判：
- 地面  = 最低实心段顶面
- 桥    = 高位悬空实心段（桥下无实心 → 单位自然通过，可被炸断）
- 隧道  = 上下实心段夹空隙
- 建筑  = 多层实心段
"""
from __future__ import annotations

import numpy as np
from numba import njit

# 单列最大 span 数（地面1 + 桥面1~2 + 隧道1~2 + 建筑若干，8 足够）
MAX_SPANS = 8

# span 列内字段索引（第 4 维）
_ZMIN = 0
_ZMAX = 1
_AREA = 2
_MAT = 3

# 材质/区域枚举（对应原 layer 语义，按需扩展）
MAT_TERRAIN = 0   # 地形/山体（低破坏优先级）
MAT_BRIDGE = 1    # 桥面板（可破坏）
MAT_BUILDING = 2  # 建筑墙/楼板（可破坏）
MAT_TUNNEL = 3    # 隧道顶/底（可破坏）

# 无实心段时的哨兵高度
NO_SURFACE = -1e9


@njit(cache=True)
def _surface_at(spans: np.ndarray, count: np.ndarray, col: int, layer: int) -> float:
    """返回第 layer 个实心段的顶面**层索引**，越界返回 -1（哨兵）。"""
    n = int(count[col])
    if layer < 0 or layer >= n:
        return -1.0
    return float(spans[col, layer, _ZMAX])


@njit(cache=True)
def surface_z_ground(spans: np.ndarray, count: np.ndarray, col: int) -> float:
    """地面层站位高度 = 最低实心段顶面。"""
    return _surface_at(spans, count, col, 0)


@njit(cache=True)
def is_solid_at(spans: np.ndarray, count: np.ndarray, col: int, z: float) -> bool:
    """该列在高度 z 处是否被实心段覆盖（弹道/移动阻挡判定）。

    桥下悬空：z 落在桥面段之下、地面段之上时返回 False。
    """
    n = int(count[col])
    for i in range(n):
        if z >= float(spans[col, i, _ZMIN]) and z < float(spans[col, i, _ZMAX]):
            return True
    return False


class Heightfield:
    """列式高度场：预分配 SoA 数组（W*H 列 × MAX_SPANS 段）。

    每列一个升序实心段列表（按 zmin 排序），spans[col, i] = [zmin, zmax, area, mat]。
    高频查询走 Numba；add/remove（加载期与破坏事件，低频）用 Python 实现，清晰优先。
    """

    def __init__(
        self,
        width: int,
        height: int,
        cs: float,
        ch: float,
        z_min: float,
        z_max: float,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.cs = float(cs)
        self.ch = float(ch)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        n_cols = self.width * self.height
        self.spans = np.zeros((n_cols, MAX_SPANS, 4), dtype=np.uint16)
        self.count = np.zeros(n_cols, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 坐标/高度编码
    # ------------------------------------------------------------------
    def _col(self, gx: int, gy: int) -> int:
        return gx + gy * self.width

    def _to_layer(self, z: float) -> int:
        """世界 z → 层索引（uint16，recast 的 span 单位）。"""
        return int(np.clip(round((z - self.z_min) / self.ch), 0, 65535))

    def _to_z(self, layer: int) -> float:
        return self.z_min + float(layer) * self.ch

    # ------------------------------------------------------------------
    # 加载期：添加实心段（可任意合并，复刻 rcAddSpan 思路）
    # ------------------------------------------------------------------
    def add_span(
        self,
        gx: int,
        gy: int,
        z0: float,
        z1: float,
        area: int = 0,
        mat: int = MAT_TERRAIN,
    ) -> bool:
        """把 [z0, z1) 实心段加入 (gx, gy) 列，与重叠段合并。

        - 零尺寸段忽略（对应 rcAddSpan 的 early return）
        - 重叠段取并集；area 取优先级高（数值大）者，mat 取后加入者
        - 列满（> MAX_SPANS）时保留最低层段（地面+桥优先）
        """
        col = self._col(gx, gy)
        smin = self._to_layer(z0)
        smax = self._to_layer(z1)
        if smax <= smin:
            return True
        n = int(self.count[col])
        segs = [
            [
                int(self.spans[col, i, _ZMIN]),
                int(self.spans[col, i, _ZMAX]),
                int(self.spans[col, i, _AREA]),
                int(self.spans[col, i, _MAT]),
            ]
            for i in range(n)
        ]
        segs.append([smin, smax, area, mat])
        segs.sort(key=lambda s: s[0])
        # 合并重叠/相邻段
        merged: list[list[int]] = []
        for s in segs:
            # 仅当 z 相邻且材质相同时合并（防止地面 BRIDGE 贴地段吞掉 TERRAIN）
            if merged and s[0] <= merged[-1][1] and s[3] == merged[-1][3]:
                if s[1] > merged[-1][1]:
                    merged[-1][1] = s[1]
                if s[2] > merged[-1][2]:
                    merged[-1][2] = s[2]
            else:
                merged.append(list(s))
        if len(merged) > MAX_SPANS:
            merged = merged[:MAX_SPANS]
        self._write_col(col, merged)
        return True

    # ------------------------------------------------------------------
    # 运行时：破坏（截断/删除与 [z0, z1) 重叠的部分）
    # ------------------------------------------------------------------
    def remove_span(self, gx: int, gy: int, z0: float, z1: float) -> bool:
        """挖掉 [z0, z1) 覆盖的实心段（自由破坏的核心原语）。

        每个受影响的段被拆成 [段底, z0) 与 [z1, 段顶) 两段（若仍有残留）。
        """
        col = self._col(gx, gy)
        rmin = self._to_layer(z0)
        rmax = self._to_layer(z1)
        if rmax <= rmin:
            return True
        n = int(self.count[col])
        keep: list[list[int]] = []
        for i in range(n):
            a = int(self.spans[col, i, _ZMIN])
            b = int(self.spans[col, i, _ZMAX])
            area = int(self.spans[col, i, _AREA])
            mat = int(self.spans[col, i, _MAT])
            if b <= rmin or a >= rmax:
                keep.append([a, b, area, mat])
            else:
                if a < rmin:
                    keep.append([a, rmin, area, mat])
                if b > rmax:
                    keep.append([rmax, b, area, mat])
        keep.sort(key=lambda s: s[0])
        if len(keep) > MAX_SPANS:
            keep = keep[:MAX_SPANS]
        self._write_col(col, keep)
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def surface_z(self, gx: int, gy: int, layer: int = 0) -> float:
        """第 layer 个实心段顶面的**世界高度**（layer=0 地面，1 桥面，...）。

        无该层段时返回 NO_SURFACE。
        """
        li = float(_surface_at(self.spans, self.count, self._col(gx, gy), layer))
        if li < 0.0:
            return NO_SURFACE
        return self._to_z(int(li))

    def ground_z(self, gx: int, gy: int) -> float:
        """地面站位世界高度（与 surface_z(layer=0) 相同，语义更明确）。"""
        li = float(surface_z_ground(self.spans, self.count, self._col(gx, gy)))
        if li < 0.0:
            return NO_SURFACE
        return self._to_z(int(li))

    def solid_at(self, gx: int, gy: int, z: float) -> bool:
        """(gx, gy) 处在**世界高度** z 是否被实心段覆盖。

        世界 z → 所在层采用向下取整（floor）：世界 13.5 落在层 23，
        段 [23, 24) 排他上界 24 不再误判。
        """
        layer = int((z - self.z_min) / self.ch)  # z >= z_min 时等价 floor
        return bool(is_solid_at(self.spans, self.count, self._col(gx, gy), float(layer)))

    def span_count(self, gx: int, gy: int) -> int:
        return int(self.count[self._col(gx, gy)])

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _write_col(self, col: int, segs: list[list[int]]) -> None:
        for i in range(MAX_SPANS):
            if i < len(segs):
                self.spans[col, i, _ZMIN] = segs[i][0]
                self.spans[col, i, _ZMAX] = segs[i][1]
                self.spans[col, i, _AREA] = segs[i][2]
                self.spans[col, i, _MAT] = segs[i][3]
            else:
                self.spans[col, i, :] = 0
        self.count[col] = len(segs)
