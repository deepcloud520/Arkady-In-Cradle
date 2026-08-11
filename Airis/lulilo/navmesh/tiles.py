"""导航/渲染 tile 分块 + 脏标记 + 后台局部重建。

算法思路参考 recastnavigation (c) 2009 Mikko Mononen, DetourTileCache，zlib License：
    https://github.com/recastnavigation/recastnavigation

对应 AGENTS.md 3.3：后台线程烘焙 + 双缓冲热交换。
设计决策（2026-08-10）：加载期统一表面提取 + 烘焙；运行时按 tile 脏处理，
后台线程局部重建表面与导航，主线程 Tick 间隙双缓冲热交换。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from Airis.lulilo.navmesh.baker import Baker, NavGraph
from Airis.lulilo.terrain.heightfield import Heightfield
from Airis.lulilo.terrain.surface import RenderData, SurfaceExtractor

logger = logging.getLogger(__name__)

# Portal 边类型：{poly_id: [(p1, p2), ...]}，p1/p2 为 (x, y)
_PortalEdges = dict[int, list[tuple[tuple[float, float], tuple[float, float]]]]


def _copy_portals(edges: _PortalEdges) -> _PortalEdges:
    """深拷贝 portal edges（避免新旧图共享可变 dict/list 引用）。"""
    return {k: list(v) for k, v in edges.items()}


@dataclass
class _Tile:
    """一个导航/渲染 tile（栅格坐标矩形，左闭右开）。"""

    gx0: int
    gy0: int
    gx1: int
    gy1: int
    dirty: bool = False


class _TileManager:
    """tile 划分 + 脏标记 + 后台局部重建（双缓冲热交换）。

    用法::

        tm = TileManager(hf, tile_size=32)

        # 加载期：全量烘焙
        tm.full_bake()

        # 运行时查询
        graph = tm.nav_graph()       # 当前 NavGraph（原子引用）
        surface = tm.render_data()   # 当前 RenderData（原子引用）

        # 破坏事件
        tm.mark_dirty(gx0, gy0, gx1, gy1)   # 标记受影响的 tile
        tm.rebuild_dirty(async_=True)        # 后台局部重建
    """

    def __init__(self, hf: Heightfield, tile_size: int = 32) -> None:
        self.hf = hf
        self.tile_size = int(tile_size)
        self._surface = SurfaceExtractor(hf)
        self._baker = Baker(hf)

        # tile 网格
        self._tiles_x = max(1, (hf.width + tile_size - 1) // tile_size)
        self._tiles_y = max(1, (hf.height + tile_size - 1) // tile_size)
        self._tiles: dict[tuple[int, int], Tile] = {}
        self._build_tiles()

        # 双缓冲：原子引用（主线程只读，后台线程生成新值后原子替换）
        self._nav_graph: NavGraph | None = None
        self._render_data: RenderData | None = None
        self._lock = threading.Lock()

        # 后台重建状态
        self._rebuilding = False
        self._rebuild_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def full_bake(self) -> None:
        """加载期全量烘焙：表面 + 导航（同步，主线程调用）。"""
        logger.info("全量烘焙开始：%dx%d 格 → %dx%d tiles",
                    self.hf.width, self.hf.height, self._tiles_x, self._tiles_y)
        rd = self._surface.extract()
        ng = self._baker.bake()
        with self._lock:
            self._render_data = rd
            self._nav_graph = ng
        logger.info("全量烘焙完成：%d 顶点, %d 多边形",
                    len(rd.vertices), len(ng.polys))

    def nav_graph(self) -> NavGraph | None:
        """返回当前生效 NavGraph（原子引用）。"""
        with self._lock:
            return self._nav_graph

    def render_data(self) -> RenderData | None:
        """返回当前生效 RenderData（原子引用）。"""
        with self._lock:
            return self._render_data

    def mark_dirty(self, gx0: int, gy0: int, gx1: int, gy1: int) -> None:
        """破坏后标记受影响 tile。

        破坏区域 [gx0, gx1) × [gy0, gy1) 命中的所有 tile 置 dirty=True。
        """
        tx0 = max(0, gx0 // self.tile_size)
        ty0 = max(0, gy0 // self.tile_size)
        tx1 = min(self._tiles_x, (gx1 + self.tile_size - 1) // self.tile_size)
        ty1 = min(self._tiles_y, (gy1 + self.tile_size - 1) // self.tile_size)

        count = 0
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                tile = self._tiles.get((tx, ty))
                if tile and not tile.dirty:
                    tile.dirty = True
                    count += 1
        logger.debug("标记脏 tile：%d 个（破坏区域 [%d,%d) × [%d,%d)）",
                     count, gx0, gx1, gy0, gy1)

    def rebuild_dirty(self, async_: bool = True) -> None:
        """重建所有脏 tile 的表面 + 导航。

        async_=True：后台线程执行，不阻塞主线程。
        async_=False：同步执行，返回时已完成。
        """
        if self._rebuilding:
            logger.debug("后台重建进行中，跳过")
            return

        dirty_tiles = [t for t in self._tiles.values() if t.dirty]
        if not dirty_tiles:
            return

        logger.info("重建 %d 个脏 tile", len(dirty_tiles))

        if async_:
            self._rebuilding = True
            self._rebuild_thread = threading.Thread(
                target=self._rebuild_worker,
                args=(dirty_tiles,),
                daemon=True,
            )
            self._rebuild_thread.start()
        else:
            self._rebuild_worker(dirty_tiles)

    def is_rebuilding(self) -> bool:
        """后台重建是否进行中。主线程 Tick 间隙检查，完成后再热交换。"""
        return self._rebuilding

    def try_swap(self) -> bool:
        """若后台重建已完成，执行双缓冲热交换。

        应在主线程 Tick 间隙调用（sim_paused=True 时）。
        返回 True 表示执行了交换。
        """
        if not self._rebuilding:
            return False
        if self._rebuild_thread is None or self._rebuild_thread.is_alive():
            return False

        # 后台线程已完成，结果存在 _pending_* 中
        with self._lock:
            if hasattr(self, '_pending_nav_graph') and self._pending_nav_graph is not None:
                self._nav_graph = self._pending_nav_graph
                self._pending_nav_graph = None
            if hasattr(self, '_pending_render_data') and self._pending_render_data is not None:
                self._render_data = self._pending_render_data
                self._pending_render_data = None

        self._rebuilding = False
        self._rebuild_thread = None

        # 清除脏标记
        for tile in self._tiles.values():
            tile.dirty = False

        logger.info("双缓冲热交换完成")
        return True

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_tiles(self) -> None:
        """按 tile_size 划分 tile 网格。"""
        for ty in range(self._tiles_y):
            for tx in range(self._tiles_x):
                gx0 = tx * self.tile_size
                gy0 = ty * self.tile_size
                gx1 = min(gx0 + self.tile_size, self.hf.width)
                gy1 = min(gy0 + self.tile_size, self.hf.height)
                self._tiles[(tx, ty)] = Tile(gx0, gy0, gx1, gy1)

    def _rebuild_worker(self, dirty_tiles: list[Tile]) -> None:
        """后台线程入口：重建所有脏 tile，结果存入 _pending_* 待主线程交换。"""
        try:
            # 收集所有脏 tile 的范围
            gx0 = min(t.gx0 for t in dirty_tiles)
            gy0 = min(t.gy0 for t in dirty_tiles)
            gx1 = max(t.gx1 for t in dirty_tiles)
            gy1 = max(t.gy1 for t in dirty_tiles)

            # 局部重建表面 + 导航
            new_rd = self._surface.extract_tile(gx0, gy0, gx1, gy1)
            new_ng = self._baker.bake_tile(gx0, gy0, gx1, gy1)

            # 合并：非脏 tile 保留旧数据，脏 tile 替换为新数据
            with self._lock:
                old_ng = self._nav_graph
                old_rd = self._render_data

            # 合并 NavGraph（简化：全量重建）
            # 注：增量合并较复杂，当前采用脏区域局部烘焙 + 旧图非脏区保留
            merged_polys, merged_adj, merged_portals = self._merge_graphs(
                old_ng, new_ng, dirty_tiles
            )

            # 暂存待交换
            self._pending_nav_graph = NavGraph(
                polys=merged_polys, adjacency=merged_adj, portal_edges=merged_portals
            )
            # 渲染数据也暂存（简化：全量 re-extract）
            self._pending_render_data = self._surface.extract()

            logger.info("后台重建完成：%d 多边形", len(merged_polys))
        except Exception:
            logger.exception("后台重建失败")

    def _merge_graphs(
        self,
        old_ng: NavGraph | None,
        new_ng: NavGraph,
        dirty_tiles: list[Tile],
    ) -> tuple[
        list[np.ndarray],
        dict[int, list[int]],
        dict[int, list[tuple[tuple[float, float], tuple[float, float]]]],
    ]:
        """合并旧导航图（非脏区）与新烘焙结果（脏区）。

        简化策略：脏 tile 覆盖区域外的多边形保留，脏区内用新多边形替换。
        跨边界邻接 + portal edges 重新计算。

        Returns:
            (all_polys, adjacency, portal_edges)
        """
        if old_ng is None or not old_ng.polys:
            return (
                list(new_ng.polys),
                dict(new_ng.adjacency),
                _copy_portals(new_ng.portal_edges),
            )

        # 收集脏区域的世界范围
        dirty_bboxes = [
            (
                t.gx0 * self.hf.cs,
                t.gy0 * self.hf.cs,
                t.gx1 * self.hf.cs,
                t.gy1 * self.hf.cs,
            )
            for t in dirty_tiles
        ]

        # 保留不在脏区域内的旧多边形
        kept_polys: list[np.ndarray] = []
        for poly in old_ng.polys:
            cx = float(np.mean(poly[:, 0]))
            cy = float(np.mean(poly[:, 1]))
            in_dirty = any(
                bx0 <= cx < bx1 and by0 <= cy < by1
                for bx0, by0, bx1, by1 in dirty_bboxes
            )
            if not in_dirty:
                kept_polys.append(poly)

        # 合并
        offset = len(kept_polys)
        all_polys = kept_polys + list(new_ng.polys)

        # 重建邻接 + portal（跨新旧边界）
        adjacency: dict[int, list[int]] = {i: [] for i in range(len(all_polys))}
        portal_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = (
            {i: [] for i in range(len(all_polys))}
        )
        # 新图内部邻接 + portal
        for ni, nbs in new_ng.adjacency.items():
            for nj in nbs:
                adjacency.setdefault(offset + ni, []).append(offset + nj)
        for ni, edges in new_ng.portal_edges.items():
            for p1, p2 in edges:
                portal_edges.setdefault(offset + ni, []).append((p1, p2))

        # 跨边界：用 shapely 求共享边
        from shapely.geometry import Polygon as ShapelyPolygon
        cs = self.hf.cs
        threshold = cs * 1.1
        shape_polys = [ShapelyPolygon(p) for p in all_polys]

        for i in range(offset):
            for j in range(offset, len(all_polys)):
                ci = np.mean(all_polys[i], axis=0)
                cj = np.mean(all_polys[j], axis=0)
                if np.linalg.norm(ci - cj) > cs * 30:
                    continue
                if shape_polys[i].distance(shape_polys[j]) >= threshold:
                    continue
                inter = shape_polys[i].intersection(shape_polys[j])
                if inter.is_empty or inter.geom_type != "LineString":
                    continue
                if inter.length < cs * 0.5:
                    continue
                coords = list(inter.coords)
                p1 = (float(coords[0][0]), float(coords[0][1]))
                p2 = (float(coords[-1][0]), float(coords[-1][1]))
                adjacency.setdefault(i, []).append(j)
                adjacency.setdefault(j, []).append(i)
                portal_edges.setdefault(i, []).append((p1, p2))
                portal_edges.setdefault(j, []).append((p2, p1))

        return all_polys, adjacency, portal_edges


# ---------------------------------------------------------------------------
# ⚠️ 废弃兼容别名（2026-08-11）
# _Tile / _TileManager 已被 SurfaceMesh 统一架构替代（terrain.surface_mesh）。
# 此处别名仅供旧测试/调试脚本兼容，新代码一律禁止使用。
# ---------------------------------------------------------------------------
Tile = _Tile
TileManager = _TileManager
