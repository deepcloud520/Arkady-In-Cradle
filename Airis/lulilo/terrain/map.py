"""2.5D 栅格地形：TerrainMap（注入式容器：Heightfield 逻辑 + 连续地表渲染场）。

对应 AGENTS.md 3.2 / 3.6 节。依赖方向：coordinates → terrain → ...（本模块仅依赖
engine.coordinates 与同包内的 heightfield / surface / surface_mesh / navmesh_builder）。

存储（2026-08-10 清理后）：
- heightfield：列式 span 高度场 —— 逻辑/碰撞/寻路唯一数据源
- continuous_ground：连续地表高度场 —— 仅渲染去量化（可选；人工编辑地形可缺省，
  SurfaceExtractor 自动回退 hf 量化层）

2026-08-12：TerrainMap 纯化为注入式容器（构造时传入已建好的 Heightfield）。
程序化地形生成（fBm + 矩形桥/隧）迁至 terrain/factory.py（TerrainFactory +
TerrainFactoryConfig，供 UI 测试场景与测试脚本使用）：战役地形为人工编辑，
generate() 纯属测试用途，不再驻留生产代码。
"""
from __future__ import annotations

import logging

import numpy as np

from Airis.lulilo.terrain.dual_contour import DualContourBuilder
from Airis.lulilo.terrain.heightfield import Heightfield
from Airis.lulilo.terrain.navmesh_builder import NavGraph, NavMeshBuilder
from Airis.lulilo.terrain.surface import SurfaceExtractor, WorldMeshRenderer
from Airis.lulilo.terrain.surface_mesh import SurfaceMesh, SurfaceMeshBuilder
from Airis.lulilo.terrain.world_mesh import WorldMesh, WorldMeshBuilder

logger = logging.getLogger(__name__)


class TerrainMap:
    """3D地形：注入式容器（Heightfield + 可选连续地表）。

    构造：``TerrainMap(heightfield, continuous_ground=None)``。地形数据由外部
    （战役人工编辑加载 / tests.terrain_factory）构建后注入；本类只负责持有与查询。
    """

    def __init__(
        self,
        heightfield: Heightfield,
        continuous_ground: np.ndarray | None = None,
        world_mesh_builder: type[WorldMeshBuilder] | None = None,
    ) -> None:
        # Heightfield 存储（逻辑/碰撞/寻路唯一数据源）
        self.heightfield = heightfield
        # 连续地面高度场（渲染层去量化用，可选；缺省时 SurfaceExtractor 回退量化层）
        self.continuous_ground = continuous_ground
        # WorldMesh 构建器（Phase A=WorldMeshBuilder 顶面 / Phase B=DualContourBuilder 完整3D）
        self._world_mesh_builder = world_mesh_builder or WorldMeshBuilder
        # SurfaceMesh 懒加载（渲染结构层共用，首次访问时构建）
        self._surface_mesh: SurfaceMesh | None = None
        # WorldMesh 懒加载（3D 世界模型：z 采样/碰撞精算，首次访问时构建）
        self._world_mesh: WorldMesh | None = None
        # 渲染用 WorldMesh（DualContour 完整 3D：顶/底/侧面），懒加载
        self._render_world_mesh: WorldMesh | None = None
        # NavGraph 懒加载（导航新管线，首次访问时构建）
        self._nav_graph: NavGraph | None = None

    @property
    def surface_mesh(self) -> SurfaceMesh:
        """获取 SurfaceMesh（GreedyMesh 面片图），仅供渲染层复用。

        ⚠️ 2026-08-12 导航已切换 nav_graph（NavMeshBuilder 新管线）；
        SurfaceMesh 仅保留给 surface.py 结构层渲染与旧测试/调试脚本。
        首次访问时从 Heightfield 构建并缓存。
        """
        if self._surface_mesh is None:
            self._surface_mesh = SurfaceMeshBuilder(self.heightfield).build()
        return self._surface_mesh

    @property
    def world_mesh(self) -> WorldMesh:
        """获取 WorldMesh（3D 面片世界模型），供物理/碰撞精算/z 采样使用。

        docs/worldmesh-3d-surface-design.md：Heightfield 管碰撞粗筛，
        WorldMesh 管世界（z 采样/精算），NavGraph 管寻路。
        首次访问时从 Heightfield 构建并缓存。

        构建器：默认 WorldMeshBuilder（Phase A 顶面）；传入
        world_mesh_builder=DualContourBuilder 启用 Phase B 完整 3D
        （顶面+底面+侧面）。
        """
        if self._world_mesh is None:
            self._world_mesh = self._world_mesh_builder(
                self.heightfield, continuous_ground=self.continuous_ground
            ).build()
        return self._world_mesh

    @property
    def nav_graph(self) -> NavGraph:
        """获取三角形 NavGraph（NavMeshBuilder 新管线），供导航层使用。

        区域生长 + 三角剖分 + 跨层垂直 portal
        （docs/navmesh-region-growth-design.md）。
        首次访问时从 Heightfield 构建并缓存。
        """
        if self._nav_graph is None:
            self._nav_graph = NavMeshBuilder(self.heightfield).build()
        return self._nav_graph

    def invalidate_geometry(self) -> None:
        """地形破坏（remove_span，如 SAL 炸桥）后失效全部懒缓存。

        下次访问 world_mesh / render_world_mesh / nav_graph 时重建
        （AGENTS.md §3.3 自由破坏 → 后台局部重建路径；本容器同步失效）。
        """
        self._world_mesh = None
        self._render_world_mesh = None
        self._nav_graph = None

    @property
    def render_world_mesh(self) -> WorldMesh:
        """获取渲染用 WorldMesh（DualContour 完整 3D：顶/底/侧面）。

        Phase D：渲染从完整 3D WorldMesh 提取（桥侧墙/桥底可见），
        与逻辑层 world_mesh（Phase A 顶面）解耦。懒加载缓存。
        """
        if self._render_world_mesh is None:
            self._render_world_mesh = DualContourBuilder(
                self.heightfield,
                continuous_ground=self.continuous_ground,
                include_ground=False,   # 渲染地面走 continuous_ground（65536 个地面 patch 是纯负担）
                build_adjacency=False,  # 渲染不寻路，跳过邻接（8s → 毫秒级）
            ).build()
        return self._render_world_mesh

    # ------------------------------------------------------------------
    # 渲染数据
    # ------------------------------------------------------------------
    def to_render_data(self) -> dict[str, np.ndarray]:
        """生成渲染网格（Phase D：从 WorldMesh 完整 3D 提取）。

        返回（渲染层消费）:
            ground_vertices (N, 9) float32 [x, y, z, r, g, b, nx, ny, nz]
            ground_indices  (M,) uint32
            bridge_vertices (N, 9) float32
            bridge_indices  (M,) uint32

        地面 = 平滑连续网格（continuous_ground 或 WorldMesh 地面导出）；
        结构 = WorldMesh 顶/底/侧面（DualContour 完整 3D）。
        """
        renderer = WorldMeshRenderer(
            self.render_world_mesh,
            self.heightfield,
            continuous_ground=self.continuous_ground,
        )
        rd = renderer.extract()

        if len(rd.vertices) == 0:
            return {
                "ground_vertices": np.zeros((0, 9), dtype=np.float32),
                "ground_indices": np.zeros((0,), dtype=np.uint32),
                "bridge_vertices": np.zeros((0, 9), dtype=np.float32),
                "bridge_indices": np.zeros((0,), dtype=np.uint32),
            }

        # 分离地面与结构：按结构顶点起始偏移
        verts = rd.vertices
        idx = rd.indices.reshape(-1, 3)
        struct_start = renderer.struct_offset
        is_struct_v = np.zeros(len(verts), dtype=bool)
        is_struct_v[struct_start:] = True
        is_bridge_tri = np.all(is_struct_v[idx], axis=1)
        ground_tris = idx[~is_bridge_tri]
        bridge_tris = idx[is_bridge_tri]

        # 地面：去重顶点
        gv, gi = self._compact_mesh(verts, ground_tris)
        bv, bi = self._compact_mesh(verts, bridge_tris)

        return {
            "ground_vertices": gv,
            "ground_indices": gi,
            "bridge_vertices": bv,
            "bridge_indices": bi,
        }

    @staticmethod
    def _compact_mesh(
        verts: np.ndarray, tri_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """从大顶点池中提取子集的紧凑表示（去重未使用顶点）。"""
        if len(tri_indices) == 0:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
        used = np.unique(tri_indices.ravel())
        old_to_new = {int(old): new for new, old in enumerate(used)}
        compact_verts = verts[used]
        compact_idx = np.array(
            [old_to_new[int(i)] for i in tri_indices.ravel()], dtype=np.uint32
        )
        return compact_verts, compact_idx
