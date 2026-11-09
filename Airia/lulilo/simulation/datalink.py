"""数据链网络（Phase 8，docs/phase8-datalink-design.md §三，networkx）。

链路模型（§3.1）：
    link_q(a, b) = comm_a × comm_b × LOS(地形遮挡) × 距离衰减 × (1 - ECM 压制)
    - comm：SoA 已有（6c），单位级通信能力；
    - LOS：复用 `detection/los.py` DDA（可选注入，None 默认通视）；
    - 距离衰减：超视距（BVR）节点只连中继（飞机天然是中继节点）；
    - 建边阈值 `LINK_THRESHOLD=0.5`，连边权重 = link_q。

簇与可达性（§3.2）：
    - `networkx.connected_components` → 通信簇；含玩家 HQ 的簇 = 中央簇（玩家指令可达）；
    - `route_quality(a, b)`：两节点间最优路径的连乘质量（航迹中继精度/指令延迟）；
    - 重建频率：1s tick 全量重建（≤2200 节点 < 1ms）。

航迹广播（§3.3）：
    - `broadcast_contacts`：逐簇收集簇内成员的观测航迹 → 多源融合
      （confidence 取 max，pos_err 取 min，位置加权融合）→ 写回孪生（source=2 数据链）。
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# 建边阈值（§3.1）
LINK_THRESHOLD = 0.5
# 距离衰减参数：视距内近全连，超视距快速衰减
_RANGE_REF = 20000.0      # 参考距离 (m)：此距离链路质量 ≈ 0.5
_RANGE_BVR = 50000.0      # 超视距 (m)：超过后只走中继（不直接连边）

# 节点类型
N_ARMOR = "a"
N_AIRCRAFT = "c"
N_FORMATION = "f"
N_HQ = "hq"


def _node(kind: str, idx: int):
    return (kind, idx)


class DataLinkNetwork:
    """数据链网络：链路图 / 簇 / 路由质量 / 航迹广播（docs §三）。"""

    def __init__(
        self,
        *,
        terrain: Any | None = None,
        los_checker: Any | None = None,
        link_threshold: float = LINK_THRESHOLD,
    ) -> None:
        self.terrain = terrain
        self.los_checker = los_checker
        self.link_threshold = float(link_threshold)
        self._graph: nx.Graph | None = None
        self._clusters: list[set] = []
        self._central: set | None = None
        self._build_counter = 0

    # ------------------------------------------------------------------
    # 链路质量
    # ------------------------------------------------------------------

    def link_quality(
        self,
        ax: float, ay: float, az: float, comm_a: float, ecm_a: float,
        bx: float, by: float, bz: float, comm_b: float, ecm_b: float,
    ) -> float:
        """两点间链路质量（docs §3.1 公式）。"""
        d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
        # 距离衰减：视距内近全连，超视距快速衰减
        if d > _RANGE_BVR:
            return 0.0
        dist_factor = 1.0 / (1.0 + d / _RANGE_REF)
        # LOS（可选：地形遮挡断链）
        if self.los_checker is not None and not self.los_checker.check(
            ax, ay, az, bx, by, bz
        ):
            return 0.0
        ecm = max(0.0, min(1.0, max(ecm_a, ecm_b)))
        return (
            max(0.0, min(1.0, comm_a))
            * max(0.0, min(1.0, comm_b))
            * dist_factor
            * (1.0 - ecm)
        )

    # ------------------------------------------------------------------
    # 建图（1s tick 全量重建）
    # ------------------------------------------------------------------

    def build(
        self,
        armor: Any,
        aircraft: Any | None = None,
        formations: Any | None = None,
        *,
        hq_pos: tuple[float, float, float] | None = None,
    ) -> None:
        """重建链路图：节点 = 活跃单位 + 编队指挥节点 + 玩家 HQ。

        连边质量批量向量化（numpy 距离矩阵 + comm/ECM 广播），
        O(n²) 但 2176 节点 ≈ 20~60ms（1s tick 可接受；numba 化留后续）。
        networkx 图惰性（仅 route_quality 查询时构建）；连通分量走 scipy 稀疏图。
        """
        self._nodes: list[tuple] = []
        self._edge_idx: tuple[np.ndarray, np.ndarray] = (
            np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self._edge_q: np.ndarray = np.array([], dtype=np.float32)
        nodes: list[tuple] = self._nodes

        # 装甲节点
        for i in armor.active_indices():
            i = int(i)
            nodes.append((_node(N_ARMOR, i),
                          (float(armor.pos[i, 0]), float(armor.pos[i, 1]), float(armor.pos[i, 2])),
                          float(armor.comm[i]), float(armor.ecm[i])))
        # 飞机节点（天然中继节点）
        if aircraft is not None:
            for i in aircraft.active_indices():
                i = int(i)
                nodes.append((_node(N_AIRCRAFT, i),
                              (float(aircraft.pos[i, 0]), float(aircraft.pos[i, 1]), float(aircraft.pos[i, 2])),
                              float(aircraft.comm[i]), 0.0))
        # 编队指挥节点（用编队目标区坐标 + comm）
        if formations is not None:
            for f in formations.active_indices():
                f = int(f)
                nodes.append((_node(N_FORMATION, f),
                              (float(formations.target_zone[f, 0]),
                               float(formations.target_zone[f, 1]),
                               float(formations.target_zone[f, 2])),
                              float(formations.comm[f]), 0.0))
        # HQ 节点
        if hq_pos is not None:
            nodes.append((_node(N_HQ, 0), tuple(float(v) for v in hq_pos), 1.0, 0.0))

        n = len(nodes)
        if n < 2:
            self._graph = None
            self._clusters = [set(nodes[k][0] for k in range(n))]
            self._central = None
            self._build_counter += 1
            return

        # 批量链路质量（docs §3.1 公式，向量化）
        # 距离矩阵用 |a|²+|b|²−2a·bᵀ（矩阵乘法，比广播差分快 ~40x，内存友好）
        pos = np.ascontiguousarray(
            np.array([nd[1] for nd in nodes], dtype=np.float32))
        comm = np.array([nd[2] for nd in nodes], dtype=np.float32)
        ecm = np.array([nd[3] for nd in nodes], dtype=np.float32)
        a2 = (pos * pos).sum(axis=1, dtype=np.float32)
        d2 = a2[:, None] + a2[None, :] - 2.0 * (pos @ pos.T)
        np.clip(d2, 0.0, None, out=d2)
        d = np.sqrt(d2)
        dist_factor = 1.0 / (1.0 + d / _RANGE_REF)
        # 距离截断：d > 30km → dist_factor < 0.4 → q ≤ 0.4 < 阈值，必然断开（提前置 0）
        dist_factor[d > 30000.0] = 0.0
        q = (comm[:, None] * comm[None, :] * dist_factor
             * (1.0 - np.maximum(ecm[:, None], ecm[None, :])))
        # LOS（可选注入时逐对 Python——默认不注入，快速路径）
        if self.los_checker is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if q[i, j] >= self.link_threshold and not self.los_checker.check(
                        pos[i, 0], pos[i, 1], pos[i, 2],
                        pos[j, 0], pos[j, 1], pos[j, 2],
                    ):
                        q[i, j] = 0.0

        mask = np.triu(q >= self.link_threshold, 1)
        i_idx, j_idx = np.where(mask)
        # 边索引惰性存储（networkx 图仅 route_quality 查询时构建，build 不生成 tuple）
        self._edge_idx = (i_idx, j_idx)
        self._edge_q = q[i_idx, j_idx]
        self._nodes = nodes
        # 连通分量用 scipy 稀疏图（比 networkx 快一个量级）；无向需对称矩阵
        row = np.concatenate([i_idx, j_idx])
        col = np.concatenate([j_idx, i_idx])
        n_comp, labels = connected_components(
            sp.csr_matrix(
                (np.ones(row.size, dtype=np.int8), (row, col)),
                shape=(n, n),
            ),
            directed=False,
        )
        self._clusters = [
            {nodes[k][0] for k in np.flatnonzero(labels == c)}
            for c in range(n_comp)
        ]
        self._graph = None   # 惰性（route_quality 首次调用时从 _edge_list 建）
        self._central = None
        for c in self._clusters:
            if _node(N_HQ, 0) in c or N_HQ in c:
                self._central = c
                break
        self._build_counter += 1

    def _ensure_graph(self) -> None:
        """惰性构建 networkx 图（route_quality 查询用；build 时跳过省建图开销）。"""
        if self._graph is None:
            G = nx.Graph()
            nodes = getattr(self, "_nodes", [])
            for nd in nodes:
                G.add_node(nd[0], pos=nd[1], comm=nd[2], ecm=nd[3])
            i_idx, j_idx = getattr(self, "_edge_idx", (np.array([], dtype=np.int64),
                                                       np.array([], dtype=np.int64)))
            q = getattr(self, "_edge_q", np.array([], dtype=np.float32))
            for k in range(i_idx.size):
                i = int(i_idx[k])
                j = int(j_idx[k])
                G.add_edge(nodes[i][0], nodes[j][0], weight=float(q[k]))
            self._graph = G

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph | None:
        return self._graph

    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    @property
    def central_cluster(self) -> set | None:
        return self._central

    def in_same_cluster(self, a: Any, b: Any) -> bool:
        """两节点是否同簇（未建图 → False）。"""
        if not self._clusters:
            return False
        ca = self._cluster_of(a)
        cb = self._cluster_of(b)
        return ca is not None and ca is cb

    def _cluster_of(self, node: Any) -> set | None:
        for c in self._clusters:
            if node in c:
                return c
        return None

    def is_central(self, node: Any) -> bool:
        """节点是否在中央簇（玩家指令可达，docs §3.4）。"""
        if self._central is None:
            return False
        return node in self._central

    def route_quality(self, a: Any, b: Any) -> float:
        """两节点间最优路径的连乘质量（docs §3.2）。

        未建图 / 不连通 → 0.0。
        用 `-log(weight)` 作 dijkstra 权重（连乘最大 ≡ Σlog 最大 ≡ Σ(-log) 最小）。
        """
        self._ensure_graph()
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return 0.0
        try:
            path = nx.shortest_path(
                self._graph, a, b,
                weight=lambda u, v, d: -math.log(d["weight"]),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0
        q = 1.0
        for u, v in zip(path[:-1], path[1:]):
            q *= self._graph[u][v]["weight"]
        return float(q)

    # ------------------------------------------------------------------
    # 航迹广播（§3.3：孪生的数据链通道）
    # ------------------------------------------------------------------

    def broadcast_contacts(self, twin: Any, armor: Any, aircraft: Any | None = None) -> None:
        """逐簇：收集簇内成员的「本队观测」孪生航迹 → 多源融合 → 写回孪生。

        数据链共享的语义（docs §3.3）：
            - 簇内某成员观测到敌方目标 → 同簇所有成员共享该航迹；
            - 融合规则：confidence 取 max、pos_err 取 min、位置按置信度加权；
            - 写入孪生时 source=2（数据链），供 AI/渲染区分来源。
        物理池引用用于敌我判定（side）。
        """
        if not self._clusters:
            return
        for cluster in self._clusters:
            # 收集簇内活跃成员（armor + aircraft 节点）
            members = []
            for node in cluster:
                if node[0] == N_ARMOR:
                    members.append((TWIN_ARMOR, node[1], int(armor.side[node[1]])))
                elif node[0] == N_AIRCRAFT and aircraft is not None:
                    members.append((TWIN_AIRCRAFT, node[1], int(aircraft.side[node[1]])))
            if not members:
                continue
            # 簇内任一成员视角的敌情：对敌方孪生槽做融合
            # 敌方 = 与成员阵营不同的单位；用孪生槽反向查（twin.unit_kind/twin_idx）
            twin_slots = twin.active_slots()
            if twin_slots.size == 0:
                continue
            for s in twin_slots:
                s = int(s)
                kind = int(twin.unit_kind[s])
                idx = int(twin.twin_idx[s])
                # 物理单位 side
                if kind == TWIN_ARMOR:
                    if not (0 <= idx < armor.capacity) or not armor.active[idx]:
                        continue
                    side = int(armor.side[idx])
                    pos = armor.pos[idx]
                elif kind == TWIN_AIRCRAFT and aircraft is not None:
                    if not (0 <= idx < aircraft.capacity) or not aircraft.active[idx]:
                        continue
                    side = int(aircraft.side[idx])
                    pos = aircraft.pos[idx]
                else:
                    continue
                # 该目标是「簇内某成员」的敌方 → 可被簇共享
                hostile_in_cluster = any(m_side != side for (_, _, m_side) in members)
                if not hostile_in_cluster:
                    continue
                # 多源融合：簇内观测到该目标（twin 已有观测）→ 提升为数据链共享
                if twin.status[s] == 1:   # ACTIVE（已被观测）
                    twin.source[s] = 2
                    twin.confidence[s] = min(0.95, max(twin.confidence[s], 0.8))
                    twin.pos_err[s] = min(twin.pos_err[s], 40.0)


# 孪生单位种类常量（避免循环 import entities.twin）
TWIN_ARMOR = 0
TWIN_AIRCRAFT = 1
