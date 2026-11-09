"""Phase 8 步骤③：DataLinkNetwork（docs/phase8-datalink-design.md §十四 步骤③验收）。

覆盖：
    链路质量公式（comm 乘积/距离衰减/ECM）、建图 + 簇分裂/合并、
    中央簇（HQ 所在）、route_quality 路径连乘、同簇/中央判定、
    broadcast_contacts 数据链航迹共享（簇内敌情融合）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.entities.pools import ArmorPool
from Airia.lulilo.entities.twin import TWIN_KIND_ARMOR, TwinPool, TwinStatus
from Airia.lulilo.simulation.datalink import (
    N_AIRCRAFT,
    N_ARMOR,
    N_FORMATION,
    N_HQ,
    DataLinkNetwork,
    _node,
)


class TestLinkQuality:
    """链路质量公式（docs §3.1）。"""

    def test_comm_product(self) -> None:
        net = DataLinkNetwork()
        q = net.link_quality(0, 0, 0, 1.0, 0.0, 100.0, 0, 0, 1.0, 0.0)
        # 近距离 → dist_factor ≈ 1 → q ≈ 1
        assert q == pytest.approx(1.0, abs=0.05)

    def test_low_comm_breaks_link(self) -> None:
        net = DataLinkNetwork()
        q = net.link_quality(0, 0, 0, 0.5, 0.0, 100.0, 0, 0, 0.5, 0.0)
        # 0.5 × 0.5 × dist ≈ 0.25 < 0.5 阈值
        assert q < 0.5

    def test_ecm_suppression(self) -> None:
        net = DataLinkNetwork()
        q0 = net.link_quality(0, 0, 0, 1.0, 0.0, 100.0, 0, 0, 1.0, 0.0)
        q1 = net.link_quality(0, 0, 0, 1.0, 0.9, 100.0, 0, 0, 1.0, 0.0)
        assert q1 < q0, "ECM 压制应降低链路质量"

    def test_bvr_no_direct_link(self) -> None:
        net = DataLinkNetwork()
        q = net.link_quality(0, 0, 0, 1.0, 0.0, 60000.0, 0, 0, 1.0, 0.0)
        assert q == pytest.approx(0.0), "超视距不直接连边（走中继）"


class TestGraphAndClusters:
    """建图 + 簇（docs §3.2）。"""

    def _rig(self, n_units: int = 6, spacing: float = 100.0):
        armor = ArmorPool(capacity=16)
        idxs = []
        for k in range(n_units):
            i = armor.spawn(k * spacing, 0.0, 0.0, side=k % 2)
            armor.comm[i] = 1.0
            idxs.append(i)
        return armor, idxs

    def test_cluster_formation(self) -> None:
        armor, idxs = self._rig(spacing=100.0)   # 近距离 → 单簇
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        assert net.cluster_count == 1
        # HQ 在中央簇
        assert net.is_central(_node(N_HQ, 0))
        assert net.in_same_cluster(_node(N_ARMOR, idxs[0]), _node(N_ARMOR, idxs[1]))

    def test_split_clusters(self) -> None:
        armor = ArmorPool(capacity=16)
        a0 = armor.spawn(0.0, 0.0, 0.0, side=0)
        armor.comm[a0] = 1.0
        a1 = armor.spawn(100000.0, 0.0, 0.0, side=1)   # 超远 → 断链
        armor.comm[a1] = 1.0
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        assert net.cluster_count == 2, "超视距应分裂成两个簇"
        assert net.is_central(_node(N_ARMOR, a0))
        assert not net.is_central(_node(N_ARMOR, a1))

    def test_ecm_splits_cluster(self) -> None:
        armor = ArmorPool(capacity=8)
        a0 = armor.spawn(0.0, 0.0, 0.0, side=0)
        armor.comm[a0] = 1.0
        a1 = armor.spawn(500.0, 0.0, 0.0, side=1)
        armor.comm[a1] = 1.0
        armor.ecm[a1] = 0.99   # 强 ECM → 断链
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        assert net.cluster_count == 2, "ECM 压制应断链分裂"

    def test_route_quality_path_product(self) -> None:
        armor = ArmorPool(capacity=8)
        idxs = []
        for k in range(3):
            i = armor.spawn(k * 100.0, 0.0, 0.0, side=0)
            armor.comm[i] = 1.0
            idxs.append(i)
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        # a → c 路径连乘（经 b）
        q = net.route_quality(_node(N_ARMOR, idxs[0]), _node(N_ARMOR, idxs[2]))
        assert 0.0 < q < 1.0
        # 同节点 → 1
        assert net.route_quality(_node(N_ARMOR, idxs[0]), _node(N_ARMOR, idxs[0])) == pytest.approx(1.0)

    def test_route_quality_disconnected(self) -> None:
        armor = ArmorPool(capacity=8)
        a0 = armor.spawn(0.0, 0.0, 0.0, side=0)
        armor.comm[a0] = 1.0
        a1 = armor.spawn(100000.0, 0.0, 0.0, side=0)
        armor.comm[a1] = 1.0
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        assert net.route_quality(_node(N_ARMOR, a0), _node(N_ARMOR, a1)) == pytest.approx(0.0)


class TestBroadcastContacts:
    """航迹广播（docs §3.3）。"""

    def test_cluster_shares_tracks(self) -> None:
        armor = ArmorPool(capacity=8)
        a0 = armor.spawn(0.0, 0.0, 0.0, side=0)
        armor.comm[a0] = 1.0
        a1 = armor.spawn(100.0, 0.0, 0.0, side=0)   # 同簇友军
        armor.comm[a1] = 1.0
        foe = armor.spawn(500.0, 0.0, 0.0, side=1)  # 敌方
        armor.comm[foe] = 1.0
        twin = TwinPool(capacity=64)
        # 我方观测到敌方
        twin.bind(TWIN_KIND_ARMOR, foe, pos=(500.0, 0.0, 0.0))
        twin.status[twin.slot_of(TWIN_KIND_ARMOR, foe)] = int(TwinStatus.ACTIVE)
        twin.confidence[twin.slot_of(TWIN_KIND_ARMOR, foe)] = 0.6
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        net.broadcast_contacts(twin, armor)
        s = twin.slot_of(TWIN_KIND_ARMOR, foe)
        assert twin.source[s] == 2, "簇内共享航迹应标记 source=2（数据链）"
        assert twin.confidence[s] >= 0.8, "数据链共享应提升置信度"

    def test_no_share_across_clusters(self) -> None:
        armor = ArmorPool(capacity=8)
        a0 = armor.spawn(0.0, 0.0, 0.0, side=0)
        armor.comm[a0] = 1.0
        foe = armor.spawn(100000.0, 0.0, 0.0, side=1)   # 另一簇（超远，无法观测）
        armor.comm[foe] = 1.0
        twin = TwinPool(capacity=64)
        twin.bind(TWIN_KIND_ARMOR, foe)
        s = twin.slot_of(TWIN_KIND_ARMOR, foe)
        # 保持 UNKNOWN（簇内无观测）→ broadcast 不应凭空共享
        net = DataLinkNetwork()
        net.build(armor, hq_pos=(0.0, 0.0, 0.0))
        net.broadcast_contacts(twin, armor)
        assert twin.source[s] != 2, "无观测的敌情不应被数据链共享"
