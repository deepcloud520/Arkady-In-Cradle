"""Phase 8 步骤⑦：巡飞弹侦察上报（docs/phase8-datalink-design.md §十四 步骤⑦验收）。

覆盖：
    ProjectileType.LOITER、LOITER 弹丸扫描附近敌方 → 上报孪生（source=1 本队观测）、
    只上报敌方（不报己方）、弹丸销毁 → 上报中断 → 孪生衰减回 UNKNOWN。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.entities.pools import (
    ProjectilePool,
    ProjectileType,
    ArmorPool,
)
from Airis.lulilo.entities.twin import (
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)
from Airis.lulilo.simulation.air_ai import LoiterRecon
from Airis.lulilo.simulation.twin_engine import TwinEngine


class _Loop:
    def __init__(self) -> None:
        self.state = type("S", (), {"tick_count": 0})()


class TestLoiterRecon:
    """巡飞弹侦察上报（docs §5.4）。"""

    def _rig(self):
        twin = TwinPool(capacity=64)
        armor = ArmorPool(capacity=8)
        proj = ProjectilePool(capacity=16)
        engine = TwinEngine(twin, armor=armor)
        loop = _Loop()
        recon = LoiterRecon(engine, armor, proj, side=0, scan_radius=300.0)
        return twin, armor, proj, engine, loop, recon

    def test_loiter_reports_enemy_to_twin(self) -> None:
        twin, armor, proj, engine, loop, recon = self._rig()
        foe = armor.spawn(100.0, 50.0, 0.0, side=1)
        twin.bind(TWIN_KIND_ARMOR, foe, pos=(0.0, 0.0, 0.0))
        # LOITER 弹丸悬停在目标附近
        pj = proj.spawn(120.0, 60.0, 50.0, 0.0, 0.0, 0.0,
                        kind=int(ProjectileType.LOITER))
        recon.tick(1.0)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, foe)
        assert twin.status[s] == int(TwinStatus.ACTIVE), "巡飞弹应上报敌方"
        assert twin.source[s] == 1, "上报源 = 本队观测"
        assert twin.confidence[s] == pytest.approx(0.85)
        assert twin.pos[s, 0] == pytest.approx(100.0), "位置 = 巡飞弹观测真值"

    def test_only_enemy_reported(self) -> None:
        twin, armor, proj, engine, loop, recon = self._rig()
        buddy = armor.spawn(100.0, 50.0, 0.0, side=0)   # 己方
        twin.bind(TWIN_KIND_ARMOR, buddy, pos=(0.0, 0.0, 0.0))
        pj = proj.spawn(120.0, 60.0, 50.0, 0.0, 0.0, 0.0,
                        kind=int(ProjectileType.LOITER))
        recon.tick(1.0)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, buddy)
        assert twin.status[s] == int(TwinStatus.UNKNOWN), "巡飞弹不报己方"

    def test_out_of_scan_range_not_reported(self) -> None:
        twin, armor, proj, engine, loop, recon = self._rig()
        foe = armor.spawn(5000.0, 0.0, 0.0, side=1)   # 超出扫描半径
        twin.bind(TWIN_KIND_ARMOR, foe, pos=(0.0, 0.0, 0.0))
        pj = proj.spawn(120.0, 60.0, 50.0, 0.0, 0.0, 0.0,
                        kind=int(ProjectileType.LOITER))
        recon.tick(1.0)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, foe)
        assert twin.status[s] == int(TwinStatus.UNKNOWN)

    def test_destroyed_loiter_stops_reporting(self) -> None:
        """弹丸被击落 → 上报中断 → 孪生衰减回 UNKNOWN（docs §5.4）。"""
        twin, armor, proj, engine, loop, recon = self._rig()
        foe = armor.spawn(100.0, 50.0, 0.0, side=1)
        twin.bind(TWIN_KIND_ARMOR, foe, pos=(0.0, 0.0, 0.0))
        pj = proj.spawn(120.0, 60.0, 50.0, 0.0, 0.0, 0.0,
                        kind=int(ProjectileType.LOITER))
        # 持续上报
        for _ in range(3):
            loop.state.tick_count += 1
            recon.tick(1.0)
            engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, foe)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        # 巡飞弹被击落
        proj.destroy(pj)
        # 无上报 → 衰减 → UNKNOWN
        for _ in range(120):
            loop.state.tick_count += 1
            recon.tick(1.0)
            engine.tick(loop, 1.0)
            if twin.status[s] == int(TwinStatus.UNKNOWN):
                break
        assert twin.status[s] == int(TwinStatus.UNKNOWN), "巡飞弹坠毁 → 情报衰减回 UNKNOWN"

    def test_projectile_type_loiter(self) -> None:
        assert int(ProjectileType.LOITER) == 2   # 追加不重排
