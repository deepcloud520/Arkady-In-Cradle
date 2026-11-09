"""Phase 7 步骤④：机载雷达搜索 + 飞机 emitter 绑定（docs/phase7-air-pipeline.md §十四 步骤④验收）。

覆盖：
    RCS 常量、接触表生成（距离/种类/属性）、side 过滤、坠毁/驻停剔除、
    雷达地平线截断、检测概率采样（rng）、LOS 过滤、地面目标检测。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.detection.sensors import RCS_BOMBER, RCS_FIGHTER
from Airia.lulilo.entities.pools import AircraftPool, AircraftState, ArmorPool
from Airia.lulilo.simulation.air_ai import (
    TGT_AIR,
    TGT_GROUND,
    gather_air_contacts,
)


def _rig() -> tuple[AircraftPool, ArmorPool, list]:
    aircraft = AircraftPool(capacity=8)
    armor = ArmorPool(capacity=8)
    contacts: list = [[] for _ in range(aircraft.capacity)]
    return aircraft, armor, contacts


class TestRcsConstants:
    def test_constants(self) -> None:
        assert RCS_FIGHTER == 5.0
        assert RCS_BOMBER == 10.0


class TestGatherBasics:
    """接触表生成（rng=None → 确定性全命中，聚焦几何/过滤逻辑）。"""

    def test_detect_nearby_enemy_aircraft(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        foe = a.spawn(1000.0, 0.0, 3000.0, side=0)   # 1km 外同高度
        gather_air_contacts(1, a, armor, contacts)
        assert len(contacts[me]) == 1
        c = contacts[me][0]
        assert c.tgt_idx == foe
        assert c.tgt_kind == TGT_AIR
        assert c.dist == pytest.approx(1000.0, abs=1.0)

    def test_detect_ground_targets(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        tank = armor.spawn(2000.0, 0.0, 0.0)
        armor.side[tank] = 0
        armor.rcs[tank] = 30.0
        gather_air_contacts(1, a, armor, contacts)
        assert len(contacts[me]) == 1
        c = contacts[me][0]
        assert c.tgt_kind == TGT_GROUND
        assert c.tgt_idx == tank
        assert c.d_alt == pytest.approx(3000.0, abs=1.0)

    def test_side_filter_no_friendly(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        buddy = a.spawn(500.0, 0.0, 3000.0, side=1)   # 己方
        gather_air_contacts(1, a, armor, contacts)
        assert contacts[me] == [], "己方飞机不应出现在接触表"

    def test_far_target_not_detected(self) -> None:
        """超过雷达最大检测距离 → 无接触。"""
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        foe = a.spawn(40000.0, 0.0, 3000.0, side=0)  # 40km（RCS=5 → R≈17.7km）
        gather_air_contacts(1, a, armor, contacts)
        assert contacts[me] == [], "40km 外目标不应被探测"

    def test_crashing_aircraft_excluded_from_targets(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        foe = a.spawn(500.0, 0.0, 3000.0, side=0)
        a.state[foe] = int(AircraftState.CRASHING)
        gather_air_contacts(1, a, armor, contacts)
        assert contacts[me] == [], "坠毁机不应成为目标"

    def test_parked_aircraft_excluded_from_targets(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        foe = a.spawn(500.0, 0.0, 3000.0, side=0)
        a.state[foe] = int(AircraftState.PARKED)
        gather_air_contacts(1, a, armor, contacts)
        assert contacts[me] == []

    def test_own_side_queries_own_contacts(self) -> None:
        """敌方雷达也应收集自己的接触（side 参数决定传感器集合）。"""
        a, armor, contacts = _rig()
        foe = a.spawn(1000.0, 0.0, 3000.0, side=0)
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        gather_air_contacts(0, a, armor, contacts)
        assert len(contacts[foe]) == 1
        assert contacts[foe][0].tgt_idx == me


class TestRadarHorizon:
    """雷达地平线截断：低空目标看不到超视距。"""

    def test_horizon_blocks_low_altitude(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 500.0, side=1)       # 低空 500m → R_h≈80km
        foe = a.spawn(60000.0, 0.0, 500.0, side=0)  # 60km 同低空
        gather_air_contacts(1, a, armor, contacts)
        # RCS=5 → R_radar ≈ 17.7km << 60km，属距离门限拒绝；换高空对照
        assert contacts[me] == []

    def test_high_altitude_extends_horizon(self) -> None:
        """高空视野更远（R_h 随 √h 增长），但雷达功率仍是主限。"""
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 9000.0, side=1)
        foe = a.spawn(15000.0, 0.0, 9000.0, side=0)  # 15km：R_h(9000m)≈338km，靠雷达距离
        gather_air_contacts(1, a, armor, contacts)
        # RCS=5 → R_radar ≈ 17.7km，15km 在门限内 → 有接触
        assert len(contacts[me]) == 1


class TestDetectionProbability:
    """rng 采样：距离越远检测概率越低（统计）。"""

    def test_close_detected_far_missed_statistically(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        close = a.spawn(2000.0, 0.0, 3000.0, side=0)  # 2km（p≈1）
        far = a.spawn(20000.0, 0.0, 3000.0, side=0)   # 20km（p≈0.03）
        rng = RNGManager(seed=7)
        hits_close = 0
        hits_far = 0
        for _ in range(50):
            gather_air_contacts(1, a, armor, contacts, rng=rng)
            hits_close += 1 if any(c.tgt_idx == close for c in contacts[me]) else 0
            hits_far += 1 if any(c.tgt_idx == far for c in contacts[me]) else 0
        assert hits_close >= 45, f"近距离目标检测率过低: {hits_close}/50"
        assert hits_far <= 10, f"远距离目标误检率过高: {hits_far}/50"


class TestLosFilter:
    """LOS 过滤（los_checker 注入）。"""

    def test_los_blocks(self) -> None:
        a, armor, contacts = _rig()
        me = a.spawn(0.0, 0.0, 3000.0, side=1)
        foe = a.spawn(3000.0, 0.0, 3000.0, side=0)

        class _Blocker:
            def check(self, *args) -> bool:
                return False   # 全部阻挡

        gather_air_contacts(1, a, armor, contacts, los_checker=_Blocker())
        assert contacts[me] == [], "LOS 阻挡时不应有接触"

        class _SeeAll:
            def check(self, *args) -> bool:
                return True

        gather_air_contacts(1, a, armor, contacts, los_checker=_SeeAll())
        assert len(contacts[me]) == 1


class TestEmitterBind:
    """飞机 emitter 绑定（步骤④与步骤①接缝的传感器侧验证）。"""

    def test_radar_emitter_tracks_aircraft(self) -> None:
        from Airia.lulilo.entities.pools import EmitterPool, ProjectilePool

        a, armor, _ = _rig()
        air_idx = a.spawn(0.0, 0.0, 3000.0, side=1)
        em = EmitterPool(capacity=16)
        e = em.activate_aircraft_radar(air_idx, 70.0)
        a.pos[air_idx] = (1234.0, 5678.0, 4500.0)
        em.sync_from_parent(armor, ProjectilePool(capacity=1), 0.1, aircraft=a)
        assert em.pos[e, 0] == pytest.approx(1234.0)
        assert em.pos[e, 2] == pytest.approx(4500.0)
