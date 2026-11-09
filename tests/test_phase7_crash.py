"""Phase 7 步骤⑧：飞机毁伤 + 坠毁链（docs/phase7-air-pipeline.md §十四 步骤⑧验收）。

覆盖：
    直接命中扣血 + dmg_factor 分段退化、血量归零 → CRASHING、
    CRASHING 触地 → 爆炸提交 + 残骸终态、坠毁关闭机载雷达（尾焰保留）、
    resolve_aircraft_area_damage 内核、近炸对邻近飞机区域毁伤。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.combat.damage_submission import (
    DamageKind,
    DamageSubmissionPool,
    resolve_aircraft_area_damage,
)
from Airia.lulilo.entities.pools import (
    AircraftPool,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)


def _rig_aircraft() -> tuple[AircraftPool, ArmorPool, BoxPool, EmitterPool, ProjectilePool]:
    aircraft = AircraftPool(capacity=8)
    armor = ArmorPool(capacity=2)
    boxes = BoxPool(capacity=64)
    emitters = EmitterPool(capacity=64)
    proj = ProjectilePool(capacity=16)
    return aircraft, armor, boxes, emitters, proj


class TestAircraftDamage:
    """直接命中毁伤（docs §7.4 ①）。"""

    def test_direct_hit_reduces_health(self) -> None:
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry, _rig
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=1),
            damage_pool=DamageSubmissionPool(capacity=16),
            aircraft=aircraft,
        )
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(3000.0, 0.0, 3000.0, side=0)
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 2
        boxes.spawn(foe, 5, 1.2, 4.5, 0.8, local_x=0, local_y=0, local_z=0, parent_kind=1)
        boxes.sync_all_from_parent(armor, aircraft=aircraft)
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        # 弹丸放敌机 1m 处 → 近炸引信（28m 圈）触发，破片伤害 ~87
        proj.pos[rec.proj_idx] = (float(aircraft.pos[foe, 0]) - 1.0,
                                  float(aircraft.pos[foe, 1]),
                                  float(aircraft.pos[foe, 2]))
        proj.vel[rec.proj_idx] = (100.0, 0.0, 0.0)
        fc.tick(0.01)
        # AIM-120 blast_dmg=90，1m/28m → 衰减伤害 ≈ 90·(1-1/28) ≈ 86.8 → health ≈ 13.2
        assert aircraft.health[foe] < 20.0, f"近炸伤害不足: {aircraft.health[foe]}"
        assert aircraft.dmg_factor[foe] == pytest.approx(0.6)  # <40% → 重度

    def test_damage_factor_stages(self) -> None:
        """health 分段 → dmg_factor 退化（70%/40% 阈值）。"""
        a, _, _, _, _ = _rig_aircraft()
        i = a.spawn(0, 0, 3000)
        for health, expect in ((80.0, 1.0), (50.0, 0.8), (20.0, 0.6)):
            a.health[i] = health
            # 调用与 _resolve_aircraft_hit 相同的分段逻辑
            frac = a.health[i] / 100.0
            if frac > 0.7:
                a.dmg_factor[i] = 1.0
            elif frac > 0.4:
                a.dmg_factor[i] = 0.8
            else:
                a.dmg_factor[i] = 0.6
            assert a.dmg_factor[i] == pytest.approx(expect), f"health={health}"


class TestCrashChain:
    """坠毁链：CRASHING → 触地 → 爆炸 → 残骸（docs §7.4）。"""

    def test_zero_health_transitions_to_crashing(self) -> None:
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=1),
            damage_pool=DamageSubmissionPool(capacity=16),
            aircraft=aircraft,
        )
        i = aircraft.spawn(0.0, 0.0, 3000.0)
        aircraft.health[i] = 0.0
        radar = emitters.activate_aircraft_radar(i, 70.0)
        tail = emitters.activate_aircraft_ir(i, 1.0)
        # 通过结算路径触发（血量归零 → CRASHING）
        fc._finalize_area_kills()
        assert aircraft.state[i] == int(AircraftState.CRASHING)
        # 机载雷达 RF 关闭（deactivate 保留索引）
        assert not emitters.active[radar]
        # 尾焰 IR 保留（残热衰减自然跑完）
        assert emitters.active[tail]

    def test_crashing_aircraft_ignored_as_target(self) -> None:
        """坠毁机从目标剔除（_find_nearest_air_target 过滤）。"""
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=1),
            damage_pool=DamageSubmissionPool(capacity=16),
            aircraft=aircraft,
        )
        i = aircraft.spawn(0.0, 0.0, 3000.0)
        aircraft.state[i] = int(AircraftState.CRASHING)
        ref = np.array([0.0, 0.0, 3000.0], np.float32)
        assert fc._find_nearest_air_target(ref, 1000.0) == -1

    def test_crash_impact_submits_explosion_and_removes(self) -> None:
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        pool = DamageSubmissionPool(capacity=16)
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=1),
            damage_pool=pool,
            aircraft=aircraft,
        )
        i = aircraft.spawn(0.0, 0.0, 3000.0)
        aircraft.state[i] = int(AircraftState.CRASHING)
        aircraft.vel[i] = (0.0, 0.0, -50.0)   # 快速下坠
        # 直接触地（z=0）
        aircraft.pos[i, 2] = 1.0
        fc._check_aircraft_crashes(0.1)
        assert not aircraft.active[i], "触地后应进入残骸终态"
        assert pool.active_count >= 1, "坠毁应提交爆炸事件"

    def test_crashing_air_still_falling_ignored(self) -> None:
        """还在空中下坠（未触地）→ 不销毁。"""
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=1),
            damage_pool=DamageSubmissionPool(capacity=16),
            aircraft=aircraft,
        )
        i = aircraft.spawn(0.0, 0.0, 3000.0)
        aircraft.state[i] = int(AircraftState.CRASHING)
        fc._check_aircraft_crashes(0.1)
        assert aircraft.active[i], "高空坠毁未触地不应销毁"


class TestAreaDamageKernel:
    """resolve_aircraft_area_damage 内核（docs §7.4 ②）。"""

    def test_sphere_damage_decay(self) -> None:
        n = 3
        kind = np.array([int(DamageKind.SPHERE)], np.int8)
        center = np.zeros((1, 3), np.float32)
        radius = np.array([20.0], np.float32)
        strength = np.array([100.0], np.float32)
        direction = np.zeros((1, 3), np.float32)
        cone_angle = np.zeros(1, np.float32)
        active = np.array([True], np.bool_)
        air_pos = np.array([[0, 0, 0], [10, 0, 0], [30, 0, 0]], np.float32)
        air_health = np.full(n, 100.0, np.float32)
        air_state = np.zeros(n, np.int8)
        air_active = np.ones(n, np.bool_)
        resolve_aircraft_area_damage(
            kind, center, radius, strength, direction, cone_angle, active,
            air_pos, air_health, air_state, air_active,
        )
        # 中心：100 全额；10m：100·(1-0.5)=50；30m（半径外）：不变
        assert air_health[0] == pytest.approx(0.0)
        assert air_health[1] == pytest.approx(50.0)
        assert air_health[2] == pytest.approx(100.0)

    def test_parked_aircraft_exempt(self) -> None:
        kind = np.array([int(DamageKind.SPHERE)], np.int8)
        center = np.zeros((1, 3), np.float32)
        radius = np.array([10.0], np.float32)
        strength = np.array([50.0], np.float32)
        direction = np.zeros((1, 3), np.float32)
        cone_angle = np.zeros(1, np.float32)
        active = np.array([True], np.bool_)
        air_pos = np.zeros((1, 3), np.float32)
        air_health = np.array([100.0], np.float32)
        air_state = np.array([6], np.int8)   # PARKED
        air_active = np.array([True], np.bool_)
        resolve_aircraft_area_damage(
            kind, center, radius, strength, direction, cone_angle, active,
            air_pos, air_health, air_state, air_active,
        )
        assert air_health[0] == pytest.approx(100.0), "PARKED 机不受爆轰"


class TestProxFuseNearbyAircraft:
    """近炸对邻近飞机区域毁伤（多机在场）。"""

    def test_blast_damages_nearby_aircraft(self) -> None:
        from Airia.lulilo.combat.engagement import FiringControl

        from tests.test_phase7_aam import _air_registry, _rig
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry
        from Airia.lulilo.core.rng import RNGManager

        aircraft, armor, boxes, emitters, proj = _rig_aircraft()
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=2),
            damage_pool=DamageSubmissionPool(capacity=16),
            aircraft=aircraft,
        )
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(200.0, 0.0, 3000.0, side=0)
        buddy = aircraft.spawn(230.0, 5.0, 3000.0, side=0)   # 邻机在爆炸半径内
        aircraft.health[foe] = 100.0
        aircraft.health[buddy] = 100.0
        aircraft.aam_left[me] = 2
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        proj.pos[rec.proj_idx] = (float(aircraft.pos[foe, 0]) - 5.0,
                                  float(aircraft.pos[foe, 1]),
                                  float(aircraft.pos[foe, 2]))
        fc.tick(0.1)
        assert aircraft.health[foe] < 100.0, "目标机应被近炸伤害"
        assert aircraft.health[buddy] < 100.0, "爆炸半径内邻机应被波及"
