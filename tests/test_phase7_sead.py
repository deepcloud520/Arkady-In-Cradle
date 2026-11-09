"""Phase 7 步骤⑩：地面防空激活（role=3）+ RWR/诱饵泛化（docs/phase7-air-pipeline.md §十四 步骤⑩验收）。

覆盖：
    SAM 防空打飞机（发射 + 制导 + 命中）、防空只打敌机/只打空中、
    RWR 泛化到飞机（平台雷达→搜索级、主动导引头→锁定级）、
    热焰诱饵诱骗红外导引头（_find_ir_hot 选最热源）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
from Airia.lulilo.combat.engagement import FiringControl
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    GuidanceType,
    ProjectileType,
    SeekerType,
    WeaponConfig,
    WeaponRegistry,
    WeaponSystem,
)
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    AircraftPool,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    ProjectilePool,
)
from Airia.lulilo.simulation.ai import RWRDetector
from Airia.lulilo.simulation.air_ai import AirDefenseController

from tests.test_phase7_aam import _air_registry


def _sam_registry() -> WeaponRegistry:
    reg = _air_registry()
    reg.register_definition(
        WeaponConfig(
            weapon_id="sam",
            name="SA-11 防空导弹",
            caliber_mm=400.0,
            muzzle_velocity=700.0,
            base_accuracy_mrad=1.0,
            fire_rate_rpm=1.0,
            max_range_m=12000.0,
            fire_control_stabilization=0.7,
            compatible_ammo=["sam"],
        ),
        [
            AmmoConfig(
                ammo_id="sam",
                name="SA-11 SAM",
                ammo_type=AmmoType.ATGM,
                pen_mm_rha=80.0,
                muzzle_velocity=700.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="dual_pulse_pn",
                seeker=SeekerType.ACTIVE_RADAR,
                emitter_power_db=75.0,
                nav_gain=4.0,
                search_fov_deg=60.0,
                track_fov_deg=12.0,
                filter_quality=0.8,
                prox_fuse_radius=25.0,
                blast_radius=15.0,
                blast_dmg=80.0,
            ),
        ],
    )
    return reg


def _rig(air_cap: int = 8, armor_cap: int = 8):
    aircraft = AircraftPool(capacity=air_cap)
    armor = ArmorPool(capacity=armor_cap)
    boxes = BoxPool(capacity=64)
    proj = ProjectilePool(capacity=32)
    emitters = EmitterPool(capacity=64)
    weapons = WeaponSystem(_sam_registry())
    control = MissileControlRegistry.with_builtin()
    firing = FiringControl(
        armor, boxes, proj,
        weapon_system=weapons,
        control=control,
        emitters=emitters,
        rng=RNGManager(seed=42),
        damage_pool=DamageSubmissionPool(capacity=64),
        aircraft=aircraft,
    )
    return aircraft, armor, firing, proj, emitters


class TestAirDefense:
    """SAM 防空激活（docs §十）。"""

    def test_sam_launch_and_hit_aircraft(self) -> None:
        from Airia.lulilo.combat.engagement import UnitCombatConfig
        from Airia.lulilo.entities.pools import PartType

        aircraft, armor, firing, proj, emitters = _rig()
        # 防空单位配置 + 发射架（TURRET 箱：can_fire 语义）
        aa = armor.spawn(0.0, 0.0, 0.0, side=1, role=3)
        firing.boxes.spawn(aa, int(PartType.TURRET), 0.6, 0.6, 0.6,
                           local_x=0.0, local_y=0.0, local_z=1.0)
        firing.boxes.sync_all_from_parent(armor)
        firing.set_unit_config(aa, UnitCombatConfig(
            weapon_id="sam", ammo_id="sam",
            crew_skill=1.0, visibility=1.0, weapon_condition=1.0,
            max_range_m=12000.0, fire_rate_rpm=1.0,
        ))
        foe = aircraft.spawn(2500.0, 0.0, 1200.0, side=0)
        aircraft.vel[foe] = (0.0, 0.0, 0.0)   # 悬停目标（防空典型：直升机/慢速机）
        aircraft.health[foe] = 100.0
        adc = AirDefenseController(armor, aircraft, firing, sam_ammo_id="sam")
        adc.tick(0.1)
        # SAM 已发射且制导目标 = 飞机
        assert proj.active_count == 1, "防空单位应发射 SAM"
        j = int(proj.active_indices()[0])
        assert proj.guid_target_kind[j] == 1
        # 推进 → SAM 命中（近炸）
        hit = False
        for _ in range(2000):
            aircraft.pos[foe] += aircraft.vel[foe] * 0.01
            firing.tick(0.01)
            if aircraft.health[foe] < 100.0 or not aircraft.active[foe]:
                hit = True
                break
            if not proj.active[j]:
                break
        assert hit, f"SAM 未命中飞机（health={aircraft.health[foe]}）"

    def test_sam_no_friendly_fire(self) -> None:
        from Airia.lulilo.combat.engagement import UnitCombatConfig
        from Airia.lulilo.entities.pools import PartType

        aircraft, armor, firing, proj, emitters = _rig()
        aa = armor.spawn(0.0, 0.0, 0.0, side=1, role=3)
        firing.boxes.spawn(aa, int(PartType.TURRET), 0.6, 0.6, 0.6,
                           local_x=0.0, local_y=0.0, local_z=1.0)
        firing.boxes.sync_all_from_parent(armor)
        firing.set_unit_config(aa, UnitCombatConfig(
            weapon_id="sam", ammo_id="sam",
            crew_skill=1.0, visibility=1.0, weapon_condition=1.0,
            max_range_m=12000.0, fire_rate_rpm=1.0,
        ))
        buddy = aircraft.spawn(5000.0, 0.0, 3000.0, side=1)   # 己方
        adc = AirDefenseController(armor, aircraft, firing, sam_ammo_id="sam")
        adc.tick(0.1)
        assert proj.active_count == 0, "不应打己方飞机"

    def test_sam_out_of_range(self) -> None:
        from Airia.lulilo.combat.engagement import UnitCombatConfig
        from Airia.lulilo.entities.pools import PartType

        aircraft, armor, firing, proj, emitters = _rig()
        aa = armor.spawn(0.0, 0.0, 0.0, side=1, role=3)
        firing.boxes.spawn(aa, int(PartType.TURRET), 0.6, 0.6, 0.6,
                           local_x=0.0, local_y=0.0, local_z=1.0)
        firing.boxes.sync_all_from_parent(armor)
        firing.set_unit_config(aa, UnitCombatConfig(
            weapon_id="sam", ammo_id="sam",
            crew_skill=1.0, visibility=1.0, weapon_condition=1.0,
            max_range_m=12000.0, fire_rate_rpm=1.0,
        ))
        foe = aircraft.spawn(20000.0, 0.0, 3000.0, side=0)   # 超射程
        adc = AirDefenseController(armor, aircraft, firing, sam_ammo_id="sam",
                                   max_range_m=12000.0)
        adc.tick(0.1)
        assert proj.active_count == 0, "超射程不应发射"


class TestRwrAircraft:
    """RWR 泛化到飞机（docs §十一.1）。"""

    def _rig_rwr(self):
        aircraft = AircraftPool(capacity=8)
        armor = ArmorPool(capacity=4)
        emitters = EmitterPool(capacity=64)
        proj = ProjectilePool(capacity=16)
        rwr = RWRDetector(armor, emitters, proj, aircraft=aircraft)
        return aircraft, armor, emitters, proj, rwr

    def test_radar_warning_level1(self) -> None:
        aircraft, armor, emitters, proj, rwr = self._rig_rwr()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        emitters.activate_aircraft_radar(foe, 70.0)   # 敌方机载雷达
        rwr.tick(0.1)
        assert aircraft.rwr_state[me] == 1, "敌方雷达开机 → 搜索级告警"

    def test_seeker_warning_level3(self) -> None:
        aircraft, armor, emitters, proj, rwr = self._rig_rwr()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(5000.0, 0.0, 3000.0, side=0)
        # 敌方 AAM 主动导引头（父=弹丸，owner=敌机）
        pj = proj.spawn(5000.0, 0.0, 3000.0, 0, 0, 0, owner=foe)
        emitters.activate_from_projectile(pj, 70.0, EmitterBand.RF, 1)   # SEEKER
        rwr.tick(0.1)
        assert aircraft.rwr_state[me] == 3, "主动导引头 → 锁定级告警"

    def test_no_warning_when_silent(self) -> None:
        aircraft, armor, emitters, proj, rwr = self._rig_rwr()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        rwr.tick(0.1)
        assert aircraft.rwr_state[me] == 0


class TestFlareDecoy:
    """热焰诱饵诱骗红外导引头（docs §11.3：_find_ir_hot 选最热源）。"""

    def test_find_ir_hot_prefers_flare(self) -> None:
        from Airia.lulilo.combat.engagement import FiringControl

        from Airia.lulilo.combat.weapons import WeaponSystem

        aircraft, armor, firing, proj, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(2000.0, 0.0, 3000.0, side=0)
        # 敌机尾焰（cur_power=1.0）
        tail = emitters.activate_aircraft_ir(foe, 1.0)
        emitters.sync_from_parent(armor, proj, 0.1, aircraft=aircraft)
        # 我方投放高功率热焰弹（cur_power=500）
        flare = emitters.deploy_decoy(0.0, 0.0, 3000.0, band=int(EmitterBand.IR),
                                      power=500.0)
        hot = firing._find_ir_hot(0.0, 0.0, 3000.0, 5000.0)
        assert hot == flare, "红外导引头应锁最热源（热焰诱饵）"
        assert hot != tail

    def test_flare_outshines_engine(self) -> None:
        """flare cur_power 瞬时远高于尾焰 → 诱骗成功（导引头目标=flare）。"""
        from Airia.lulilo.combat.engagement import FiringControl

        aircraft, armor, firing, proj, emitters = _rig()
        foe = aircraft.spawn(2000.0, 0.0, 3000.0, side=0)
        emitters.activate_aircraft_ir(foe, 1.0)
        emitters.sync_from_parent(armor, proj, 0.1, aircraft=aircraft)
        flare = emitters.deploy_decoy(0.0, 0.0, 3000.0, band=int(EmitterBand.IR),
                                      power=1000.0)
        hot = firing._find_ir_hot(0.0, 0.0, 3000.0, 5000.0)
        assert hot == flare
        assert emitters.cur_power[flare] > emitters.cur_power[
            [i for i in emitters.active_indices() if emitters.kind[i] == 3][0]
        ]
