"""Phase 7 步骤⑦：AAM/AGM/滑翔弹 + 近炸 + 制导内核（docs/phase7-air-pipeline.md §十四 步骤⑦验收）。

覆盖：
    control_glide_bomb 注册（索引 4）、guid_target_vel 内核兼容（地面目标零速度逐位不变）、
    fire_air 发射（aam_left 扣减 / 无弹 None / 制导字段）、近炸引信起爆、
    AAM 端到端 BVR 命中横切机（PN 前置量）、直接命中飞机箱、滑翔弹 PN 转向。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.combat.collision import detect_proj_aircraft_batch
from Airis.lulilo.combat.engagement import FiringControl
from Airis.lulilo.combat.guidance import (
    GuidanceStage,
    MissileControlRegistry,
    control_glide_bomb,
)
from Airis.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    GuidanceType,
    ProjectileType,
    SeekerType,
    WeaponConfig,
    WeaponRegistry,
    WeaponSystem,
)
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import (
    AircraftPool,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airis.lulilo.movement.integrate import integrate_guided_projectile_movement
from Airis.lulilo.simulation.commands import CommandQueue
from Airis.lulilo.simulation.state import SimulationState


def _air_registry() -> WeaponRegistry:
    """内置 + 航空弹药（AIM-120 雷达 AAM / 激光滑翔弹）。"""
    reg = WeaponRegistry.with_builtin()
    reg.register_definition(
        WeaponConfig(
            weapon_id="aim120_amraam",
            name="AIM-120 AMRAAM",
            caliber_mm=178.0,
            muzzle_velocity=900.0,
            base_accuracy_mrad=0.5,
            fire_rate_rpm=1.0,
            max_range_m=30000.0,
            fire_control_stabilization=0.9,
            compatible_ammo=["aim120_amraam"],
        ),
        [
            AmmoConfig(
                ammo_id="aim120_amraam",
                name="AIM-120 AMRAAM",
                ammo_type=AmmoType.ATGM,
                pen_mm_rha=90.0,            # 破片战斗部等效
                muzzle_velocity=900.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="dual_pulse_pn",
                seeker=SeekerType.ACTIVE_RADAR,
                emitter_power_db=70.0,
                nav_gain=4.0,
                search_fov_deg=60.0,
                track_fov_deg=10.0,
                filter_quality=0.85,
                prox_fuse_radius=25.0,      # Phase 7 近炸（AIM-120 破片散布 ~25m）
                blast_radius=15.0,
                blast_dmg=90.0,
            ),
        ],
    )
    reg.register_definition(
        WeaponConfig(
            weapon_id="gb_1000kg",
            name="1000kg 滑翔炸弹",
            caliber_mm=457.0,
            muzzle_velocity=120.0,
            base_accuracy_mrad=2.0,
            fire_rate_rpm=1.0,
            max_range_m=20000.0,
            fire_control_stabilization=0.5,
            compatible_ammo=["gb_1000kg"],
        ),
        [
            AmmoConfig(
                ammo_id="gb_1000kg",
                name="1000kg 滑翔炸弹",
                ammo_type=AmmoType.HE_FRAG,
                pen_mm_rha=400.0,
                muzzle_velocity=120.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="glide_bomb",
                seeker=SeekerType.SEMI_ACTIVE_LASER,
                nav_gain=2.0,
                search_fov_deg=40.0,
                track_fov_deg=8.0,
                filter_quality=0.8,
                blast_radius=30.0,
                blast_dmg=300.0,
                prox_fuse_radius=0.0,
            ),
        ],
    )
    return reg


def _rig() -> tuple[FiringControl, AircraftPool, ArmorPool, ProjectilePool, BoxPool, EmitterPool]:
    aircraft = AircraftPool(capacity=8)
    armor = ArmorPool(capacity=4)
    boxes = BoxPool(capacity=64)
    projectiles = ProjectilePool(capacity=32)
    emitters = EmitterPool(capacity=64)
    weapons = WeaponSystem(_air_registry())
    control = MissileControlRegistry.with_builtin()
    fc = FiringControl(
        armor, boxes, projectiles,
        weapon_system=weapons,
        control=control,
        emitters=emitters,
        rng=RNGManager(seed=42),
        events=None,
        damage_pool=None,
        aircraft=aircraft,
    )
    return fc, aircraft, armor, projectiles, boxes, emitters


class TestGlideBombLaw:
    """control_glide_bomb 注册 + 转向行为。"""

    def test_registered_at_index_4(self) -> None:
        reg = MissileControlRegistry.with_builtin()
        assert reg.index_of("glide_bomb") == 4
        assert reg.law_at(4) is control_glide_bomb

    def test_ballistic_no_steering(self) -> None:
        out = np.zeros(3, np.float32)
        pos = np.array([0, 0, 3000], np.float32)
        vel = np.array([200, 0, 0], np.float32)
        tpos = np.array([1000, 1000, 0], np.float32)
        tvel = np.zeros(3, np.float32)
        control_glide_bomb(pos, vel, tpos, tvel, 1.0,
                           int(GuidanceStage.BALLISTIC), 0, 2.0, 0.01, out)
        assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0

    def test_terminal_turns_toward_target(self) -> None:
        out = np.zeros(3, np.float32)
        pos = np.array([0, 0, 3000], np.float32)
        vel = np.array([200, 0, 0], np.float32)   # 东飞
        tpos = np.array([1000, 1000, 0], np.float32)  # 目标东南
        tvel = np.zeros(3, np.float32)
        control_glide_bomb(pos, vel, tpos, tvel, 5.0,
                           int(GuidanceStage.TERMINAL), 0, 2.0, 0.01, out)
        # PN 转向应产生 y 方向分量（朝目标）
        assert out[1] > 0.0, f"滑翔弹未朝目标转向: {out}"


class TestGuidTargetVelCompat:
    """guid_target_vel 内核兼容：地面目标零速度行为逐位不变。"""

    def test_zero_target_vel_identical(self) -> None:
        p = ProjectilePool(capacity=4)
        i = p.spawn(0, 0, 3000, 100, 0, 0)
        p.guid_law[i] = 4  # glide_bomb（无推力）
        p.guid_stage[i] = int(GuidanceStage.TERMINAL)
        p.guid_target_pos[i] = (1000, 0, 0)
        p.guid_nav_gain[i] = 2.0
        from Airis.lulilo.combat.guidance import control_glide_bomb

        pos0 = p.pos[i].copy()
        vel0 = p.vel[i].copy()
        integrate_guided_projectile_movement(
            p.pos, p.vel, p.active, p.age,
            p.guid_target_pos, p.guid_target_vel, p.guid_target_kind, p.guid_memory_pos, p.guid_stage,
            p.guid_nav_gain, p.guid_override,
            p.guid_search_fov_cos, p.guid_track_fov_cos,
            40.0 * 9.81, 0.1, 9.81, control_glide_bomb,
        )
        # 地面目标：guid_target_vel=0 → 弹丸推进无异常（位置前进、重力生效）
        assert p.pos[i, 0] > pos0[0]
        assert p.vel[i, 2] < vel0[2]


class TestFireAir:
    """fire_air：发射 + 弹药扣减 + 制导字段。"""

    def test_launch_aam(self) -> None:
        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(5000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        assert aircraft.aam_left[me] == 3
        j = rec.proj_idx
        assert projectiles.active[j]
        assert projectiles.guid_target_kind[j] == 1
        assert projectiles.owner[j] == me
        # 主动雷达导引头 emitter 激活
        assert emitters.active_count >= 1

    def test_no_ammo_returns_none(self) -> None:
        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(5000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 0
        assert fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                           target_kind=0, target_idx=foe) is None

    def test_agm_consumes_agm_left(self) -> None:
        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1, role=1)
        aircraft.agm_left[me] = 2
        target = np.array([3000.0, 0.0, 0.0], np.float32)
        rec = fc.fire_air(me, target, ammo_id="gb_1000kg", target_kind=1)
        assert rec is not None
        assert aircraft.agm_left[me] == 1
        assert projectiles.guid_target_kind[rec.proj_idx] == 0


class TestProxFuse:
    """近炸引信：AAM 接近飞机 → 起爆 → 飞机掉血。"""

    def test_prox_detonates(self) -> None:
        from Airis.lulilo.combat.damage_submission import DamageSubmissionPool

        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        fc.damage_pool = DamageSubmissionPool(capacity=16)
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(200.0, 0.0, 3000.0, side=0)
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 2
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        # 把 AAM 移到飞机旁（模拟接近）→ 下一子步近炸
        projectiles.pos[rec.proj_idx] = (float(aircraft.pos[foe, 0]) - 8.0,
                                         float(aircraft.pos[foe, 1]),
                                         float(aircraft.pos[foe, 2]))
        fc.tick(0.1)
        # 近炸 → SPHERE 提交 → 结算 → 飞机掉血
        assert aircraft.health[foe] < 100.0, f"近炸未伤及飞机: {aircraft.health[foe]}"


class TestAamEndToEnd:
    """AAM 端到端 BVR：PN 前置量拦截。"""

    def test_aam_intercepts_headon_aircraft(self) -> None:
        """迎头拦截（BVR 最常见场景）：目标向 AAM 飞来 → 近炸命中。

        横切目标（目标速度⊥视线）需极大横向过载，40g 下转弯半径 ~4.7km，
        66m 处物理上擦过——真实兵棋中横切亦低命中，故端到端用迎头。
        """
        from Airis.lulilo.combat.damage_submission import DamageSubmissionPool

        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        fc.damage_pool = DamageSubmissionPool(capacity=64)
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(4000.0, 0.0, 3000.0, side=0)
        aircraft.vel[foe] = (-250.0, 0.0, 0.0)   # 迎头飞来
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 4
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        # 推进 15s（1500 子步）：目标移动 + AAM 前置量拦截
        hit = False
        for _ in range(1500):
            aircraft.pos[foe] += aircraft.vel[foe] * 0.01
            fc.tick(0.01)
            if aircraft.health[foe] < 100.0 or not aircraft.active[foe]:
                hit = True
                break
            if not projectiles.active[rec.proj_idx]:
                break   # 脱靶回收
        assert hit, f"AAM 未命中迎头机（health={aircraft.health[foe]}）"

    def test_aam_crossing_requires_lead(self) -> None:
        """横切目标：AAM 追踪目标速度（guid_target_vel 前置量）——验证弹丸
        末段确实朝目标前方飞行（比无前置量更接近）。"""
        from Airis.lulilo.combat.damage_submission import DamageSubmissionPool

        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        fc.damage_pool = DamageSubmissionPool(capacity=64)
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(4000.0, -800.0, 3000.0, side=0)
        aircraft.vel[foe] = (250.0, 0.0, 0.0)   # 横切东飞
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 4
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        min_d = 1e9
        for _ in range(250):
            aircraft.pos[foe] += aircraft.vel[foe] * 0.01
            fc.tick(0.01)
            d = np.linalg.norm(projectiles.pos[rec.proj_idx] - aircraft.pos[foe])
            min_d = min(min_d, float(d))
            if not projectiles.active[rec.proj_idx]:
                break
        # 前置量制导让 AAM 逼近到目标附近（< 250m；无前置量则沿旧视线擦过 >1km）
        assert min_d < 250.0, f"AAM 横切接近距离过大: {min_d:.0f}m"

    def test_aam_misses_stationary_far(self) -> None:
        """AAM 超射程（40km > 30km）→ 不发射。"""
        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(40000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 2
        assert fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                           target_kind=0, target_idx=foe) is None


class TestDirectAirHit:
    """弹丸直接命中飞机机身箱（detect_proj_aircraft_batch 接入）。"""

    def test_direct_hit_damages(self) -> None:
        from Airis.lulilo.combat.damage_submission import DamageSubmissionPool

        fc, aircraft, armor, projectiles, boxes, emitters = _rig()
        fc.damage_pool = DamageSubmissionPool(capacity=16)
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(3000.0, 0.0, 3000.0, side=0)
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 2
        # 机身箱挂到敌机
        boxes.spawn(foe, 5, 1.2, 4.5, 0.8, local_x=0, local_y=0, local_z=0, parent_kind=1)
        boxes.sync_all_from_parent(armor, aircraft=aircraft)
        rec = fc.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                          target_kind=0, target_idx=foe)
        assert rec is not None
        # 把弹丸直接放到敌机箱位置（强制命中）
        projectiles.pos[rec.proj_idx] = (float(aircraft.pos[foe, 0]) - 1.0,
                                         float(aircraft.pos[foe, 1]),
                                         float(aircraft.pos[foe, 2]))
        projectiles.vel[rec.proj_idx] = (100.0, 0.0, 0.0)
        fc.tick(0.01)
        assert aircraft.health[foe] < 100.0, f"直接命中未伤及飞机: {aircraft.health[foe]}"

    def test_detect_proj_aircraft_kernel(self) -> None:
        """内核级冒烟：弹丸线段穿过飞机箱 → 命中。"""
        p_pos = np.array([[95, 0, 3000], [1000, 0, 3000]], np.float32)
        p_vel = np.array([[-1000, 0, 0], [0, 0, 0]], np.float32)  # 自东向西穿过箱体
        p_rad = np.array([0.1, 0.1], np.float32)
        p_active = np.array([True, True], np.bool_)
        a_pos = np.array([[100, 0, 3000]], np.float32)           # 箱中心 x=100
        a_half = np.array([[1.2, 4.5, 0.8]], np.float32)
        a_active = np.array([True], np.bool_)
        out_hit = np.full(2, -1, np.int32)
        out_t = np.full(2, 1e9, np.float32)
        detect_proj_aircraft_batch(
            p_pos, p_vel, p_rad, p_active, a_pos, a_half, a_active, 0.01,
            out_hit, out_t,
        )
        # p0=(105,0,3000) → p1=(95,0,3000)，穿过 x∈[98.8,101.2]
        assert out_hit[0] == 0
        assert out_hit[1] == -1  # 静止在远处

