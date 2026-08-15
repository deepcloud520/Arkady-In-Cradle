"""Phase 7 步骤①：AircraftPool / AirFormationPool 与各池接缝单测。

覆盖（docs/phase7-air-pipeline.md §十四 步骤① 验收）：
    spawn/destroy 生命周期、free_stack 复用、扩容 + 非零默认回填、空间航路点、
    BoxPool.parent_kind 飞机同步、EmitterPool 飞机父实体、ProjectilePool.guid_target_vel、
    序列化往返（AircraftPool + AirFormationPool + 弹丸新字段）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.entities.pools import (
    MAX_AIR_FORMATIONS,
    MAX_AIRCRAFT,
    AirFormationPool,
    AircraftPool,
    AircraftRole,
    AircraftState,
    BoxPool,
    EMITTER_PARENT_AIRCRAFT,
    EMITTER_KIND_IR_SOURCE,
    EMITTER_KIND_PLATFORM_RADAR,
    EmitterBand,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.simulation.serialization import (
    deserialize_battlefield,
    serialize_battlefield,
)
from Airis.lulilo.simulation.state import SimulationState


# ---------------------------------------------------------------------------
# AircraftPool
# ---------------------------------------------------------------------------


class TestAircraftPool:
    """AircraftPool 生命周期测试。"""

    @pytest.fixture
    def pool(self) -> AircraftPool:
        return AircraftPool()

    def test_spawn_basic(self, pool: AircraftPool) -> None:
        idx = pool.spawn(100.0, 200.0, 3000.0, side=1, role=int(AircraftRole.FIGHTER))
        assert idx >= 0
        assert pool.active[idx]
        assert pool.pos[idx, 0] == pytest.approx(100.0)
        assert pool.pos[idx, 1] == pytest.approx(200.0)
        assert pool.pos[idx, 2] == pytest.approx(3000.0)
        assert pool.side[idx] == 1
        assert pool.role[idx] == int(AircraftRole.FIGHTER)
        assert pool.state[idx] == int(AircraftState.CRUISE)  # 默认巡航

    def test_spawn_defaults(self, pool: AircraftPool) -> None:
        """spawn 默认字段应回填非零基线（避免残值）。"""
        idx = pool.spawn(0, 0, 0)
        assert pool.health[idx] == pytest.approx(100.0)
        assert pool.dmg_factor[idx] == pytest.approx(1.0)
        assert pool.fuel[idx] == pytest.approx(1.0)
        assert pool.comm[idx] == pytest.approx(1.0)
        assert pool.rcs[idx] == pytest.approx(5.0)
        assert pool.target_idx[idx] == -1
        assert pool.formation_id[idx] == -1
        assert pool.contact_age[idx] == pytest.approx(1e9)
        assert pool.bingo[idx] == pytest.approx(0.3)

    def test_destroy_releases_index(self, pool: AircraftPool) -> None:
        idx = pool.spawn(0, 0, 0)
        pool.destroy(idx)
        assert not pool.active[idx]
        idx2 = pool.spawn(1, 1, 1)
        assert idx2 == idx, "销毁后的索引应被立即复用"

    def test_active_count_and_indices(self, pool: AircraftPool) -> None:
        assert pool.active_count == 0
        a = pool.spawn(0, 0, 0)
        b = pool.spawn(1, 1, 1)
        assert pool.active_count == 2
        pool.destroy(a)
        idxs = pool.active_indices()
        assert int(idxs[0]) == b
        assert pool.active_count == 1

    def test_set_waypoints(self, pool: AircraftPool) -> None:
        idx = pool.spawn(0, 0, 0)
        pool.set_waypoints(idx, [(100, 200, 3000), (400, 500, 3500)])
        assert pool.wp_count[idx] == 2
        assert pool.wp[idx, 0, 0] == pytest.approx(100.0)
        assert pool.wp[idx, 1, 2] == pytest.approx(3500.0)
        assert pool.wp_idx[idx] == 0

    def test_expand_backfills_nonzero_defaults(self) -> None:
        """扩容后新段必须回填非零默认（6c 修的坑，AGENTS.md §3.1）。"""
        pool = AircraftPool(capacity=4)
        for _ in range(5):  # 触发 _expand
            pool.spawn(0, 0, 0)
        assert pool.capacity > 4
        assert pool.active_count == 5
        # 扩容段索引的默认值
        for i in pool.active_indices():
            i = int(i)
            assert pool.health[i] == pytest.approx(100.0)
            assert pool.dmg_factor[i] == pytest.approx(1.0)
            assert pool.fuel[i] == pytest.approx(1.0)
            assert pool.comm[i] == pytest.approx(1.0)
            assert pool.rcs[i] == pytest.approx(5.0)
            assert pool.target_idx[i] == -1


class TestAirFormationPool:
    """AirFormationPool 轻量池测试。"""

    def test_spawn_destroy(self) -> None:
        pool = AirFormationPool()
        f = pool.spawn(side=1, target_zone=(100.0, 200.0, 3000.0))
        assert pool.active[f]
        assert pool.side[f] == 1
        assert pool.member_count[f] == 0
        assert pool.comm[f] == pytest.approx(1.0)
        pool.destroy(f)
        assert not pool.active[f]

    def test_expand(self) -> None:
        pool = AirFormationPool(capacity=2)
        for _ in range(3):
            pool.spawn()
        assert pool.capacity > 2
        assert pool.active_count == 3


# ---------------------------------------------------------------------------
# BoxPool.parent_kind（飞机父实体）
# ---------------------------------------------------------------------------


class TestBoxAircraftParent:
    """飞机挂机身碰撞箱（parent_kind=1）的同步。"""

    def test_sync_aircraft_box(self) -> None:
        aircraft = AircraftPool()
        air_idx = aircraft.spawn(100.0, 200.0, 3000.0)
        aircraft.heading[air_idx] = np.pi / 2  # 90° 航向
        boxes = BoxPool(capacity=8)
        # 机身箱：局部偏移 (1,0,0)，heading=90° → 世界偏移应为 (0,1,0)
        from Airis.lulilo.entities.pools import ArmorPool

        armor = ArmorPool(capacity=1)  # 空装甲占位（飞机测试不涉及地面箱）
        b = boxes.spawn(air_idx, int(PartType.AIRCRAFT), 1.2, 4.5, 0.8,
                        local_x=1.0, local_y=0.0, local_z=0.0,
                        parent_kind=1)
        boxes.sync_all_from_parent(armor, aircraft=aircraft)
        assert boxes.active[b]
        assert boxes.pos[b, 0] == pytest.approx(100.0, abs=1e-4)
        assert boxes.pos[b, 1] == pytest.approx(201.0, abs=1e-4)
        assert boxes.pos[b, 2] == pytest.approx(3000.0, abs=1e-4)
        # 父飞机销毁 → 箱级联失活
        aircraft.destroy(air_idx)
        boxes.sync_all_from_parent(armor, aircraft=aircraft)
        assert not boxes.active[b]

    def test_armor_box_still_works_without_aircraft(self) -> None:
        """缺省 aircraft=None 时装甲箱行为不变（兼容旧调用点）。"""
        from Airis.lulilo.entities.pools import ArmorPool

        armor = ArmorPool()
        a = armor.spawn(0, 0, 0)
        armor.yaw[a] = 0.0
        boxes = BoxPool(capacity=8)
        b = boxes.spawn(a, int(PartType.HULL), 1.0, 3.0, 1.0, local_x=0.0, local_y=0.0, local_z=1.0)
        boxes.sync_all_from_parent(armor)
        assert boxes.pos[b, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EmitterPool 飞机父实体
# ---------------------------------------------------------------------------


class TestEmitterAircraftParent:
    """机载雷达 / 尾焰绑定飞机并随位置同步。"""

    def test_activate_aircraft_radar(self) -> None:
        from Airis.lulilo.entities.pools import EmitterPool, ArmorPool, ProjectilePool

        aircraft = AircraftPool()
        air_idx = aircraft.spawn(0, 0, 3000.0)
        em = EmitterPool(capacity=16)
        e = em.activate_aircraft_radar(air_idx, 70.0)
        assert em.parent_kind[e] == EMITTER_PARENT_AIRCRAFT
        assert em.parent_idx[e] == air_idx
        assert em.band[e] == int(EmitterBand.RF)
        assert em.kind[e] == EMITTER_KIND_PLATFORM_RADAR
        # 飞机移动 → sync → emitter 位置跟随
        aircraft.pos[air_idx] = (500.0, 600.0, 3500.0)
        armor = ArmorPool(capacity=1)
        proj = ProjectilePool(capacity=1)
        em.sync_from_parent(armor, proj, 0.1, aircraft=aircraft)
        assert em.pos[e, 0] == pytest.approx(500.0)
        assert em.pos[e, 1] == pytest.approx(600.0)
        assert em.pos[e, 2] == pytest.approx(3500.0)

    def test_aircraft_destroy_cascades_rf(self) -> None:
        from Airis.lulilo.entities.pools import EmitterPool, ArmorPool, ProjectilePool

        aircraft = AircraftPool()
        air_idx = aircraft.spawn(0, 0, 3000.0)
        em = EmitterPool(capacity=16)
        e = em.activate_aircraft_radar(air_idx, 70.0)
        aircraft.destroy(air_idx)
        armor = ArmorPool(capacity=1)
        proj = ProjectilePool(capacity=1)
        em.sync_from_parent(armor, proj, 0.1, aircraft=aircraft)
        assert not em.active[e], "飞机销毁 → 非红外 emitter 级联失活"

    def test_aircraft_ir_charge(self) -> None:
        from Airis.lulilo.entities.pools import EmitterPool, ArmorPool, ProjectilePool

        aircraft = AircraftPool()
        air_idx = aircraft.spawn(0, 0, 3000.0)
        em = EmitterPool(capacity=16)
        e = em.activate_aircraft_ir(air_idx, 1.0)
        assert em.band[e] == int(EmitterBand.IR)
        assert em.kind[e] == EMITTER_KIND_IR_SOURCE
        assert em.cur_power[e] == pytest.approx(0.0)  # 从 0 充能
        armor = ArmorPool(capacity=1)
        proj = ProjectilePool(capacity=1)
        em.sync_from_parent(armor, proj, 0.1, aircraft=aircraft)
        assert em.cur_power[e] == pytest.approx(1.0)  # 父存活 → 充能到额定


# ---------------------------------------------------------------------------
# ProjectilePool.guid_target_vel
# ---------------------------------------------------------------------------


class TestProjectileGuidTargetVel:
    """AAM 目标速度字段（docs/phase7-air-pipeline.md §7.2）。"""

    def test_spawn_zero_and_set(self) -> None:
        p = ProjectilePool(capacity=8)
        idx = p.spawn(0, 0, 0, 0, 0, 0)
        assert p.guid_target_vel[idx, 0] == pytest.approx(0.0)
        p.guid_target_vel[idx] = (250.0, 0.0, 0.0)
        assert p.guid_target_vel[idx, 0] == pytest.approx(250.0)

    def test_destroy_clears(self) -> None:
        p = ProjectilePool(capacity=8)
        idx = p.spawn(0, 0, 0, 0, 0, 0)
        p.guid_target_vel[idx] = (250.0, 0.0, 0.0)
        p.destroy(idx)
        idx2 = p.spawn(0, 0, 0, 0, 0, 0)
        assert p.guid_target_vel[idx2, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 序列化往返
# ---------------------------------------------------------------------------


class TestSerialization:
    """AircraftPool / AirFormationPool / 弹丸新字段 无损往返。"""

    def test_aircraft_round_trip(self) -> None:
        aircraft = AircraftPool(capacity=8)
        idx = aircraft.spawn(100.0, 200.0, 3000.0, side=1, role=int(AircraftRole.FIGHTER))
        aircraft.vel[idx] = (250.0, 0.0, 0.0)
        aircraft.health[idx] = 55.0
        aircraft.fuel[idx] = 0.7
        aircraft.rwr_state[idx] = 3
        aircraft.set_waypoints(idx, [(1000, 1000, 3000)])

        state = SimulationState(aircraft_pool=aircraft)
        blob = serialize_battlefield(state)
        save = deserialize_battlefield(blob)
        assert save.aircraft is not None
        a2 = save.aircraft
        assert a2.capacity == aircraft.capacity
        assert a2.active_count == 1
        j = int(a2.active_indices()[0])
        assert a2.pos[j, 0] == pytest.approx(100.0)
        assert a2.vel[j, 0] == pytest.approx(250.0)
        assert a2.health[j] == pytest.approx(55.0)
        assert a2.fuel[j] == pytest.approx(0.7)
        assert a2.rwr_state[j] == 3
        assert a2.wp[j, 0, 2] == pytest.approx(3000.0)
        assert a2.comm[j] == pytest.approx(1.0)
        # 读档后可正常销毁（free_stack 重建正确）
        a2.destroy(j)
        assert a2.active_count == 0

    def test_air_formation_round_trip(self) -> None:
        pool = AirFormationPool(capacity=4)
        f = pool.spawn(side=1, target_zone=(10.0, 20.0, 30.0))
        pool.member_count[f] = 3

        state = SimulationState(air_formation_pool=pool)
        blob = serialize_battlefield(state)
        save = deserialize_battlefield(blob)
        assert save.air_formations is not None
        g = int(save.air_formations.active_indices()[0])
        assert save.air_formations.side[g] == 1
        assert save.air_formations.member_count[g] == 3
        assert save.air_formations.target_zone[g, 0] == pytest.approx(10.0)

    def test_projectile_guid_target_vel_round_trip(self) -> None:
        from Airis.lulilo.entities.pools import MAX_PROJ

        p = ProjectilePool(capacity=16)
        idx = p.spawn(0, 0, 0, 10, 0, 0)
        p.guid_target_vel[idx] = (250.0, 30.0, 5.0)

        state = SimulationState(projectile_pool=p)
        blob = serialize_battlefield(state)
        save = deserialize_battlefield(blob)
        assert save.projectiles is not None
        j = int(save.projectiles.active_indices()[0])
        assert save.projectiles.guid_target_vel[j, 0] == pytest.approx(250.0)
        assert save.projectiles.guid_target_vel[j, 1] == pytest.approx(30.0)

    def test_constants(self) -> None:
        assert MAX_AIRCRAFT == 128
        assert MAX_AIR_FORMATIONS == 32
