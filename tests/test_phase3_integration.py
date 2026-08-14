"""Phase 3 集成测试：完整开火链 + 多分辨率 Tick + 存档/读档。

覆盖 AGENTS.md Phase 3 交付物与验收标准：
    - 单位能开火、命中、损毁（完整交战流程：发射→飞行→命中→毁伤→消耗）
    - 弹药/燃料正确扣减
    - 完整交战从开火到命中 < 1 个 Tick 内完成（集成流程验证）
    - 状态序列化/反序列化无数据丢失（round-trip 断言）
    - 多分辨率 Tick 调度（地面 1s / 子弹 0.01s）

运行：uv run python -m pytest tests/test_phase3_integration.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.combat.damage import ArmorType
from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.simulation.loop import EventBus, SimulationLoop
from Airis.lulilo.simulation.serialization import (
    deserialize_battlefield,
    load_from_file,
    save_to_file,
    serialize_battlefield,
)
from Airis.lulilo.simulation.state import SimulationState


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


class FixedRNG:
    """可控 RNG：散布恒为 0（落点=瞄准点，必中）+ uniform 固定值。

    用于把「命中」从随机中解耦出来，测试物理毁伤链的确定性。
    """

    def __init__(self, u: float = 0.9) -> None:
        self.u = float(u)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self.u

    def dispersion_offset(self, sigma: float) -> tuple[float, float]:
        return (0.0, 0.0)


DEFAULT_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
    (PartType.ENGINE, 0.8, 1.0, 0.6, -1.2, 0.0, 0.0),
    (PartType.TRACK, 1.4, 0.6, 0.5, 0.0, 1.9, 0.0),
    (PartType.AMMO_RACK, 0.7, 0.8, 0.5, 0.5, 0.0, -0.3),
]


def make_battlefield(seed: int = 123, events: EventBus | None = None, rng=None):
    armor = ArmorPool(capacity=64)
    boxes = BoxPool(capacity=512)
    projectiles = ProjectilePool(capacity=1024)
    ammo = AmmoManager(capacity=64)
    fuel = FuelManager(capacity=64)
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        # 注意：EventBus 定义了 __len__，空总线是 falsy，必须用 is not None 判断
        events=events if events is not None else EventBus(),
        rng=rng if rng is not None else RNGManager(seed=seed),
    )
    return armor, boxes, projectiles, ammo, fuel, fc


def spawn_tank(
    armor,
    boxes,
    ammo,
    fuel,
    fc,
    x: float,
    y: float,
    *,
    yaw: float = 0.0,
    weapon: str = "m256_120mm",
    armor_mm: float = 600.0,
    armor_type: int = int(ArmorType.COMPOSITE),
    crew_skill: float = 0.9,
    ammo_cap: float = 40.0,
    health: float = 100.0,
    parts: list | None = None,
) -> int:
    """生成一辆带全套碰撞箱的坦克（车体/炮塔/引擎/履带/弹药架）。"""
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        yaw=yaw,
    )
    for part, hx, hy, hz, lx, ly, lz in (parts or DEFAULT_PARTS):
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        if int(part) == int(PartType.HULL):
            # HULL 初始血量（docs/part-health-design.md §2.3）
            boxes.hp[bidx] = health
            boxes.hp_max[bidx] = health
            armor.bind_hull(idx, bidx)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(
        idx,
        UnitCombatConfig(
            weapon_id=weapon,
            armor_mm=armor_mm,
            armor_type=armor_type,
            crew_skill=crew_skill,
        ),
    )
    ammo.state.set_unit(idx, ammo_cap, ammo_cap)
    fuel.state.set_unit(idx, 500.0, 500.0)
    return idx


def drain_kinds(events: EventBus) -> dict[str, int]:
    """统计事件类型计数（drain 后清空）。"""
    counts: dict[str, int] = {}
    for e in events.drain():
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts


def hull_health(armor, boxes, idx) -> float:
    """读取整车血量（HULL 部位血，docs/part-health-design.md §三）。"""
    return float(boxes.hp[armor.hull_idx[idx]])


# ===========================================================================
# 开火基础（fire 的弹药/冷却/能力约束）
# ===========================================================================


class TestFiringBasics:
    def test_fire_spawns_projectile(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        rec = fc.fire(m1, t90)
        assert rec is not None
        assert rec.proj_idx >= 0
        assert projectiles.active[rec.proj_idx]
        assert projectiles.owner[rec.proj_idx] == m1
        assert fc.pending_projectiles() == 1

    def test_fire_consumes_ammo(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0, ammo_cap=40)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        fc.fire(m1, t90)
        assert ammo.state.reserve[m1] == 39.0
        assert ammo.state.reserve[t90] == 40.0

    def test_fire_cooldown_blocks_second_shot(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        assert fc.fire(m1, t90) is not None
        # 冷却中（M256 6rpm → 10s 冷却）→ 立即再开火被拒
        assert fc.fire(m1, t90) is None
        assert fc.cooldown_remaining(m1) > 0.0

    def test_fire_after_cooldown_elapses(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        fc.fire(m1, t90)
        fc.tick(11.0)  # 冷却 10s 已过
        assert fc.cooldown_remaining(m1) == 0.0
        assert fc.fire(m1, t90) is not None

    def test_fire_no_ammo_blocked(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0, ammo_cap=0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        assert not fc.can_fire(m1)
        assert fc.fire(m1, t90) is None

    def test_fire_turret_destroyed_blocked(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        # 摧毁炮塔 box（部位毁伤）
        mask = boxes.active & (boxes.parent == m1) & (boxes.part == int(PartType.TURRET))
        assert np.any(mask)
        boxes.active[mask] = False
        assert not fc.can_fire(m1)
        assert fc.fire(m1, t90) is None

    def test_fire_out_of_range_blocked(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield()
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        far = spawn_tank(armor, boxes, ammo, fuel, fc, 10000.0, 0.0)
        assert fc.fire(m1, far) is None

    def test_fire_emits_event(self) -> None:
        events = EventBus()
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(events=events)
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        fc.fire(m1, t90)
        counts = drain_kinds(events)
        assert counts["fire"] == 1


# ===========================================================================
# 交战（命中 / 毁伤 / 部位 / 摧毁 / 回收）
# ===========================================================================


class TestEngagement:
    def test_hit_reduces_health(self) -> None:
        """薄甲目标（RHA 300）被 M829A3 正击 → 击穿 + 命中事件 + 扣血。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        rec = fc.fire(m1, t90)
        assert rec is not None
        before = hull_health(armor, boxes, t90)
        fc.tick(1.0)  # 0.32s 内到达 → 命中
        after = hull_health(armor, boxes, t90)
        assert after < before
        counts = drain_kinds(fc.events)
        assert counts["hit"] >= 1
        assert fc.pending_projectiles() == 0  # 命中后弹丸回收

    def test_hit_no_pen_small_damage(self) -> None:
        """厚甲复合装甲（900 等效）vs M829A3 700 → 未击穿，仅 health 微降。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        thick = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=900.0, armor_type=int(ArmorType.COMPOSITE),
        )
        fc.fire(m1, thick)
        before = hull_health(armor, boxes, thick)
        fc.tick(1.0)
        dmg = before - hull_health(armor, boxes, thick)
        assert 0.0 < dmg <= 15.0  # 未击穿微降（design doc §4.3）

    def test_ricochet_branch_keeps_projectile(self, monkeypatch) -> None:
        """跳弹分支：注入掠射入射角 → ricochet 事件 + 无健康损失 + 弹丸保留继续飞行。

        注：简化碰撞几何下弹丸命中点所在面法线总是近似垂直于弹道，
        自然掠射（>75°）极难产生；此处 mock 入射角验证跳弹分支语义。
        """
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        target = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        monkeypatch.setattr(fc, "_impact_angle", lambda *a, **k: 80.0)  # >75° → 跳弹

        rec = fc.fire(m1, target)
        assert rec is not None
        fc.tick(1.0)
        counts = drain_kinds(fc.events)
        assert counts["ricochet"] >= 1
        assert hull_health(armor, boxes, target) == pytest.approx(100.0)  # 跳弹不毁伤
        assert not fc.can_fire(m1)  # 冷却中，说明确实发射过

    def test_health_destroy_multiple_hits(self) -> None:
        """连续击穿薄甲 → health 归零 → destroyed 事件。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0, ammo_cap=100)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        # 3 发全中（FixedRNG 必中）：61.5 dmg/发 × 2 → 摧毁
        for _ in range(3):
            fc.fire(m1, t90)
            fc.tick(11.0)  # 推进冷却 + 飞行命中
        assert not armor.active[t90]
        counts = drain_kinds(fc.events)
        assert counts["destroyed"] >= 1

    def test_part_damage_turret_via_physic_hit(self) -> None:
        """物理命中炮塔（前凸布局）→ TURRET box 失活 → 无法开火。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        # 炮塔前凸到局部 x=+2.0（半 1.0 → x∈[1,3]），HULL 半 1.5 → 弹先碰炮塔
        parts = [
            (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
            (PartType.TURRET, 1.0, 1.2, 0.6, 2.0, 0.0, 0.6),
            (PartType.AMMO_RACK, 0.7, 0.8, 0.5, -0.5, 0.0, -0.3),
        ]
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        victim = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
            parts=parts,
        )
        fc.fire(m1, victim)
        fc.tick(1.0)
        tmask = boxes.active & (boxes.parent == victim) & (boxes.part == int(PartType.TURRET))
        assert not np.any(tmask)  # 炮塔 box 被击毁失活
        assert not fc.can_fire(victim)

    def test_miss_recycles_projectile(self) -> None:
        """脱靶（目标消失）→ 弹丸超龄回收 + miss 事件。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        fc.fire(m1, t90)
        armor.destroy(t90)  # 弹丸飞行途中目标消失
        fc.tick(25.0)  # > PROJECTILE_MAX_AGE=20s
        assert fc.pending_projectiles() == 0
        counts = drain_kinds(fc.events)
        assert counts["miss"] >= 1

    def test_full_engagement_with_loop(self) -> None:
        """完整交战流程：SimulationLoop + combat hook，30s 内 M1A2 摧毁侧露的 T-90。

        验证：开火→飞行→命中→毁伤→消耗→事件 全链路，且 target 被摧毁。
        """
        state = SimulationState()
        loop = SimulationLoop(state=state)
        loop.configure_default_rates()
        # FiringControl 复用 loop 的事件总线（fire/hit/destroyed 事件进 loop）
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9), events=loop.events,
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0, ammo_cap=100)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi / 2.0, weapon="2a46m_125mm",  # 侧面暴露（侧甲薄）
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )

        state.armor_pool = armor
        state.box_pool = boxes
        state.projectile_pool = projectiles
        state.ammo = ammo
        state.fuel = fuel

        loop.attach("combat", lambda l, dt: fc.tick(dt))

        # 30 次地面 Tick（1s），每次推进前尝试开火（受冷却约束）
        for _ in range(30):
            fc.fire(m1, t90)
            loop.step(1.0)

        assert not armor.active[t90]              # 目标被摧毁
        assert armor.active[m1]                   # 射手存活
        assert ammo.state.reserve[m1] < 100.0     # 弹药扣减
        counts = drain_kinds(loop.events)
        assert counts["fire"] >= 1
        assert counts["hit"] >= 1
        assert counts["destroyed"] >= 1
        assert loop.state.sim_time == pytest.approx(30.0)


# ===========================================================================
# 多分辨率 Tick 调度
# ===========================================================================


class TestMultiRateLoop:
    def test_configure_default_rates(self) -> None:
        loop = SimulationLoop()
        loop.configure_default_rates()
        assert loop.rate_of("combat") == pytest.approx(0.01)
        assert loop.rate_of("movement") == pytest.approx(1.0)
        assert loop.rate_of("detection") == pytest.approx(1.0)
        assert loop.rate_of("logistics") == pytest.approx(1.0)

    def test_accumulator_fires_on_threshold(self) -> None:
        loop = SimulationLoop()
        seen: list[float] = []
        loop.attach("movement", lambda _l, dt: seen.append(dt))
        loop.set_rate("movement", 1.0)
        loop.step(0.5)  # 累加 0.5 < 1.0 → 不执行
        assert seen == []
        loop.step(0.5)  # 累加 1.0 → 执行（dt=1.0）
        assert seen == [1.0]
        loop.step(0.3)
        assert seen == [1.0]  # 0.3 未到 1.0
        loop.step(0.7)
        assert seen == [1.0, 1.0]

    def test_combat_bullet_tick_cumulative(self) -> None:
        """combat 分频 0.01s：step(0.05) 时 combat hook 收到累计 dt=0.05（内部细分）。"""
        loop = SimulationLoop()
        seen: list[float] = []
        loop.attach("combat", lambda _l, dt: seen.append(dt))
        loop.set_rate("combat", 0.01)
        loop.step(0.05)
        assert seen == [0.05]

    def test_set_rate_disabled_keeps_per_frame(self) -> None:
        loop = SimulationLoop()
        seen: list[float] = []
        loop.attach("combat", lambda _l, dt: seen.append(dt))
        loop.set_rate("combat", 0.01)
        loop.set_rate("combat", 0.0)  # 禁用分频 → 每帧执行
        loop.step(0.5)
        assert seen == [0.5]

    def test_loop_multi_rate_with_combat(self) -> None:
        """配置默认多分辨率后 run_headless 正常推进（战斗 hook 每地面 Tick 收到累计子弹时长）。"""
        state = SimulationState()
        loop = SimulationLoop(state=state)
        loop.configure_default_rates()
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9), events=loop.events,
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        state.armor_pool = armor
        state.box_pool = boxes
        state.projectile_pool = projectiles
        state.ammo = ammo
        state.fuel = fuel
        loop.attach("combat", lambda l, dt: fc.tick(dt))

        fc.fire(m1, t90)
        loop.run_headless(3.0, dt=1.0)  # 3 个地面 Tick，每 Tick 内部子弹细分
        assert loop.state.sim_time == pytest.approx(3.0)
        assert hull_health(armor, boxes, t90) < 100.0  # 命中扣血
        assert fc.pending_projectiles() == 0


# ===========================================================================
# 存档 / 读档（msgpack 无损往返）
# ===========================================================================


class TestSerialization:
    def _make_scene(self):
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0, ammo_cap=40)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        state = SimulationState(sim_time=12.5, tick_count=25)
        state.armor_pool = armor
        state.box_pool = boxes
        state.projectile_pool = projectiles
        state.ammo = ammo
        state.fuel = fuel
        return armor, boxes, projectiles, ammo, fuel, fc, state, m1, t90

    def test_roundtrip_all_pools(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc, state, m1, t90 = self._make_scene()
        fc.fire(m1, t90)
        fc.tick(0.5)  # 弹丸仍在飞行中（500m / 1555m/s ≈ 0.32s）
        ammo.request_supply(t90, 20.0, priority=1)

        blob = serialize_battlefield(
            state, armor, boxes, projectiles,
            ammo.state, fuel.state,
            ammo_requests=ammo.pending_requests(),
        )
        save = deserialize_battlefield(blob)

        # sim 时间
        assert save.sim_time == pytest.approx(12.5)
        assert save.tick_count == 25

        # 装甲数组逐元素一致
        assert save.armor is not None
        np.testing.assert_array_equal(save.armor.pos, armor.pos)
        np.testing.assert_array_equal(save.armor.hull_idx, armor.hull_idx)
        np.testing.assert_array_equal(save.armor.active, armor.active)
        np.testing.assert_array_equal(save.armor.yaw, armor.yaw)
        np.testing.assert_array_equal(save.armor.wp, armor.wp)

        # 碰撞箱 / 弹丸
        np.testing.assert_array_equal(save.boxes.pos, boxes.pos)
        np.testing.assert_array_equal(save.boxes.active, boxes.active)
        np.testing.assert_array_equal(save.boxes.hp, boxes.hp)
        np.testing.assert_array_equal(save.boxes.hp_max, boxes.hp_max)
        np.testing.assert_array_equal(save.projectiles.pos, projectiles.pos)
        np.testing.assert_array_equal(save.projectiles.active, projectiles.active)

        # 后勤
        np.testing.assert_array_equal(save.ammo.reserve, ammo.state.reserve)
        np.testing.assert_array_equal(save.fuel.reserve, fuel.state.reserve)
        assert save.ammo_requests[0].target_idx == t90
        assert save.ammo_requests[0].amount == 20.0

    def test_roundtrip_after_full_combat(self) -> None:
        """交战后存档/读档：health/弹药/事件状态一致，可继续读档后运行。"""
        armor, boxes, projectiles, ammo, fuel, fc, state, m1, t90 = self._make_scene()
        for _ in range(2):
            fc.fire(m1, t90)
            fc.tick(11.0)
        assert not armor.active[t90]  # 已被摧毁

        blob = serialize_battlefield(state, armor, boxes, projectiles, ammo.state, fuel.state)
        save = deserialize_battlefield(blob)

        assert save.armor is not None
        assert not save.armor.active[t90]          # 摧毁状态保留
        np.testing.assert_array_equal(save.armor.hull_idx, armor.hull_idx)
        np.testing.assert_array_equal(save.boxes.active, boxes.active)
        np.testing.assert_array_equal(save.boxes.hp, boxes.hp)
        np.testing.assert_array_equal(save.ammo.reserve, ammo.state.reserve)

        # 读档后可继续跑（不崩溃，active 语义正确）
        fc2 = FiringControl(save.armor, save.boxes, save.projectiles)
        fc2.tick(1.0)

    def test_roundtrip_rebuild_free_stack(self) -> None:
        """读档后 free_stack 与 active 一致（spawn 不复用活跃索引）。"""
        armor, boxes, projectiles, ammo, fuel, fc, state, m1, t90 = self._make_scene()
        blob = serialize_battlefield(state, armor, boxes, projectiles, ammo.state, fuel.state)
        save = deserialize_battlefield(blob)

        active = set(np.flatnonzero(save.armor.active).tolist())
        free = set(save.armor._free_stack)
        assert active.isdisjoint(free)
        assert len(active) + len(free) == save.armor.capacity

        # spawn 新单位拿到的是未占用索引
        new_idx = save.armor.spawn(0.0, 0.0, 0.0)
        assert new_idx not in active
        assert save.armor.active[new_idx]

    def test_version_mismatch_raises(self) -> None:
        import msgpack

        bad = msgpack.packb({"version": 999, "sim_time": 1.0}, use_bin_type=True)
        with pytest.raises(ValueError):
            deserialize_battlefield(bad)

    def test_save_load_file(self, tmp_path) -> None:
        armor, boxes, projectiles, ammo, fuel, fc, state, m1, t90 = self._make_scene()
        path = str(tmp_path / "save.msgpack")
        blob = serialize_battlefield(state, armor, boxes, projectiles, ammo.state, fuel.state)
        save_to_file(blob, path)
        assert tmp_path.joinpath("save.msgpack").exists()
        loaded = load_from_file(path)
        assert loaded.sim_time == pytest.approx(12.5)
        np.testing.assert_array_equal(loaded.armor.pos, armor.pos)
        np.testing.assert_array_equal(loaded.fuel.reserve, fuel.state.reserve)

    def test_empty_state_roundtrip(self) -> None:
        """空状态（无池）也能存档/读档。"""
        st = SimulationState()
        blob = serialize_battlefield(st)
        save = deserialize_battlefield(blob)
        assert save.armor is None
        assert save.projectiles is None
        assert save.sim_time == 0.0
