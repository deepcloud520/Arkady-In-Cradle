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

from Airia.lulilo.combat.damage import ArmorType
from Airia.lulilo.combat.engagement import (
    FiringControl,
    UnitCombatConfig,
    solve_direct_fire_elevation,
)
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    PartType,
    ProjectilePool,
)
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.logistics.fuel import FuelManager
from Airia.lulilo.simulation.loop import EventBus, SimulationLoop
from Airia.lulilo.simulation.serialization import (
    deserialize_battlefield,
    load_from_file,
    save_to_file,
    serialize_battlefield,
)
from Airia.lulilo.simulation.state import SimulationState


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

    def chance_batch(self, prob, size: int) -> np.ndarray:
        """试验桩：禁止归一化暴击（恒 False）→ 回到纯血量确定性判定。"""
        return np.zeros(size, dtype=bool)


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
    """生成一辆带全套碰撞箱的坦克（车体/炮塔/引擎/履带/弹药架）。

    parts 元组：7 元素 (part, hx, hy, hz, lx, ly, lz) 或
    9 元素追加 (armor_mm, shock_factor)——部位自身装甲厚度 + 震荡系数
    （2026-08-18；缺省 0/1.0 → 命中判定兜底整车值）。
    """
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        yaw=yaw,
    )
    for spec in (parts or DEFAULT_PARTS):
        part, hx, hy, hz, lx, ly, lz = spec[:7]
        part_mm = float(spec[7]) if len(spec) > 7 else 0.0    # 局部部位厚度（勿遮蔽参数 armor_mm）
        part_shock = float(spec[8]) if len(spec) > 8 else 1.0
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz,
                           armor_mm=part_mm, shock_factor=part_shock)
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
# 直射火控仰角修正（2026-08-18：修复「直射弹道下坠」）
# ===========================================================================


class TestDirectFireSuperelevation:
    def test_solve_direct_fire_elevation_signs(self) -> None:
        """低伸解：近距（下坠<高度差）负仰角；远距（下坠>高度差）正仰角；不可达 None。"""
        # 近距 200m：下坠 ≈0.08m < z0=0.5 → 炮口下压（负仰角）
        near = solve_direct_fire_elevation(1555.0, 200.0, 0.5, 9.81)
        assert near is not None and near < 0.0
        # 远距 4000m：下坠 ≈33m > z0=0.5 → 上抛（正仰角）
        far = solve_direct_fire_elevation(1555.0, 4000.0, 0.5, 9.81)
        assert far is not None and far > 0.0
        # 初速不足 → 不可达
        assert solve_direct_fire_elevation(50.0, 4000.0, 0.5, 9.81) is None

    def test_long_range_direct_fire_hits(self) -> None:
        """长距直射不再下坠脱靶：超仰火控使弹丸落在目标中心高度。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(
            armor, boxes, ammo, fuel, fc, 1500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
        )
        rec = fc.fire(m1, t90)
        assert rec is not None
        before = hull_health(armor, boxes, t90)
        fc.tick(3.0)  # 1500m ≈ 0.96s 飞行，含超仰弧线
        assert hull_health(armor, boxes, t90) < before
        assert fc.pending_projectiles() == 0

    def test_fire_leads_moving_target(self) -> None:
        """前置量（预留量）：按目标速度预测命中点，aim 前移到预测位置。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        t90 = spawn_tank(armor, boxes, ammo, fuel, fc, 500.0, 0.0, yaw=np.pi)
        armor.vel[t90] = (10.0, 0.0, 0.0)  # 目标以 10 m/s 沿 +x 移动
        rec = fc.fire(m1, t90)
        assert rec is not None
        # 前置量 = 速度 × 飞行时间（t ≈ R / 初速；M829A3 初速 1555 m/s）
        mv = 1555.0
        t_flight = 500.0 / mv
        assert rec.aim_x == pytest.approx(500.0 + 10.0 * t_flight, abs=0.5)
        assert rec.aim_y == pytest.approx(0.0, abs=1e-3)


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
        """物理命中炮塔（前凸布局）→ TURRET box 失活 → 无法开火。

        注（2026-08-18，穿深管线）：M829A3 现走 A-T 长杆模型（0m≈523mm，
        物理模型，未启用标定锚），TURRET 270mm 单发后效 ≈45.7 < 60 满血，
        需两发（双射手）才能毁伤炮塔；旧 DeMarre 单发 70.6 即可。测试意图
        （炮塔毁 → 失能）不变，仅射弹数适配新穿深量级。
        """
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
        m1b = spawn_tank(armor, boxes, ammo, fuel, fc, 20.0, 0.0)
        victim = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
            parts=parts,
        )
        fc.fire(m1, victim)
        fc.fire(m1b, victim)
        fc.tick(1.0)
        tmask = boxes.active & (boxes.parent == victim) & (boxes.part == int(PartType.TURRET))
        assert not np.any(tmask)  # 炮塔 box 被击毁失活
        assert not fc.can_fire(victim)

    def test_part_damage_sets_armor_flag_for_renderer(self) -> None:
        """部位毁伤 = 单位状态：TURRET 毁 → armor.parts_destroyed 置位（渲染层数据源）。

        回归（2026-08-18）：旧渲染聚合靠 box.parent/active 推断，但引擎毁伤
        即释放槽位（parent 置 -1、回 free_stack），已毁部位永远显示不出——
        「一直扣血但部位不坏」。模拟层毁伤时必须同步置单位级标记。

        注（2026-08-18，穿深管线）：A-T 长杆模型下单发后效不足以毁伤
        TURRET 满血（60），用双射手两发（同 test_part_damage_turret_via_physic_hit）。
        """
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        # 炮塔前凸布局（同 test_part_damage_turret_via_physic_hit）
        parts = [
            (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
            (PartType.TURRET, 1.0, 1.2, 0.6, 2.0, 0.0, 0.6),
            (PartType.AMMO_RACK, 0.7, 0.8, 0.5, -0.5, 0.0, -0.3),
        ]
        m1 = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        m1b = spawn_tank(armor, boxes, ammo, fuel, fc, 20.0, 0.0)
        victim = spawn_tank(
            armor, boxes, ammo, fuel, fc, 500.0, 0.0,
            yaw=np.pi, weapon="2a46m_125mm",
            armor_mm=300.0, armor_type=int(ArmorType.RHA),
            parts=parts,
        )
        # 满血单位：无部位毁伤
        assert not np.any(armor.parts_destroyed[victim])
        fc.fire(m1, victim)
        fc.fire(m1b, victim)
        fc.tick(1.0)
        # 炮塔 box 已毁（槽位释放：active=False, parent=-1）…
        tmask = boxes.active & (boxes.parent == victim) & (boxes.part == int(PartType.TURRET))
        assert not np.any(tmask)
        # …但单位级毁伤标记仍在（渲染层读这里，不依赖 box 槽位生命周期）
        assert bool(armor.parts_destroyed[victim, int(PartType.TURRET)])
        assert not armor.parts_destroyed[victim, int(PartType.HULL)]
        assert not armor.parts_destroyed[victim, int(PartType.AMMO_RACK)]

    def test_per_part_armor_mm(self) -> None:
        """每部位自身装甲厚度（BoxPool.armor_mm）覆盖整车兜底值。

        回归（2026-08-18）：部位厚度不再统一由 cfg.armor_mm × PART_ARMOR_FACTOR
        推导。TRACK 700mm（>650 穿深）不穿；TRACK 100mm 击穿毁伤——
        旧兜底 900×0.5=450mm 两车都会穿，区分不出薄厚。
        """
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        # 两个射手（m256 冷却 10s，单射手连发第二发会被冷却挡掉）
        m1a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        m1b = spawn_tank(armor, boxes, ammo, fuel, fc, 20.0, 0.0)

        def _mk(x: float, y: float, track_mm: float) -> int:
            # TRACK 前凸到局部 x=+1.6（半 0.5 → 世界 x∈[pos+1.1, pos+2.1]），
            # yaw=π 时朝向射手；与 armor AABB（half_x=1.5）重叠 0.4m，
            # 碰撞粗筛能命中（同 test_part_damage_turret_via_physic_hit 布局）
            parts = [
                (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0, 900.0),
                (PartType.TRACK, 0.5, 1.4, 0.5, 1.6, 0.0, 0.0, track_mm),
            ]
            return spawn_tank(
                armor, boxes, ammo, fuel, fc, x, y,
                yaw=np.pi, weapon="2a46m_125mm",
                armor_mm=900.0, armor_type=int(ArmorType.RHA),
                parts=parts,
            )

        thick = _mk(500.0, 0.0, 700.0)      # TRACK 700mm > 650 穿深 → 不穿
        thin = _mk(600.0, 100.0, 100.0)     # TRACK 100mm < 650 → 击穿（y 错开，避免弹道误穿 thick）
        # 直瞄前凸 TRACK 中心（yaw=π → 世界 x = pos_x − 1.6，z=0）：超仰火控后弹丸
        # 落在该点命中 TRACK。旧测试靠「直瞄车顶 + 重力下坠误坠履带」命中，已随
        # 直射弹道下坠修复失效（2026-08-18），改用 aim_at 显式瞄准履带。
        fc.fire(m1a, thick, aim_at=(498.4, 0.0, 0.0))
        fc.fire(m1b, thin, aim_at=(598.4, 100.0, 0.0))
        fc.tick(1.0)
        # 厚履带：部位 700mm 生效 → TRACK 未毁、仅微震荡
        assert np.any(
            boxes.active & (boxes.parent == thick) & (boxes.part == int(PartType.TRACK))
        )
        assert not armor.parts_destroyed[thick, int(PartType.TRACK)]
        # 薄履带：击穿 → TRACK 毁伤（单位级标记置位）
        assert not np.any(
            boxes.active & (boxes.parent == thin) & (boxes.part == int(PartType.TRACK))
        )
        assert bool(armor.parts_destroyed[thin, int(PartType.TRACK)])

    def test_shock_factor_zero_turret_no_shock_damage(self) -> None:
        """无人炮塔（shock_factor=0）：打不穿也不传导震荡扣 HULL 血。

        回归（2026-08-18）：未击穿震荡 = 后效 × SHOCK_DMG_RATIO × 部位 shock_factor。
        T-14 式无人炮塔未击穿 → HULL 血完全不动；有人炮塔对照 → 仅微震荡。
        """
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(
            rng=FixedRNG(0.9)
        )
        # 两个射手（m256 冷却 10s，单射手连发第二发会被冷却挡掉）
        m1a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        m1b = spawn_tank(armor, boxes, ammo, fuel, fc, 20.0, 0.0)

        def _mk(x: float, y: float, shock: float) -> int:
            parts = [
                (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0, 900.0, 1.0),
                (PartType.TURRET, 1.0, 1.2, 0.6, 2.0, 0.0, 0.6, 900.0, shock),
            ]
            return spawn_tank(
                armor, boxes, ammo, fuel, fc, x, y,
                yaw=np.pi, weapon="2a46m_125mm",
                armor_mm=900.0, armor_type=int(ArmorType.RHA),
                parts=parts,
            )

        auto = _mk(500.0, 0.0, 0.0)    # 无人炮塔（T-14 风格）
        manned = _mk(600.0, 100.0, 1.0)  # 有人炮塔（y 错开，避免弹道误穿 auto）
        fc.fire(m1a, auto)
        fc.fire(m1b, manned)
        fc.tick(1.0)
        # 弹先碰前凸炮塔（900mm > 650 穿深）→ 未击穿
        assert hull_health(armor, boxes, auto) == pytest.approx(100.0)   # 无人：零震荡
        assert 0.0 < 100.0 - hull_health(armor, boxes, manned) <= 15.0   # 有人：微震荡
        # 两车炮塔均未毁（未击穿），单位存活
        assert not armor.parts_destroyed[auto, int(PartType.TURRET)]
        assert not armor.parts_destroyed[manned, int(PartType.TURRET)]
        assert armor.active[auto] and armor.active[manned]

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
