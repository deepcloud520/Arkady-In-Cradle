"""Phase 6c 步骤⑤ 测试：炮兵编队 AI + 曲射火控（docs/phase6-ai-pipeline.md §六）。

覆盖：弹道解算（solve_ballistic）/ 曲射开火（fire_indirect 初速/弹药/冷却）/
落点引爆（artillery_impact + 提交池毁伤）/ 炮兵编队 AI（友军/玩家指令、优先级、
comm 门控、齐射）/ loop 指令路由集成（command_handler STRIKE 直达）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig, solve_ballistic
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.simulation.ai import (
    ArtilleryAI,
    FormationAI,
    UnitRole,
)
from Airis.lulilo.simulation.commands import Command, CommandType
from Airis.lulilo.simulation.loop import EventBus, SimulationLoop
from Airis.lulilo.simulation.state import SimulationState

# 简化碰撞箱（车体 + 炮塔，够开火链 + 部位毁伤用）
_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]

_G = 9.81


def _simulate_ballistic(
    v: float, elev: float, R: float, z0: float, dt: float = 0.01
) -> tuple[float, float]:
    """数值积分验证弹道解算：返回 (落地水平距离, 落地 z)。"""
    vx = v * math.cos(elev)
    vz = v * math.sin(elev)
    x, z = 0.0, z0
    guard = 0
    while z > 0.0 and guard < 100000:
        x += vx * dt
        vz -= _G * dt
        z += vz * dt
        guard += 1
    return x, z


def _spawn_gun(
    armor: ArmorPool,
    boxes: BoxPool,
    fc: FiringControl,
    ammo: AmmoManager,
    x: float,
    y: float,
    *,
    side: int = 0,
    formation_id: int = -1,
    health: float = 100.0,
    rpm: float = 120.0,
    max_range: float = 20000.0,
) -> int:
    """生成一门炮兵（role=ARTILLERY，M109 155mm 曲射配置）。"""
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        side=side, role=int(UnitRole.ARTILLERY), formation_id=formation_id,
    )
    for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        if int(part) == int(PartType.HULL):
            boxes.hp[bidx] = health
            boxes.hp_max[bidx] = health
            armor.bind_hull(idx, bidx)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(
        idx,
        UnitCombatConfig(
            weapon_id="m109_155mm",
            ammo_id="m107_155mm_he_frag",
            armor_mm=60.0,
            crew_skill=0.9,
            max_range_m=max_range,
            fire_rate_rpm=rpm,
        ),
    )
    ammo.state.set_unit(idx, 40.0, 40.0)
    return idx


def _spawn_enemy(
    armor: ArmorPool,
    boxes: BoxPool,
    fc: FiringControl,
    ammo: AmmoManager,
    x: float,
    y: float,
    *,
    health: float = 100.0,
    ammo_capacity: float = 0.0,
) -> int:
    """生成敌方坦克（有装甲配置可被毁伤；无弹药 → 不反击）。"""
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        side=1, role=0,
    )
    for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        if int(part) == int(PartType.HULL):
            boxes.hp[bidx] = health
            boxes.hp_max[bidx] = health
            armor.bind_hull(idx, bidx)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(
        idx,
        UnitCombatConfig(weapon_id="m256_120mm", armor_mm=600.0, crew_skill=0.7),
    )
    ammo.state.set_unit(idx, ammo_capacity, ammo_capacity)
    return idx


def _make_combat(
    seed: int = 7,
) -> tuple[
    ArmorPool, BoxPool, ProjectilePool, AmmoManager, FiringControl, EventBus
]:
    armor = ArmorPool(capacity=32)
    boxes = BoxPool(capacity=256)
    projectiles = ProjectilePool(capacity=512)
    ammo = AmmoManager(capacity=32)
    events = EventBus()
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo,
        events=events,
        rng=RNGManager(seed=seed),
    )
    return armor, boxes, projectiles, ammo, fc, events


# ===========================================================================
# 弹道解算
# ===========================================================================


class TestSolveBallistic:
    def test_fixed_elevation_lands_at_range(self) -> None:
        """固定 45° 反解初速：数值积分落点 ≈ 目标射距。"""
        R = 2000.0
        v, elev = solve_ballistic(0.0, R, 1.5, _G, elevation_deg=45.0)
        assert v == pytest.approx(math.sqrt(R * R * _G / (R + 1.5)), rel=1e-6)
        land_x, _ = _simulate_ballistic(v, elev, R, 1.5)
        assert land_x == pytest.approx(R, abs=15.0)

    def test_auto_solve_uses_muzzle_velocity(self) -> None:
        """默认（elevation_deg=None）：用弹药初速解低伸角，落点 ≈ 射距。"""
        v, elev = solve_ballistic(827.0, 2000.0, 1.5, _G)
        assert v == pytest.approx(827.0)
        assert elev < math.radians(30.0)  # 低伸解
        land_x, _ = _simulate_ballistic(v, elev, 2000.0, 1.5)
        assert land_x == pytest.approx(2000.0, abs=15.0)

    def test_unreachable_returns_none(self) -> None:
        """初速不足（v²/g < R）→ None（射距不可达）。"""
        assert solve_ballistic(100.0, 5000.0, 1.5, _G) is None

    def test_fixed_elevation_too_low_returns_none(self) -> None:
        """仰角过低 + 落点低于炮口 → 无解（z0 + R·tanθ ≤ 0）。"""
        assert solve_ballistic(0.0, 100.0, -50.0, _G, elevation_deg=1.0) is None


# ===========================================================================
# 曲射开火（fire_indirect）
# ===========================================================================


class TestFireIndirect:
    def test_spawns_ballistic_shell(self) -> None:
        """fire_indirect：生成间瞄弹丸（indirect 标记 + 垂直初速 + 弹药消耗 + 冷却）。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0)

        rec = fc.fire_indirect(gun, 2000.0, 500.0)
        assert rec is not None
        assert rec.target_idx == -1
        assert rec.ammo_id == "m107_155mm_he_frag"

        pi = rec.proj_idx
        assert bool(proj.indirect[pi]) is True
        assert proj.vel[pi, 2] > 0.0  # 向上初速（高抛）
        assert proj.guid_stage[pi] == 3  # BALLISTIC（不触发导引头检测）
        assert ammo.state.reserve[gun] == pytest.approx(39.0)  # 每发 1 单位
        assert fc.cooldown_remaining(gun) > 0.0  # 射速冷却

        kinds = {ev.kind for ev in events.drain()}
        assert "fire" in kinds

    def test_out_of_range_returns_none(self) -> None:
        """超射程（cfg.max_range_m）→ None。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, max_range=1000.0)
        assert fc.fire_indirect(gun, 2000.0, 0.0) is None

    def test_no_ammo_returns_none(self) -> None:
        """无弹药（AmmoManager reserve=0）→ can_fire False → None。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0)
        ammo.consume(gun, 40.0)  # 打空
        assert fc.fire_indirect(gun, 1000.0, 0.0) is None

    def test_shell_lands_at_aim(self) -> None:
        """端到端：弹丸沿曲射弹道落地（无装甲干扰），落点 ≈ 瞄准点 + 散布。"""
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=3)
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0)
        aim = (2000.0, 500.0)
        rec = fc.fire_indirect(gun, aim[0], aim[1])

        impact = None
        for _ in range(80):
            fc.tick(0.5)
            for ev in events.drain():
                if ev.kind == "artillery_impact":
                    impact = ev.data
        assert impact is not None, "弹丸未落地（应在 ~2.5s 内命中）"
        # 容差 = 散布（σ≈2.3m，3σ 内）+ 离散检测过冲（≤ 一个子步 vx·dt≈8m）
        assert impact["x"] == pytest.approx(aim[0], abs=25.0)
        assert impact["y"] == pytest.approx(aim[1], abs=25.0)
        assert abs(impact["z"]) <= 1.0  # 地面高度（无地形 → 0）

    def test_ground_burst_damages_area(self) -> None:
        """落地爆破（SPHERE 提交池）：爆炸半径内目标掉血（45° 高抛避开浅弹道擦碰）。"""
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=11)
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0)
        enemy = _spawn_enemy(armor, boxes, fc, ammo, 2000.0, 0.0)
        # 45° 高抛瞄在敌后 15m：落点距敌 ∈ [8, 22]m（散布 3σ≈7m 内），
        # 且下降段在敌上方高空（45° 斜率=1 → 距落点 15m 时 z≈15m > 车顶）→ 必落地爆
        fc.fire_indirect(gun, 2015.0, 0.0, elevation_deg=45.0)

        impact_seen = False
        for _ in range(200):
            fc.tick(0.5)
            for ev in events.drain():
                if ev.kind == "artillery_impact":
                    impact_seen = True
        assert impact_seen

        # 爆炸半径 25m：落点距敌 ≈ 25m 内 → 线性衰减毁伤 > 0
        hull = int(armor.hull_idx[enemy])
        assert boxes.hp[hull] < 100.0
        # 渲染数据源：爆炸记录可消费
        explosions = fc.damage_pool.drain_explosions()
        assert len(explosions) >= 1
        assert explosions[0]["radius"] == pytest.approx(25.0)

    def test_direct_hit_still_blasts(self) -> None:
        """瞄在目标上：直射命中或落地爆，HE-FRAG 都经提交池造成毁伤。"""
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=5)
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0)
        enemy = _spawn_enemy(armor, boxes, fc, ammo, 1500.0, 0.0)
        fc.fire_indirect(gun, 1500.0, 0.0)

        for _ in range(80):
            fc.tick(0.5)
            events.drain()

        hull = int(armor.hull_idx[enemy])
        assert boxes.hp[hull] < 100.0  # 敌掉血（爆破毁伤）


# ===========================================================================
# 炮兵编队 AI（ArtilleryAI）
# ===========================================================================


class TestArtilleryAI:
    def _make(self, seed: int = 1):
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=seed)
        formations = FormationAIPool(capacity=8)
        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        return armor, boxes, proj, ammo, fc, events, formations, artillery

    def test_friendly_request_fires(self) -> None:
        """友军请求（request_strike）→ 炮兵编队 AI 分派炮位 → 开火。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)

        assert artillery.request_strike((2000.0, 0.0)) is True
        loop = SimulationLoop()
        artillery.tick(loop, 1.0)

        assert proj.active_count >= 1
        assert bool(proj.indirect[proj.active_indices()[0]]) is True
        kinds = {ev.kind for ev in events.drain()}
        assert "artillery_fire" in kinds
        assert artillery.pending_count == 0  # 单发任务已消耗

    def test_rounds_salvo_across_guns(self) -> None:
        """齐射数 rounds=2 + 2 门炮 → 同一 tick 两发（多炮位分摊）。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)
        _spawn_gun(armor, boxes, fc, ammo, 50.0, 0.0, formation_id=fidx)

        artillery.request_strike((2000.0, 0.0), rounds=2)
        loop = SimulationLoop()
        artillery.tick(loop, 1.0)
        assert proj.active_count == 2  # 两门炮各打一发
        assert artillery.pending_count == 0

    def test_player_command_priority(self) -> None:
        """玩家 STRIKE（priority=1）先于友军请求（priority=0）分配炮位。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)

        # 友军先下单，玩家后下单（单炮位只能打一发 → 必须打玩家的）
        artillery.request_strike((3000.0, 0.0))
        cmd = Command(CommandType.STRIKE, "player", {"coords": (1000.0, 100.0)})
        assert artillery.handle_command(cmd) is True

        loop = SimulationLoop()
        artillery.tick(loop, 1.0)
        assert proj.active_count == 1
        pidx = int(proj.active_indices()[0])
        # 初速方向指向玩家目标（1000, 100）（弹丸尚未积分，仍在炮口，查速度）
        assert float(proj.vel[pidx, 0]) > 0.0
        kinds = {ev.kind for ev in events.drain()}
        assert "artillery_fire" in kinds
        # 友军任务仍在队列（未被执行）
        assert artillery.pending_count == 1

    def test_handle_command_target_idx(self) -> None:
        """玩家指令用 target_idx → 自动取活目标坐标。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)
        enemy = _spawn_enemy(armor, boxes, fc, ammo, 1800.0, 300.0)

        cmd = Command(CommandType.STRIKE, "player", {"target_idx": enemy})
        assert artillery.handle_command(cmd) is True
        loop = SimulationLoop()
        artillery.tick(loop, 1.0)
        assert proj.active_count == 1

    def test_comm_gating_blocks_orders(self) -> None:
        """编队 comm < 阈值（指挥链断裂）→ 不收指令，不开火。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        formations.comm[fidx] = 0.1
        _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)

        artillery.request_strike((2000.0, 0.0))
        loop = SimulationLoop()
        artillery.tick(loop, 1.0)
        assert proj.active_count == 0
        assert artillery.pending_count == 1  # 任务保留，等通信恢复

    def test_no_gun_no_fire(self) -> None:
        """无炮位成员 → 任务排队不执行。"""
        armor, boxes, proj, ammo, fc, events, formations, artillery = self._make()
        formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        artillery.request_strike((2000.0, 0.0))
        loop = SimulationLoop()
        artillery.tick(loop, 1.0)
        assert proj.active_count == 0

    def test_non_strike_command_not_consumed(self) -> None:
        """非 STRIKE 指令不被炮兵 AI 消费。"""
        _, _, _, _, _, _, _, artillery = self._make()
        assert artillery.handle_command(Command(CommandType.HOLD, "x")) is False


# ===========================================================================
# loop 集成：command_handler 路由 STRIKE → 炮兵 AI
# ===========================================================================


class TestLoopIntegration:
    def test_strike_command_routed_to_artillery(self) -> None:
        """玩家 STRIKE 入 CommandQueue → loop 消费 → 炮兵 AI 开火 → 落地毁伤。"""
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=2)
        formations = FormationAIPool(capacity=8)
        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        ai = FormationAI(
            armor, formations, fc=fc, boxes=boxes,
            artillery=artillery, events=events,
        )

        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)
        enemy = _spawn_enemy(armor, boxes, fc, ammo, 1500.0, 0.0)

        loop = SimulationLoop(
            state=SimulationState(armor_pool=armor, formation_pool=formations)
        )
        loop.command_handler = artillery.handle_command
        loop.attach("ai", ai.tick)
        loop.attach("combat", lambda l, dt: fc.tick(dt))
        loop.configure_default_rates()

        loop.commands.push(
            Command(CommandType.STRIKE, "player", {"target_idx": enemy})
        )
        for _ in range(30):
            loop.step(1.0)

        assert proj.active_count == 0  # 弹丸已落地（销毁）
        hull = int(armor.hull_idx[enemy])
        assert boxes.hp[hull] < 100.0  # 敌被炮击毁伤

    def test_ai_skips_artillery_in_unit_execution(self) -> None:
        """FormationAI 单位执行跳过炮兵成员（不直射/不机动），由 ArtilleryAI 驱动。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        ai = FormationAI(
            armor, formations, fc=fc, boxes=boxes,
            artillery=artillery, events=events,
        )
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, formation_id=fidx)
        # 敌坦克出现 → FormationAI 会给炮兵成员发 ENGAGE 指令，但执行被跳过
        _spawn_enemy(armor, boxes, fc, ammo, 800.0, 0.0)

        loop = SimulationLoop()
        ai.tick(loop, 1.0)

        # 炮兵成员不直射（无弹丸）且不写路标（不机动）
        assert proj.active_count == 0
        assert armor.wp_count[gun] == 0
