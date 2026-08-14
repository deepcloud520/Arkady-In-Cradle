"""Phase 6c 步骤 ② 测试：编队 AI 指挥链 + 单位 AI 执行（docs/phase6-ai-pipeline.md §三/§四）。

覆盖：评分内核 / 意图 FSM / 贪心目标分配 / 受击面防御 / 单位执行（射击/机动/自保）/
loop 调度集成 / 端到端交战。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
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
    FormationAI,
    FormationIntent,
    FormationSituation,
    UnitOrder,
    score_targets,
)
from Airis.lulilo.simulation.loop import EventBus, SimulationLoop
from Airis.lulilo.simulation.state import SimulationState

# 简化碰撞箱（车体 + 炮塔，够开火链用）
_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]

# 纯 AI 测试共享的 HULL 血量池（docs/part-health-design.md §2.2；
# 装甲经 _spawn_tank 建 HULL box，AI 经注入 boxes 读血）
_AI_BOXES = BoxPool(capacity=512)


def _spawn_tank(
    armor: ArmorPool,
    x: float,
    y: float,
    *,
    team: int = 0,
    formation_id: int = -1,
    health: float = 100.0,
    create_hull: bool = True,
) -> int:
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        team=team, formation_id=formation_id,
    )
    if create_hull:
        bidx = _AI_BOXES.spawn(idx, int(PartType.HULL), 1.5, 3.0, 1.0, 0.0, 0.0, 0.0)
        _AI_BOXES.hp[bidx] = health
        _AI_BOXES.hp_max[bidx] = health
        armor.bind_hull(idx, bidx)
    return idx


def _spawn_combat_tank(
    armor: ArmorPool,
    boxes: BoxPool,
    fc: FiringControl,
    ammo: AmmoManager,
    x: float,
    y: float,
    *,
    team: int = 0,
    formation_id: int = -1,
    health: float = 100.0,
) -> int:
    idx = _spawn_tank(
        armor, x, y, team=team, formation_id=formation_id,
        health=health, create_hull=False,
    )
    for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        if int(part) == int(PartType.HULL):
            # HULL 初始血量（docs/part-health-design.md §2.3）
            boxes.hp[bidx] = health
            boxes.hp_max[bidx] = health
            armor.bind_hull(idx, bidx)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(
        idx,
        UnitCombatConfig(weapon_id="m256_120mm", armor_mm=600.0, crew_skill=0.9),
    )
    ammo.state.set_unit(idx, 40.0, 40.0)
    return idx


def _make_ai(armor: ArmorPool, formations: FormationAIPool, events=None, fc=None) -> FormationAI:
    return FormationAI(armor, formations, fc=fc, boxes=_AI_BOXES, events=events)


def _make_loop(armor: ArmorPool, formations: FormationAIPool) -> SimulationLoop:
    return SimulationLoop(
        state=SimulationState(armor_pool=armor, formation_pool=formations)
    )


# ===========================================================================
# 目标评分内核
# ===========================================================================


class TestScoreTargets:
    def test_distance_threat(self) -> None:
        """距离越近威胁越高。"""
        dist = np.array([100.0, 1000.0], dtype=np.float32)
        speed = np.zeros(2, dtype=np.float32)
        wp = np.zeros(2, dtype=np.float32)
        health = np.ones(2, dtype=np.float32)
        out = np.zeros(2, dtype=np.float32)
        score_targets(dist, speed, wp, health, 1.0, 0.0, 0.0, 0.0, 2000.0, out)
        assert out[0] > out[1]

    def test_low_health_priority(self) -> None:
        """残血目标优先补刀（w_health 上调时）。"""
        dist = np.array([1000.0, 1000.0], dtype=np.float32)
        speed = np.zeros(2, dtype=np.float32)
        wp = np.zeros(2, dtype=np.float32)
        health = np.array([1.0, 0.1], dtype=np.float32)
        out = np.zeros(2, dtype=np.float32)
        score_targets(dist, speed, wp, health, 0.0, 0.0, 0.0, 1.0, 2000.0, out)
        assert out[1] > out[0]


# ===========================================================================
# 编队意图 FSM
# ===========================================================================


class TestFormationIntentFSM:
    def _make(self) -> tuple[ArmorPool, FormationAIPool, FormationAI]:
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        return armor, formations, ai

    def test_maneuver_to_advance_on_advantage(self) -> None:
        """遭遇敌且兵力优势 → ADVANCE。"""
        armor, formations, ai = self._make()
        fidx = formations.spawn(team=0, role=0, target_zone=(100.0, 100.0, 0.0))
        for i in range(3):  # 友 3
            _spawn_tank(armor, float(i * 10), 0.0, team=0, formation_id=fidx)
        _spawn_tank(armor, 500.0, 0.0, team=1)  # 敌 2
        _spawn_tank(armor, 510.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert int(formations.intent[fidx]) == int(FormationIntent.ADVANCE)

    def test_maneuver_to_defend_on_disadvantage(self) -> None:
        """遭遇敌且兵力劣势 → DEFEND。"""
        armor, formations, ai = self._make()
        fidx = formations.spawn(team=0, role=0)
        _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx)  # 友 2
        _spawn_tank(armor, 10.0, 0.0, team=0, formation_id=fidx)
        for i in range(3):  # 敌 3
            _spawn_tank(armor, 500.0 + i * 10, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert int(formations.intent[fidx]) == int(FormationIntent.DEFEND)

    def test_advance_to_retreat_on_loss(self) -> None:
        """进攻中损失超阈值 → RETREAT。"""
        armor, formations, ai = self._make()
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx, health=30.0)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert int(formations.intent[fidx]) == int(FormationIntent.RETREAT)

    def test_hold_command_overrides(self) -> None:
        """编队级 HOLD 命令 → 意图 HOLD。"""
        armor, formations, ai = self._make()
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        formations.order[fidx] = int(UnitOrder.HOLD)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert int(formations.intent[fidx]) == int(FormationIntent.HOLD)


# ===========================================================================
# 指挥链：贪心目标分配
# ===========================================================================


class TestCommandChain:
    def test_assign_targets(self) -> None:
        """编队 AI 统一分配目标，成员按 order/target_idx 执行。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        m1 = _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx)
        m2 = _spawn_tank(armor, 10.0, 0.0, team=0, formation_id=fidx)
        e1 = _spawn_tank(armor, 500.0, 0.0, team=1)
        e2 = _spawn_tank(armor, 510.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)

        assert armor.target_idx[m1] in (e1, e2)
        assert armor.target_idx[m2] in (e1, e2)
        # 两个目标都至少分配到一个成员
        assigned = {int(armor.target_idx[m1]), int(armor.target_idx[m2])}
        assert {e1, e2}.issubset(assigned)
        # ADVANCE → 成员执行指令 ENGAGE
        assert int(armor.order[m1]) == int(UnitOrder.ENGAGE)

    def test_concurrent_fire_cap(self) -> None:
        """每目标并发开火上限（防过度杀伤）。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        members = [_spawn_tank(armor, float(i * 10), 0.0, team=0, formation_id=fidx) for i in range(4)]
        _spawn_tank(armor, 500.0, 0.0, team=1)  # 敌 1
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assigned = sum(1 for m in members if int(armor.target_idx[m]) >= 0)
        assert assigned == FormationAI.MAX_CONCURRENT_FIRE  # 3

    def test_hit_defense(self) -> None:
        """进攻编队命令受击严重成员转 DEFEND（受击面防御）。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        healthy = _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx)
        hurt = _spawn_tank(armor, 10.0, 0.0, team=0, formation_id=fidx, health=40.0)
        _spawn_tank(armor, 500.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert int(armor.order[hurt]) == int(UnitOrder.DEFEND)
        assert int(armor.order[healthy]) == int(UnitOrder.ENGAGE)


# ===========================================================================
# 单位 AI 执行
# ===========================================================================


class TestUnitExecution:
    def _make_combat(self) -> tuple[ArmorPool, BoxPool, ProjectilePool, AmmoManager, FiringControl, FormationAIPool, FormationAI, EventBus]:
        armor = ArmorPool(capacity=32)
        boxes = BoxPool(capacity=256)
        projectiles = ProjectilePool(capacity=512)
        ammo = AmmoManager(capacity=32)
        events = EventBus()
        fc = FiringControl(
            armor, boxes, projectiles,
            ammo=ammo,
            events=events,
            rng=RNGManager(seed=7),
        )
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations, events=events, fc=fc)
        return armor, boxes, projectiles, ammo, fc, formations, ai, events

    def test_engage_fires(self) -> None:
        """编队下达进攻 → 成员对分配目标自主开火。"""
        armor, boxes, proj, ammo, fc, formations, ai, events = self._make_combat()
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        m = _spawn_combat_tank(armor, boxes, fc, ammo, 0.0, 0.0, team=0, formation_id=fidx)
        e = _spawn_combat_tank(armor, boxes, fc, ammo, 100.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)

        # 有弹丸生成 + ai_fire 事件
        assert proj.active_count >= 1
        kinds = {ev.kind for ev in events.drain()}
        assert "ai_fire" in kinds
        assert int(armor.target_idx[m]) == e

    def test_engage_moves_when_out_of_range(self) -> None:
        """目标超射程 → 机动靠近（写 wp）。"""
        armor, boxes, proj, ammo, fc, formations, ai, events = self._make_combat()
        fidx = formations.spawn(team=0, role=0)
        formations.intent[fidx] = int(FormationIntent.ADVANCE)
        m = _spawn_combat_tank(armor, boxes, fc, ammo, 0.0, 0.0, team=0, formation_id=fidx)
        _spawn_combat_tank(armor, boxes, fc, ammo, 5000.0, 0.0, team=1)  # 超射程
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert armor.wp_count[m] > 0  # 已下达移动路标

    def test_maneuver_moves(self) -> None:
        """MANEUVER 指令 → 向编队 target_zone 写 wp。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0, target_zone=(300.0, 400.0, 0.0))
        m = _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx)
        armor.order[m] = int(UnitOrder.MANEUVER)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert armor.wp_count[m] > 0
        assert armor.wp[m, 0, 0] == pytest.approx(300.0)

    def test_self_preserve_on_low_comm(self) -> None:
        """通信断裂（comm 低）→ 自主自保：对最近威胁还击。"""
        armor, boxes, proj, ammo, fc, formations, ai, events = self._make_combat()
        fidx = formations.spawn(team=0, role=0)
        m = _spawn_combat_tank(armor, boxes, fc, ammo, 0.0, 0.0, team=0, formation_id=fidx)
        armor.comm[m] = 0.1  # 低于 COMM_THRESHOLD
        _spawn_combat_tank(armor, boxes, fc, ammo, 100.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert proj.active_count >= 1  # 还击开火

    def test_self_preserve_retreat_low_health(self) -> None:
        """自保：血量过低 → 撤退（写 wp）。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0)
        m = _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx, health=20.0)
        armor.comm[m] = 0.1
        _spawn_tank(armor, 100.0, 0.0, team=1)
        loop = _make_loop(armor, formations)
        ai.tick(loop, 1.0)
        assert armor.wp_count[m] > 0  # 撤退机动


# ===========================================================================
# loop 调度集成
# ===========================================================================


class TestLoopIntegration:
    def test_ai_hook_in_schedule(self) -> None:
        """configure_default_rates 后 ai 每 1s 累积触发。"""
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations)
        fidx = formations.spawn(team=0, role=0)
        _spawn_tank(armor, 0.0, 0.0, team=0, formation_id=fidx)
        _spawn_tank(armor, 100.0, 0.0, team=1)

        loop = _make_loop(armor, formations)
        loop.attach("ai", ai.tick)
        loop.configure_default_rates()
        # 两次 0.5s 累积 1s → 触发 AI 决策
        loop.step(0.5)
        loop.step(0.5)
        assert int(formations.intent[fidx]) in (
            int(FormationIntent.ADVANCE),
            int(FormationIntent.DEFEND),
        )

    def test_end_to_end_engagement(self) -> None:
        """完整闭环：AI 开火 → combat 推进 → 敌掉血。"""
        armor = ArmorPool(capacity=32)
        boxes = BoxPool(capacity=256)
        projectiles = ProjectilePool(capacity=512)
        ammo = AmmoManager(capacity=32)
        events = EventBus()
        fc = FiringControl(
            armor, boxes, projectiles,
            ammo=ammo,
            events=events,
            rng=RNGManager(seed=3),
        )
        formations = FormationAIPool(capacity=8)
        ai = _make_ai(armor, formations, events=events, fc=fc)
        fidx = formations.spawn(team=0, role=0)
        _spawn_combat_tank(armor, boxes, fc, ammo, 0.0, 0.0, team=0, formation_id=fidx)
        e = _spawn_combat_tank(armor, boxes, fc, ammo, 100.0, 0.0, team=1)

        def combat_hook(loop: SimulationLoop, dt: float) -> None:
            fc.tick(dt)

        loop = _make_loop(armor, formations)
        loop.attach("ai", ai.tick)
        loop.attach("combat", combat_hook)
        loop.configure_default_rates()
        for _ in range(20):
            loop.step(1.0)

        # 敌至少被命中过（血量下降）或仍有弹丸飞行
        assert boxes.hp[armor.hull_idx[e]] < 100.0 or projectiles.active_count > 0
