"""Phase 6c 步骤⑥ 测试：反炮闭环 + 编队疏散（docs/phase6-ai-pipeline.md §六.3）。

覆盖：炮口闪光日志（MuzzleFlashLog 登记/敌我过滤/老化）/ 开火暴露（闪光登记）/
反炮检测（敌方闪光 → 自动反炮 STRIKE）/ 来袭弹告警疏散（转移路标 + 事件）/
双阵营闭环（开火→暴露→反炮→转移）。
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
    ArtilleryAI,
    MuzzleFlashLog,
    UnitOrder,
    UnitRole,
)
from Airis.lulilo.simulation.loop import EventBus, SimulationLoop

_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]


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
            boxes.hp[bidx] = 100.0
            boxes.hp_max[bidx] = 100.0
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
            fire_rate_rpm=120.0,
        ),
    )
    ammo.state.set_unit(idx, 40.0, 40.0)
    return idx


# ===========================================================================
# 炮口闪光日志
# ===========================================================================


class TestMuzzleFlashLog:
    def test_add_and_enemy_filter(self) -> None:
        log = MuzzleFlashLog()
        log.add(0, 1.0, 2.0, 3.0)
        log.add(1, 10.0, 20.0)
        assert len(log) == 2
        assert len(log.enemy_flashes(0)) == 1  # 阵营 0 只看到敌（阵营 1）的闪光
        assert log.enemy_flashes(0)[0].side == 1
        assert len(log.enemy_flashes(1)) == 1
        assert log.enemy_flashes(1)[0].x == pytest.approx(1.0)

    def test_expiry(self) -> None:
        log = MuzzleFlashLog(max_age_s=5.0)
        log.add(1, 0.0, 0.0)
        log.tick(4.0)
        assert len(log.enemy_flashes(0)) == 1  # 未过期
        log.tick(2.0)
        assert len(log.enemy_flashes(0)) == 0  # 超过 5s 过期

    def test_ids_monotonic(self) -> None:
        log = MuzzleFlashLog()
        f0 = log.add(1, 0.0, 0.0)
        f1 = log.add(0, 0.0, 0.0)
        assert f0.id != f1.id


# ===========================================================================
# 反炮闭环
# ===========================================================================


class TestCounterBattery:
    def _make(self, seed: int = 1, *, side: int = 0, other_side: int = 1):
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=seed)
        formations = FormationAIPool(capacity=8)
        log = MuzzleFlashLog()
        mine = ArtilleryAI(
            armor, formations, fc=fc, events=events,
            side=side, muzzle_flashes=log,
        )
        other = ArtilleryAI(
            armor, formations, fc=fc, events=events,
            side=other_side, muzzle_flashes=log,
        )
        return armor, boxes, proj, ammo, fc, events, formations, log, mine, other

    def test_firing_registers_muzzle_flash(self) -> None:
        """开火 → 炮口闪光登记（暴露）。"""
        armor, boxes, proj, ammo, fc, events, formations, log, mine, _ = self._make()
        fidx = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx)

        mine.request_strike((5000.0, 0.0))
        loop = SimulationLoop()
        mine.tick(loop, 1.0)

        assert proj.active_count == 1
        assert len(log) == 1
        f = log.enemy_flashes(1)[0]
        assert f.side == 0
        assert f.x == pytest.approx(0.0)
        assert f.y == pytest.approx(0.0)

    def test_enemy_flash_triggers_counter_strike(self) -> None:
        """敌方闪光 → 反炮 STRIKE（自动下单 + counter_battery 事件）。"""
        armor, boxes, proj, ammo, fc, events, formations, log, mine, other = self._make()
        fidx1 = formations.spawn(side=1, role=int(UnitRole.ARTILLERY))
        gun1 = _spawn_gun(armor, boxes, fc, ammo, 3000.0, 0.0, side=1, formation_id=fidx1)

        # 敌方（阵营 0）在 (0,0) 开火暴露
        log.add(0, 0.0, 0.0)
        loop = SimulationLoop()
        other.tick(loop, 1.0)  # other 是阵营 1，检测到阵营 0 闪光 → 反炮并立即开火

        kinds = {ev.kind for ev in events.drain()}
        assert "counter_battery" in kinds
        assert "artillery_fire" in kinds
        # 反炮弹丸瞄准敌方闪光位置 (0,0)
        assert proj.active_count == 1
        pidx = int(proj.active_indices()[0])
        assert float(proj.guid_target_pos[pidx, 0]) == pytest.approx(0.0, abs=25.0)
        assert float(proj.guid_target_pos[pidx, 1]) == pytest.approx(0.0, abs=25.0)

    def test_incoming_hostile_shell_displaces(self) -> None:
        """来袭敌方曲射弹（落点贴近炮位）→ 疏散转移（路标 + 事件）。"""
        armor, boxes, proj, ammo, fc, events, formations, log, mine, other = self._make()
        fidx0 = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun0 = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx0)

        # 敌方（阵营 1）在 (3000,0) 向我炮位 (0,0) 曲射
        fidx1 = formations.spawn(side=1, role=int(UnitRole.ARTILLERY))
        gun1 = _spawn_gun(armor, boxes, fc, ammo, 3000.0, 0.0, side=1, formation_id=fidx1)
        rec = fc.fire_indirect(gun1, 0.0, 0.0)
        assert rec is not None

        loop = SimulationLoop()
        mine.tick(loop, 1.0)  # 阵营 0 检测来袭弹 → 疏散

        assert armor.wp_count[gun0] > 0  # 写入了转移路标
        kinds = {ev.kind for ev in events.drain()}
        assert "artillery_displace" in kinds

    def test_friendly_shell_does_not_displace(self) -> None:
        """我方自己的曲射弹（落点贴近）不触发疏散。"""
        armor, boxes, proj, ammo, fc, events, formations, log, mine, _ = self._make()
        fidx0 = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        gun0 = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx0)
        # 我方另一门炮（阵营 0）向我炮位 (0,0) 曲射（友军，不应疏散）
        gun_ally = _spawn_gun(armor, boxes, fc, ammo, 3000.0, 0.0, side=0, formation_id=fidx0)
        assert fc.fire_indirect(gun_ally, 0.0, 0.0) is not None

        loop = SimulationLoop()
        mine.tick(loop, 1.0)

        assert armor.wp_count[gun0] == 0  # 无疏散
        kinds = {ev.kind for ev in events.drain()}
        assert "artillery_displace" not in kinds


# ===========================================================================
# 双阵营闭环（开火→暴露→反炮→转移）
# ===========================================================================


class TestCounterBatteryLoop:
    def test_full_closed_loop(self) -> None:
        """端到端闭环：A 开火暴露 → B 反炮 → A 检测来袭弹疏散。"""
        armor, boxes, proj, ammo, fc, events = _make_combat(seed=11)
        formations = FormationAIPool(capacity=8)
        log = MuzzleFlashLog()
        side_a = ArtilleryAI(armor, formations, fc=fc, events=events, side=0, muzzle_flashes=log)
        side_b = ArtilleryAI(armor, formations, fc=fc, events=events, side=1, muzzle_flashes=log)

        fa = formations.spawn(side=0, role=int(UnitRole.ARTILLERY))
        fb = formations.spawn(side=1, role=int(UnitRole.ARTILLERY))
        gun_a = _spawn_gun(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fa)
        gun_b = _spawn_gun(armor, boxes, fc, ammo, 3000.0, 0.0, side=1, formation_id=fb)

        loop = SimulationLoop()

        # 1. A 向远处目标开火 → 暴露（闪光）
        side_a.request_strike((6000.0, 0.0))
        side_a.tick(loop, 1.0)
        assert len(log.enemy_flashes(1)) == 1  # A 的闪光对 B 可见

        # 2. B 检测到 A 闪光 → 反炮 STRIKE → B 炮位曲射 A 阵地
        side_b.tick(loop, 1.0)
        kinds = {ev.kind for ev in events.drain()}
        assert "counter_battery" in kinds
        assert "artillery_fire" in kinds  # B 已开火反炮

        # 3. A 检测到 B 来袭弹 → 疏散转移
        side_a.tick(loop, 1.0)
        assert armor.wp_count[gun_a] > 0  # A 炮位转移
        assert armor.wp_count[gun_b] == 0  # B 尚未被威胁
        kinds = {ev.kind for ev in events.drain()}
        assert "artillery_displace" in kinds

    def test_side_parameter_isolates_formations(self) -> None:
        """side 参数隔离编队：side=0 的 AI 不驱动 side=1 的炮兵编队。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        side_a = ArtilleryAI(armor, formations, fc=fc, events=events, side=0)

        fb = formations.spawn(side=1, role=int(UnitRole.ARTILLERY))
        _spawn_gun(armor, boxes, fc, ammo, 3000.0, 0.0, side=1, formation_id=fb)

        side_a.request_strike((5000.0, 0.0))
        loop = SimulationLoop()
        side_a.tick(loop, 1.0)

        assert proj.active_count == 0  # side=0 无编队 → 不开火
        assert side_a.pending_count == 1  # 任务保留（无己方炮位）
