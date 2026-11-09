"""Phase 6c 步骤⑩ 测试：场景装配器 + 完整战役流程集成（docs/phase6-ai-pipeline.md §十二）。

覆盖：ScenarioConfig → build_scenario 装配（pools/AI/炮兵/反炮/通信簇/loop）/
坦克遭遇战端到端（开火命中毁伤）/ 炮兵反炮闭环端到端（玩家 STRIKE → 开火 →
反炮 → 疏散）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.simulation.ai import UnitRole
from Airia.lulilo.simulation.commands import Command, CommandType
from Airia.lulilo.simulation.scenario import (
    FormationSpec,
    ScenarioConfig,
    UnitSpec,
    build_scenario,
)


def _tank_clash_config(seed: int = 3) -> ScenarioConfig:
    """友 4 坦克 vs 敌 4 坦克，近距离遭遇（1000m，均在射程内）。"""
    return ScenarioConfig(
        name="tank_clash",
        seed=seed,
        formations=[
            FormationSpec(side=0, role=0),
            FormationSpec(side=1, role=0),
        ],
        units=[
            *[UnitSpec(side=0, x=-500.0, y=float(i) * 30.0, formation=0) for i in range(4)],
            *[UnitSpec(side=1, x=500.0, y=float(i) * 30.0, formation=1) for i in range(4)],
        ],
    )


def _artillery_duel_config() -> ScenarioConfig:
    """友 2 炮兵 vs 敌 2 炮兵（相距 3000m，均在曲射射程内）。"""
    return ScenarioConfig(
        name="artillery_duel",
        seed=7,
        formations=[
            FormationSpec(side=0, role=int(UnitRole.ARTILLERY)),
            FormationSpec(side=1, role=int(UnitRole.ARTILLERY)),
        ],
        units=[
            UnitSpec(side=0, role=int(UnitRole.ARTILLERY), x=0.0, y=0.0, formation=0, armor_mm=60.0),
            UnitSpec(side=0, role=int(UnitRole.ARTILLERY), x=0.0, y=50.0, formation=0, armor_mm=60.0),
            UnitSpec(side=1, role=int(UnitRole.ARTILLERY), x=3000.0, y=0.0, formation=1, armor_mm=60.0),
            UnitSpec(side=1, role=int(UnitRole.ARTILLERY), x=3000.0, y=50.0, formation=1, armor_mm=60.0),
        ],
    )


# ===========================================================================
# 装配
# ===========================================================================


class TestScenarioAssembly:
    def test_build_wires_all_subsystems(self) -> None:
        rt = build_scenario(_tank_clash_config())
        assert rt.armor.active_count == 8
        assert rt.formations.active_count == 2
        assert len(rt.artilleries) == 0           # 纯坦克场景无炮兵
        assert rt.loop.has("ai") and rt.loop.has("combat") and rt.loop.has("movement")
        assert rt.loop.command_handler is None    # 无炮兵 → 无 STRIKE 路由

    def test_build_artillery_creates_two_ai(self) -> None:
        rt = build_scenario(_artillery_duel_config())
        assert len(rt.artilleries) == 2           # 双方各一个炮兵 AI
        assert rt.loop.command_handler is not None


# ===========================================================================
# 坦克遭遇战端到端
# ===========================================================================


class TestTankClash:
    def test_engage_and_damage(self) -> None:
        rt = build_scenario(_tank_clash_config(seed=3))
        rt.loop.run_headless(30.0, 1.0)

        kinds = {ev.kind for ev in rt.events.drain()}
        assert "fire" in kinds                    # 双方开火

        # 至少一方掉血（命中毁伤经提交池/开火链结算）
        damaged = sum(
            1 for i in rt.armor.active_indices()
            if rt.boxes.hp[int(rt.armor.hull_idx[int(i)])] < 100.0
        )
        assert damaged >= 1


# ===========================================================================
# 炮兵反炮闭环端到端
# ===========================================================================


class TestArtilleryDuel:
    def test_counter_battery_closed_loop(self) -> None:
        rt = build_scenario(_artillery_duel_config())
        # 玩家 STRIKE → 友方炮兵开火（暴露）
        rt.loop.commands.push(
            Command(CommandType.STRIKE, "player", {"coords": (1500.0, 0.0)})
        )
        for _ in range(10):
            rt.loop.step(1.0)

        kinds = {ev.kind for ev in rt.events.drain()}
        assert "artillery_fire" in kinds          # 友方开火
        assert "counter_battery" in kinds         # 敌方反炮
        assert "artillery_displace" in kinds      # 友方疏散转移

        # 友方炮位已写转移路标
        friend_guns = [
            int(i) for i in rt.armor.active_indices()
            if int(rt.armor.side[i]) == 0 and int(rt.armor.role[i]) == int(UnitRole.ARTILLERY)
        ]
        assert any(rt.armor.wp_count[g] > 0 for g in friend_guns)
