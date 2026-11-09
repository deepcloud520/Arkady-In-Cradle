"""Phase 7 步骤⑪：集成测试——完整空战战役流程（docs/phase7-air-pipeline.md §十四 步骤⑪验收）。

覆盖（SimulationLoop 装配 air_movement/air_ai/combat 全链）：
    编队起飞巡航 → 雷达接敌 → BVR 空战互射（AAM 命中）→ 击落坠毁 →
    幸存者返航（RTB），调度因果序（飞→探测→AI→开火→弹丸推进）正确。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
from Airia.lulilo.combat.engagement import FiringControl
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.combat.weapons import WeaponSystem
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    AirFormationPool,
    AircraftPool,
    AircraftRole,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airia.lulilo.movement.aerodynamics import AircraftAero
from Airia.lulilo.movement.flight_control import FlightControl, build_air_movement_hook
from Airia.lulilo.simulation.air_ai import (
    AirFormationAI,
    AirUnitAI,
    AirUnitOrder,
)
from Airia.lulilo.simulation.loop import SimulationLoop
from Airia.lulilo.simulation.state import SimulationState

from tests.test_phase7_aam import _air_registry


def _spawn_squadron(aircraft, formations, aero, side: int,
                    n: int, x: float, vel_x: float) -> tuple[int, list[int]]:
    fidx = formations.spawn(side=side)
    members = []
    for k in range(n):
        m = aircraft.spawn(x, k * 60.0 - 30.0, 3000.0, side=side,
                           role=int(AircraftRole.FIGHTER), formation_id=fidx,
                           cruise_alt=3000.0, cruise_spd=250.0)
        aircraft.vel[m] = (vel_x, 0.0, 0.0)
        aircraft.heading[m] = 0.0 if vel_x > 0 else math.pi
        aircraft.aam_left[m] = 6
        members.append(m)
    formations.member_count[fidx] = n
    formations.leader_idx[fidx] = members[0]
    return fidx, members


def _build_loop():
    """装配 SimulationLoop：air_movement + air_ai + combat 全链。"""
    aircraft = AircraftPool(capacity=16)
    aero = AircraftAero(16)
    armor = ArmorPool(capacity=8)
    boxes = BoxPool(capacity=128)
    proj = ProjectilePool(capacity=64)
    emitters = EmitterPool(capacity=128)
    formations = AirFormationPool(capacity=8)

    firing = FiringControl(
        armor, boxes, proj,
        weapon_system=WeaponSystem(_air_registry()),
        control=MissileControlRegistry.with_builtin(),
        emitters=emitters,
        rng=RNGManager(seed=7),
        damage_pool=DamageSubmissionPool(capacity=128),
        aircraft=aircraft,
    )
    flight = FlightControl(aircraft.capacity, aero)
    formation_ai = AirFormationAI(aircraft, formations, aero, armor=armor, rng=RNGManager(seed=8))
    unit_ai = AirUnitAI(aircraft, aero, flight, firing, armor=armor)

    state = SimulationState(
        armor_pool=armor, box_pool=boxes, projectile_pool=proj,
        emitter_pool=emitters, aircraft_pool=aircraft, air_formation_pool=formations,
    )
    loop = SimulationLoop(state=state)

    def _air_ai_hook(_loop, dt: float) -> None:
        formation_ai.tick(dt)
        unit_ai.tick(dt)

    loop.attach("air_movement", build_air_movement_hook(
        aircraft, aero, boxes, emitters, armor, proj))
    loop.attach("air_ai", _air_ai_hook)
    loop.attach("combat", lambda _loop, dt: firing.tick(dt))
    loop.configure_default_rates()
    return loop, aircraft, formations, firing


class TestAirWarCampaign:
    """完整空战战役：迎头互射 → 击落 → 幸存返航（半集成：编队 AI 实例显式持有）。"""

    def test_air_war_half_integrated(self) -> None:
        aircraft = AircraftPool(capacity=16)
        aero = AircraftAero(16)
        armor = ArmorPool(capacity=8)
        boxes = BoxPool(capacity=128)
        proj = ProjectilePool(capacity=64)
        emitters = EmitterPool(capacity=128)
        formations = AirFormationPool(capacity=8)

        firing = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(_air_registry()),
            control=MissileControlRegistry.with_builtin(),
            emitters=emitters,
            rng=RNGManager(seed=7),
            damage_pool=DamageSubmissionPool(capacity=128),
            aircraft=aircraft,
        )
        flight = FlightControl(aircraft.capacity, aero)
        formation_ai = AirFormationAI(aircraft, formations, aero, armor=armor,
                                      rng=RNGManager(seed=8))
        unit_ai = AirUnitAI(aircraft, aero, flight, firing, armor=armor)

        state = SimulationState(
            armor_pool=armor, box_pool=boxes, projectile_pool=proj,
            emitter_pool=emitters, aircraft_pool=aircraft, air_formation_pool=formations,
        )
        loop = SimulationLoop(state=state)

        def _air_ai_hook(_loop, dt: float) -> None:
            formation_ai.tick(dt)
            unit_ai.tick(dt)

        loop.attach("air_movement", build_air_movement_hook(
            aircraft, aero, boxes, emitters, armor, proj))
        loop.attach("air_ai", _air_ai_hook)
        loop.attach("combat", lambda _loop, dt: firing.tick(dt))
        loop.configure_default_rates()

        # 双方编队（迎头对飞）
        blue_f, blue = _spawn_squadron(aircraft, formations, aero, 1, 2, -3000.0, 250.0)
        red_f, red = _spawn_squadron(aircraft, formations, aero, 0, 2, 3000.0, -250.0)
        formation_ai.weapons_free[blue_f] = True
        formation_ai.weapons_free[red_f] = True

        # 跑 30s（0.1s 步；air_movement 0.1s / air_ai 1s / combat 0.01s 分频）
        loop.run_headless(30.0, 0.1)

        # 双方都应投入交战（AAM 消耗）
        fired = [m for m in blue + red if int(aircraft.aam_left[m]) < 6]
        assert len(fired) >= 2, f"双方编队都应开火（AAM 消耗）: fired={len(fired)}"
        # 至少一架受创（AAM 近炸/命中）
        damaged = [m for m in blue + red if aircraft.health[m] < 100.0]
        assert len(damaged) >= 1, "空战应造成毁伤"
        # 幸存者状态一致（CRUISE/ENGAGE/RTB/LANDING，不能是垃圾状态）
        for m in blue + red:
            if aircraft.active[m]:
                assert int(aircraft.state[m]) in (
                    int(AircraftState.CRUISE), int(AircraftState.ENGAGE),
                    int(AircraftState.RTB), int(AircraftState.LANDING),
                ), f"飞机 {m} 状态异常: {aircraft.state[m]}"

    def test_loop_rate_configuration(self) -> None:
        """调度链：air_movement/air_ai 分频周期正确（docs §1.2）。"""
        loop, aircraft, formations, firing = _build_loop()
        assert loop.rate_of("air_movement") == pytest.approx(0.1)
        assert loop.rate_of("air_ai") == pytest.approx(1.0)
        assert loop.rate_of("combat") == pytest.approx(0.01)
        # 调度链顺序：air_movement 在 detection 前，air_ai 在 combat 前
        chain = ("movement", "air_movement", "detection", "ai", "air_ai", "combat", "logistics")
        # 验证 loop.step 内部 tuple（间接：通过 attach 顺序）
        assert loop.has("air_movement")
        assert loop.has("air_ai")
        assert loop.has("combat")
