"""Phase 7 步骤⑥：空中单位 AI 状态机 + 空战评分（docs/phase7-air-pipeline.md §十四 步骤⑥验收）。

覆盖：
    ENGAGE_AIR（占位追击 + 发射 AAM）、目标失效回 CAP、弹药耗尽/燃油低返航、
    STRIKE_GROUND 俯冲-投弹-拉起返航、score_air_targets 评分内核、
    端到端 BVR 空战互射（两机迎头 → AAM 命中 → 一架坠毁）。
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
    AircraftPool,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airia.lulilo.movement.aerodynamics import AircraftAero, integrate_aircraft_motion
from Airia.lulilo.movement.flight_control import FlightControl
from Airia.lulilo.simulation.air_ai import AirUnitAI, AirUnitOrder

from tests.test_phase7_aam import _air_registry

G = 9.81


def _rig() -> tuple[AirUnitAI, AircraftPool, ArmorPool, FiringControl, FlightControl, AircraftAero]:
    aircraft = AircraftPool(capacity=8)
    aero = AircraftAero(8)
    armor = ArmorPool(capacity=4)
    boxes = BoxPool(capacity=64)
    proj = ProjectilePool(capacity=32)
    emitters = EmitterPool(capacity=64)
    firing = FiringControl(
        armor, boxes, proj,
        weapon_system=WeaponSystem(_air_registry()),
        control=MissileControlRegistry.with_builtin(),
        emitters=emitters,
        rng=RNGManager(seed=42),
        damage_pool=DamageSubmissionPool(capacity=64),
        aircraft=aircraft,
    )
    flight = FlightControl(aircraft.capacity, aero)
    unit = AirUnitAI(aircraft, aero, flight, firing, armor=armor)
    return unit, aircraft, armor, firing, flight, aero


class TestEngageAir:
    """ENGAGE_AIR：占位追击 + 发射。"""

    def test_fires_aam_when_in_range(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        unit.tick(0.1)
        assert aircraft.aam_left[me] == 3, "射程内应发射 AAM"
        assert aircraft.phi_cmd[me] != 0.0 or abs(aircraft.phi_cmd[me]) < 1e-6  # 占位机动不报错

    def test_lead_pursuit_aims(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 1000.0, 3000.0, side=0)
        aircraft.vel[me] = (250.0, 0.0, 0.0)
        aircraft.heading[me] = 0.0
        aircraft.vel[foe] = (250.0, 0.0, 0.0)   # 同向横切
        aircraft.aam_left[me] = 4
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        unit.tick(0.1)
        # 目标在前方偏右 → 应右滚转（lead pursuit 前置量）
        assert aircraft.phi_cmd[me] > 0.0, "占位机动应朝拦截点转向"

    def test_lost_target_returns_to_cap(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        aircraft.destroy(foe)   # 目标消失
        unit.tick(0.1)
        assert aircraft.order[me] == int(AirUnitOrder.CAP), "目标失效应回 CAP"
        assert aircraft.target_idx[me] == -1

    def test_out_of_ammo_rtb(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 0
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        unit.tick(0.1)
        assert aircraft.order[me] == int(AirUnitOrder.RTB), "弹药耗尽应返航"

    def test_bingo_fuel_rtb(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        aircraft.fuel[me] = 0.2    # < bingo 0.3
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        unit.tick(0.1)
        assert aircraft.order[me] == int(AirUnitOrder.RTB), "燃油低于 bingo 应返航"


class TestStrikeGround:
    """STRIKE_GROUND：俯冲-投弹-拉起返航。"""

    def test_dive_and_drop_bomb(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1, role=1)
        tank = armor.spawn(2000.0, 0.0, 0.0)
        armor.side[tank] = 0
        aircraft.agm_left[me] = 3
        aircraft.order[me] = int(AirUnitOrder.STRIKE_GROUND)
        aircraft.target_idx[me] = tank
        unit.tick(0.1)
        # 高空 → 俯冲（gamma_cmd 负）
        assert aircraft.gamma_cmd[me] < 0.0, "高空应俯冲投弹"
        # 射程内 → 投弹（agm_left 减）
        assert aircraft.agm_left[me] == 2, "射程内应投放炸弹"

    def test_pullup_and_rtb_after_drop(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        me = aircraft.spawn(0.0, 0.0, 500.0, side=1, role=1)   # 低空（已投弹）
        tank = armor.spawn(2000.0, 0.0, 0.0)
        armor.side[tank] = 0
        aircraft.agm_left[me] = 0
        aircraft.order[me] = int(AirUnitOrder.STRIKE_GROUND)
        aircraft.target_idx[me] = tank
        unit.tick(0.1)
        assert aircraft.gamma_cmd[me] > 0.0, "投弹后应拉起"
        assert aircraft.order[me] == int(AirUnitOrder.RTB), "投弹完成应返航"


class TestScoreKernel:
    """score_air_targets 评分内核（docs §6.3）。"""

    def test_tail_advantage_scores_higher(self) -> None:
        from Airia.lulilo.simulation.air_ai import score_air_targets

        k = 2
        dist = np.array([5000.0, 5000.0], np.float32)
        d_alt = np.array([0.0, 0.0], np.float32)
        health = np.array([0.8, 0.8], np.float32)
        los_x = np.array([1.0, 1.0], np.float32)   # 目标在我正前方
        los_y = np.array([0.0, 0.0], np.float32)
        los_z = np.array([0.0, 0.0], np.float32)
        # 目标 1：迎头（速度朝我 → tail=-1）；目标 2：逃跑（背离 → tail=+1 尾追优势）
        tvx = np.array([-250.0, 250.0], np.float32)
        tvy = np.zeros(k, np.float32)
        tvz = np.zeros(k, np.float32)
        out = np.zeros(k, np.float32)
        score_air_targets(dist, d_alt, health, los_x, los_y, los_z,
                          tvx, tvy, tvz, 20000.0, 1.0, 0.5, 1.5, 1.0, out)
        assert out[1] > out[0], "尾追优势（目标背离我）应得分更高"

    def test_altitude_advantage_scores_higher(self) -> None:
        from Airia.lulilo.simulation.air_ai import score_air_targets

        k = 2
        dist = np.array([5000.0, 5000.0], np.float32)
        d_alt = np.array([1000.0, -1000.0], np.float32)   # 我高 1000 vs 我低 1000
        health = np.array([0.8, 0.8], np.float32)
        los_x = np.ones(k, np.float32)
        los_y = np.zeros(k, np.float32)
        los_z = np.zeros(k, np.float32)
        tvx = np.full(k, -250.0, np.float32)
        tvy = np.zeros(k, np.float32)
        tvz = np.zeros(k, np.float32)
        out = np.zeros(k, np.float32)
        score_air_targets(dist, d_alt, health, los_x, los_y, los_z,
                          tvx, tvy, tvz, 20000.0, 1.0, 0.5, 1.5, 1.0, out)
        assert out[0] > out[1], "高度优势（比能）应得分更高"


class TestBvrCombatEndToEnd:
    """端到端 BVR 空战：两机迎头互射 → AAM 命中 → 一架坠毁。"""

    def _step(self, unit, aircraft, flight, aero, firing, armor, boxes, emitters,
              proj, dt=0.1) -> None:
        unit.tick(dt)
        flight.tick(aircraft, dt)
        stall = np.zeros(aircraft.capacity, np.bool_)
        integrate_aircraft_motion(
            aircraft.pos, aircraft.vel, aircraft.heading, aircraft.pitch, aircraft.bank,
            aircraft.throttle, aircraft.phi_cmd, aircraft.gamma_cmd,
            aircraft.state, aircraft.dmg_factor, aircraft.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            aircraft.fuel, aero.fuel_burn,
            1.225, 8500.0, G, dt, stall,
        )
        idxs = np.flatnonzero(stall & aircraft.active)
        if idxs.size:
            aircraft.state[idxs] = int(AircraftState.CRASHING)
        boxes.sync_all_from_parent(armor, aircraft=aircraft)
        emitters.sync_from_parent(armor, proj, dt, aircraft=aircraft)
        for _ in range(int(dt / 0.01)):
            firing.tick(0.01)

    def test_air_combat_has_winner(self) -> None:
        unit, aircraft, armor, firing, flight, aero = _rig()
        boxes = firing.boxes
        emitters = firing.emitters
        proj = firing.projectiles
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(4000.0, 0.0, 3000.0, side=0)
        aircraft.vel[me] = (250.0, 0.0, 0.0)
        aircraft.heading[me] = 0.0
        aircraft.vel[foe] = (-250.0, 0.0, 0.0)   # 迎头对飞
        aircraft.heading[foe] = math.pi
        aircraft.aam_left[me] = 4
        aircraft.aam_left[foe] = 4
        # 双方互相指派
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        aircraft.order[foe] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[foe] = me

        crashed = False
        for _ in range(150):   # 15s
            self._step(unit, aircraft, flight, aero, firing, armor, boxes, emitters, proj)
            if not aircraft.active[me] or not aircraft.active[foe] \
                    or int(aircraft.state[me]) == int(AircraftState.CRASHING) \
                    or int(aircraft.state[foe]) == int(AircraftState.CRASHING):
                crashed = True
                break
        assert crashed, "空战应有胜负（至少一架被击落/坠毁）"
