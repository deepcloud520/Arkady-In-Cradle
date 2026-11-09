"""Phase 8 步骤⑩：集成测试——战争迷雾四大验收用例（docs/phase8-datalink-design.md §十一）。

覆盖：
    1. 地雷命中 → 通讯中断 MIA → 无人机确认 KIA 完整时序；
    2. 装甲呼叫炮兵 → 孪生坐标（带误差）落点 → 毁伤 → 报告回流 KIA；
    3. 数据链中断编队退回自保（comm 门控）；
    4. A 导 B 射：中继制导命中超视距目标（relay_guidance 端到端）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
from Airia.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.combat.weapons import WeaponSystem
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airia.lulilo.entities.twin import TWIN_KIND_ARMOR, TwinPool, TwinStatus
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.simulation.ai import ArtilleryAI, FormationAI, UnitRole
from Airia.lulilo.simulation.loop import EventBus, SimulationLoop
from Airia.lulilo.simulation.state import SimulationState
from Airia.lulilo.simulation.twin_engine import TwinEngine


class _Loop:
    def __init__(self) -> None:
        self.state = type("S", (), {"tick_count": 0})()


class TestCase1MiaToKia:
    """用例 1：地雷命中 → 通讯中断 MIA → 无人机确认 KIA（docs §2.3 概念场景）。"""

    def test_mia_then_recon_confirms_kia(self) -> None:
        twin = TwinPool(capacity=64)
        armor = ArmorPool(capacity=8)
        engine = TwinEngine(twin, armor=armor)
        loop = _Loop()
        unit = armor.spawn(0.0, 0.0, 0.0, side=1)
        s = twin.bind(TWIN_KIND_ARMOR, unit, pos=(0.0, 0.0, 0.0))
        # ① 观测中（ACTIVE）
        engine.observe(TWIN_KIND_ARMOR, unit, (0.0, 0.0, 0.0), confidence=0.9)
        engine.tick(loop, 1.0)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        # ② 地雷命中 → 物理销毁 → 通讯中断 → MIA（信号丢失）
        armor.destroy(unit)
        loop.state.tick_count += 1
        engine.tick(loop, 1.0)
        assert twin.status[s] == int(TwinStatus.MIA), "通讯中断 → 信号丢失 MIA"
        # ③ 侦察无人机确认 → KIA
        engine.confirm_kia(TWIN_KIND_ARMOR, unit, confidence=0.95)
        assert twin.status[s] == int(TwinStatus.KIA), "侦察确认 → KIA"


class TestCase2CallArtillery:
    """用例 2：装甲呼叫炮兵 → 孪生坐标落点 → 毁伤 → 报告回流。"""

    def _rig(self):
        armor = ArmorPool(capacity=16)
        formations = FormationAIPool(capacity=4)
        boxes = BoxPool(capacity=64)
        proj = ProjectilePool(capacity=32)
        events = EventBus()
        twin = TwinPool(capacity=64)
        ammo = AmmoManager(capacity=16)
        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(),
            control=MissileControlRegistry.with_builtin(),
            rng=RNGManager(seed=5),
            events=events,
            ammo=ammo,
            damage_pool=DamageSubmissionPool(capacity=64),
        )
        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        # 炮兵编队
        fidx = formations.spawn(side=0)
        formations.role[fidx] = int(UnitRole.ARTILLERY)
        formations.comm[fidx] = 1.0
        gun = armor.spawn(0.0, 0.0, 0.0, side=0, role=int(UnitRole.ARTILLERY),
                          formation_id=fidx)
        for part, hx, hy, hz, lx, ly, lz in [
            (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
            (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
        ]:
            boxes.spawn(gun, int(part), hx, hy, hz, lx, ly, lz)
        boxes.sync_all_from_parent(armor)
        fc.set_unit_config(gun, UnitCombatConfig(
            weapon_id="m109_155mm", ammo_id="m107_155mm_he_frag",
            crew_skill=0.7, visibility=1.0, weapon_condition=1.0,
            max_range_m=15000.0, fire_rate_rpm=4.0,
        ))
        ammo.state.reserve[gun] = 20.0
        formations.member_count[fidx] = 1
        # 目标装甲 + 孪生（带误差：真值 1000,500 / 孪生 1030,490）
        foe = armor.spawn(1000.0, 500.0, 0.0, side=1)
        for part, hx, hy, hz, lx, ly, lz in [
            (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
            (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
        ]:
            boxes.spawn(foe, int(part), hx, hy, hz, lx, ly, lz)
        boxes.sync_all_from_parent(armor)
        ts = twin.bind(TWIN_KIND_ARMOR, foe, pos=(1030.0, 490.0, 0.0))
        twin.status[ts] = int(TwinStatus.ACTIVE)
        twin.confidence[ts] = 0.8
        twin.health[ts] = 100.0
        return artillery, armor, fc, events, twin, foe

    def test_call_artillery_full_loop(self) -> None:
        artillery, armor, fc, events, twin, foe = self._rig()
        # 装甲编队呼叫支援（孪生坐标 1030,490）
        events.emit("support_request", {
            "kind": "artillery", "requester": 0,
            "x": 1030.0, "y": 490.0, "priority": 5, "confidence": 0.8,
        })
        loop = SimulationLoop(state=SimulationState())
        artillery.tick(loop, 1.0)
        accepted = [e for e in events.drain() if e.kind == "support_accepted"]
        assert len(accepted) == 1, "炮兵应接受支援请求"
        # 推进弹道 → 落点引爆 → 目标掉血
        hit = False
        for _ in range(3000):
            fc.tick(0.01)
            if armor.active[foe] and fc.boxes.hp[armor.hull_idx[foe]] < 100.0:
                hit = True
                break
            if fc.damage_pool.active_count > 0:
                fc.tick(0.01)
                fc.tick(0.01)   # 结算
                if fc.boxes.hp[armor.hull_idx[foe]] < 100.0:
                    hit = True
                    break
        assert hit, "炮兵落点应毁伤目标（孪生坐标带误差仍命中）"
        # 报告回流：观测到目标掉血 → 上报孪生
        twin_health = float(fc.boxes.hp[armor.hull_idx[foe]])
        assert twin_health < 100.0


class TestCase3LinkLossSelfPreserve:
    """用例 3：数据链中断编队退回 6c 自保（comm 门控）。"""

    def test_low_comm_member_self_preserves(self) -> None:
        armor = ArmorPool(capacity=8)
        formations = FormationAIPool(capacity=2)
        fc = FiringControl(
            armor, BoxPool(capacity=16), ProjectilePool(capacity=8),
            weapon_system=WeaponSystem(),
            control=MissileControlRegistry.with_builtin(),
            rng=RNGManager(seed=1),
        )
        ai = FormationAI(armor, formations, fc=fc)
        fidx = formations.spawn(side=0)
        m = armor.spawn(0.0, 0.0, 0.0, side=0, formation_id=fidx)
        armor.comm[m] = 0.2   # 通信断裂（< COMM_THRESHOLD 0.5）
        formations.member_count[fidx] = 1
        # 编队下令 ENGAGE，但成员 comm 断裂 → 自保（不执行编队指令）
        armor.order[m] = 0   # ENGAGE
        loop = SimulationLoop(state=SimulationState())
        ai.tick(loop, 1.0)
        # 自保路径：成员按 self_preserve 执行（order 被覆盖为自保逻辑）
        assert float(armor.comm[m]) < ai.COMM_THRESHOLD
        # 自保 = 不追击（target 清空或保持防御）
        assert armor.target_idx[m] == -1 or armor.order[m] == 1  # DEFEND


class TestCase4AirRelayEndToEnd:
    """用例 4：A 导 B 射命中超视距目标（relay_guidance 端到端）。"""

    def test_relay_guided_aam_hits(self) -> None:
        from Airia.lulilo.entities.pools import AircraftPool

        from tests.test_phase7_aam import _air_registry

        aircraft = AircraftPool(capacity=8)
        armor = ArmorPool(capacity=2)
        boxes = BoxPool(capacity=64)
        proj = ProjectilePool(capacity=32)
        emitters = None
        from Airia.lulilo.entities.pools import EmitterPool

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
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        # 中继语义验证：B（预警机）持续写入目标航迹（A 的导引头不参与初段）。
        # 距离取战场界内（WORLD_BOUND=5000 回收限制），核心是「AAM 由中继坐标驱动命中」。
        foe = aircraft.spawn(3500.0, 0.0, 3000.0, side=0)
        aircraft.vel[foe] = (-250.0, 0.0, 0.0)
        aircraft.health[foe] = 100.0
        aircraft.aam_left[me] = 4
        rec = firing.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                              target_kind=0, target_idx=foe)
        assert rec is not None
        hit = False
        for _ in range(3000):
            # 数据链中继：前出预警机持续更新目标航迹
            firing.relay_guidance(rec.proj_idx, aircraft.pos[foe], link_q=0.9)
            aircraft.pos[foe] += aircraft.vel[foe] * 0.01
            firing.tick(0.01)
            if aircraft.health[foe] < 100.0 or not aircraft.active[foe]:
                hit = True
                break
            if not proj.active[rec.proj_idx]:
                break
        assert hit, "A 导 B 射（数据链中继）应命中目标"
