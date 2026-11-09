"""Phase 8 步骤⑤⑥：装甲 AI（孪生接触源 + 支援请求链 + 评分折扣）+ 炮兵自主接单
（docs/phase8-datalink-design.md §十四 步骤⑤⑥验收）。

覆盖：
    TwinContactProvider（位置/血量/置信度来自孪生）、KIA/己方过滤、
    score_targets confidence 折扣（迷雾影响决策）、
    支援请求链（目标超射程 → support_request 事件）、
    目标分配低置信过滤（看不清不打）、
    炮兵自主接单闭环（support_request → fire_indirect 孪生坐标落点）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airia.lulilo.entities.twin import (
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.simulation.ai import (
    ArtilleryAI,
    Contact,
    FormationAI,
    TwinContactProvider,
    UnitRole,
    score_targets,
)
from Airia.lulilo.simulation.loop import EventBus, SimulationLoop
from Airia.lulilo.simulation.state import SimulationState


def _rig_formation_ai() -> tuple[FormationAI, ArmorPool, FormationAIPool, TwinPool, EventBus]:
    armor = ArmorPool(capacity=16)
    formations = FormationAIPool(capacity=4)
    twin = TwinPool(capacity=64)
    events = EventBus()
    provider = TwinContactProvider(twin, armor)
    ai = FormationAI(armor, formations, contact_provider=provider, events=events)
    return ai, armor, formations, twin, events


def _spawn_squad(armor, formations, x: float, side: int, n: int = 3):
    fidx = formations.spawn(side=side)
    members = []
    for k in range(n):
        m = armor.spawn(x + k * 10.0, 0.0, 0.0, side=side, formation_id=fidx)
        armor.comm[m] = 1.0
        members.append(m)
    formations.member_count[fidx] = n
    formations.comm[fidx] = 1.0
    return fidx, members


class TestTwinContactProvider:
    """孪生接触源（docs §5.1）。"""

    def test_contacts_from_twin(self) -> None:
        ai, armor, formations, twin, events = _rig_formation_ai()
        _, blue = _spawn_squad(armor, formations, 0.0, 0)   # 己方
        foe = armor.spawn(500.0, 300.0, 0.0, side=1)
        armor.side[foe] = 1
        s = twin.bind(TWIN_KIND_ARMOR, foe, pos=(480.0, 290.0, 0.0))
        twin.status[s] = int(TwinStatus.ACTIVE)
        twin.confidence[s] = 0.7
        twin.health[s] = 55.0
        provider = TwinContactProvider(twin, armor)
        contacts = provider(0)   # 蓝方视角
        assert len(contacts) == 1
        c = contacts[0]
        assert c.idx == foe
        assert c.x == pytest.approx(480.0), "位置来自孪生（带误差）"
        assert c.health == pytest.approx(55.0), "血量来自孪生（滞后）"
        assert c.confidence == pytest.approx(0.7)
        assert c.twin_slot == s

    def test_friendly_and_kia_filtered(self) -> None:
        ai, armor, formations, twin, events = _rig_formation_ai()
        _, blue = _spawn_squad(armor, formations, 0.0, 0)
        buddy = armor.spawn(100.0, 0.0, 0.0, side=0)   # 己方
        s1 = twin.bind(TWIN_KIND_ARMOR, buddy)
        twin.status[s1] = int(TwinStatus.ACTIVE)
        twin.confidence[s1] = 0.9
        foe = armor.spawn(200.0, 0.0, 0.0, side=1)
        s2 = twin.bind(TWIN_KIND_ARMOR, foe)
        twin.status[s2] = int(TwinStatus.KIA)   # 已确认击毁
        twin.confidence[s2] = 0.95
        provider = TwinContactProvider(twin, armor)
        assert provider(0) == [], "己方 + KIA 都不应出现在敌接触表"


class TestScoreConfidence:
    """score_targets confidence 折扣（docs §5.1）。"""

    def test_confidence_discounts_score(self) -> None:
        k = 2
        dist = np.array([1000.0, 1000.0], np.float32)
        speed = np.zeros(k, np.float32)
        pen = np.zeros(k, np.float32)
        health = np.array([0.5, 0.5], np.float32)
        conf = np.array([1.0, 0.3], np.float32)   # 目标 2 迷雾浓
        out = np.zeros(k, np.float32)
        score_targets(dist, speed, pen, health, conf,
                      1.0, 1.0, 1.0, 1.0, 5000.0, out)
        assert out[0] > out[1], "高置信目标得分应更高（迷雾折扣）"
        assert out[1] == pytest.approx(out[0] * 0.3, abs=1e-6)


class TestSupportRequestChain:
    """支援请求链（docs §5.2）。"""

    def _rig_with_fc(self):
        armor = ArmorPool(capacity=16)
        formations = FormationAIPool(capacity=4)
        twin = TwinPool(capacity=64)
        events = EventBus()
        boxes = BoxPool(capacity=64)
        proj = ProjectilePool(capacity=32)
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry

        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(),
            control=MissileControlRegistry.with_builtin(),
            rng=RNGManager(seed=1),
            events=events,
        )
        provider = TwinContactProvider(twin, armor)
        ai = FormationAI(armor, formations, fc=fc, contact_provider=provider,
                         events=events)
        return ai, armor, formations, twin, events, fc

    def test_requests_when_out_of_range(self) -> None:
        ai, armor, formations, twin, events, fc = self._rig_with_fc()
        fidx, members = _spawn_squad(armor, formations, 0.0, 0)
        # 给成员配武器（最大射程 4km）
        for m in members:
            fc.set_unit_config(m, UnitCombatConfig(
                weapon_id="m256_120mm", ammo_id="m829a3_apfsds",
                crew_skill=0.7, visibility=1.0, weapon_condition=1.0,
                max_range_m=4000.0, fire_rate_rpm=6.0,
            ))
        foe = armor.spawn(10000.0, 0.0, 0.0, side=1)   # 10km 超射程
        s = twin.bind(TWIN_KIND_ARMOR, foe, pos=(10000.0, 0.0, 0.0))
        twin.status[s] = int(TwinStatus.ACTIVE)
        twin.confidence[s] = 0.9
        ai.tick(SimulationLoop(state=SimulationState()), 1.0)
        reqs = [e for e in events.drain() if e.kind == "support_request"]
        assert len(reqs) >= 1, "超射程高置信目标应触发支援请求"
        assert reqs[0].data["kind"] == "artillery"
        assert reqs[0].data["x"] == pytest.approx(10000.0)   # 孪生坐标

    def test_no_request_low_confidence(self) -> None:
        ai, armor, formations, twin, events, fc = self._rig_with_fc()
        fidx, members = _spawn_squad(armor, formations, 0.0, 0)
        for m in members:
            fc.set_unit_config(m, UnitCombatConfig(
                weapon_id="m256_120mm", ammo_id="m829a3_apfsds",
                crew_skill=0.7, visibility=1.0, weapon_condition=1.0,
                max_range_m=4000.0, fire_rate_rpm=6.0,
            ))
        foe = armor.spawn(10000.0, 0.0, 0.0, side=1)
        s = twin.bind(TWIN_KIND_ARMOR, foe, pos=(10000.0, 0.0, 0.0))
        twin.status[s] = int(TwinStatus.ACTIVE)
        twin.confidence[s] = 0.1   # 迷雾浓 → 看不清不打
        ai.tick(SimulationLoop(state=SimulationState()), 1.0)
        reqs = [e for e in events.drain() if e.kind == "support_request"]
        assert reqs == [], "低置信目标不应触发支援请求"

    def test_no_assign_low_confidence_target(self) -> None:
        ai, armor, formations, twin, events, fc = self._rig_with_fc()
        fidx, members = _spawn_squad(armor, formations, 0.0, 0)
        formations.intent[fidx] = 1   # ADVANCE
        foe = armor.spawn(500.0, 0.0, 0.0, side=1)
        s = twin.bind(TWIN_KIND_ARMOR, foe, pos=(500.0, 0.0, 0.0))
        twin.status[s] = int(TwinStatus.ACTIVE)
        twin.confidence[s] = 0.1
        ai.tick(SimulationLoop(state=SimulationState()), 1.0)
        targets = [int(armor.target_idx[m]) for m in members]
        assert all(t == -1 for t in targets), "低置信目标不应被分配"


class TestArtilleryAutoAccept:
    """炮兵自主接单（docs §6.1）。"""

    def _rig_artillery(self):
        armor = ArmorPool(capacity=16)
        formations = FormationAIPool(capacity=4)
        boxes = BoxPool(capacity=64)
        proj = ProjectilePool(capacity=32)
        events = EventBus()
        ammo = AmmoManager(capacity=16)
        from Airia.lulilo.combat.weapons import WeaponSystem
        from Airia.lulilo.combat.guidance import MissileControlRegistry

        fc = FiringControl(
            armor, boxes, proj,
            weapon_system=WeaponSystem(),
            control=MissileControlRegistry.with_builtin(),
            rng=RNGManager(seed=1),
            events=events,
            ammo=ammo,
        )
        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        # 炮兵编队
        fidx = formations.spawn(side=0)
        formations.role[fidx] = int(UnitRole.ARTILLERY)   # 编队角色 = 炮兵
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
        ammo.set_reserve(gun, 10.0) if hasattr(ammo, "set_reserve") else None
        ammo.state.reserve[gun] = 10.0   # 弹药基数
        formations.member_count[fidx] = 1
        return artillery, armor, formations, fc, events, gun

    def test_accepts_support_request_and_fires(self) -> None:
        artillery, armor, formations, fc, events, gun = self._rig_artillery()
        # 装甲编队发 support_request（孪生坐标带误差）
        events.emit("support_request", {
            "kind": "artillery",
            "requester": 0,
            "x": 5000.0, "y": 3000.0,
            "priority": 5,
            "confidence": 0.8,
        })
        loop = SimulationLoop(state=SimulationState())
        artillery.tick(loop, 1.0)
        accepted = [e for e in events.drain() if e.kind == "support_accepted"]
        assert len(accepted) == 1, "炮兵应接受支援请求"
        assert fc.projectiles.active_count >= 1, "应执行曲射开火"
        # 弹丸速度方向朝目标（孪生坐标 5000,3000）
        pj = int(fc.projectiles.active_indices()[0])
        v = fc.projectiles.vel[pj]
        assert v[0] > 0.0 and v[1] > 0.0, f"弹丸应朝目标方向: {v}"

    def test_ignores_non_artillery_request(self) -> None:
        artillery, armor, formations, fc, events, gun = self._rig_artillery()
        events.emit("support_request", {"kind": "cas", "x": 5000.0, "y": 0.0})
        artillery.tick(SimulationLoop(state=SimulationState()), 1.0)
        assert fc.projectiles.active_count == 0, "非炮兵请求不接单"
