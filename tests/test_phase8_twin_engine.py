"""Phase 8 步骤②：TwinEngine 四通道 + 死亡双通道报告（docs/phase8-datalink-design.md §十四 步骤②验收）。

覆盖：
    通道① 传感器注入（ACTIVE + confidence + last_update）、观测质量门控（低 comm 降级）、
    通道③ 时间衰减 → UNKNOWN、KIA 置信衰减回退、
    死亡双通道（confirm_kia 立即 KIA / 物理失活 → MIA）、
    通道④ 延迟报告注入（倒计时到点写孪生）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.entities.pools import ArmorPool
from Airia.lulilo.entities.twin import (
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)
from Airia.lulilo.simulation.twin_engine import TwinEngine


class _Loop:
    def __init__(self) -> None:
        self.state = type("S", (), {"tick_count": 0})()


def _rig() -> tuple[TwinPool, TwinEngine, ArmorPool, _Loop]:
    twin = TwinPool(capacity=64)
    armor = ArmorPool(capacity=8)
    engine = TwinEngine(twin, armor=armor)
    loop = _Loop()
    return twin, engine, armor, loop


class TestSensorInject:
    """通道①：传感器观测注入。"""

    def test_observe_writes_twin(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(100.0, 200.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i, pos=(0.0, 0.0, 0.0))
        loop.state.tick_count = 5
        engine.observe(TWIN_KIND_ARMOR, i, (100.0, 200.0, 0.0), (1.0, 0.0, 0.0), 0.9)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        assert twin.confidence[s] == pytest.approx(0.9)
        assert twin.last_update[s] == pytest.approx(5.0)
        assert twin.pos[s, 0] == pytest.approx(100.0)
        assert twin.vel[s, 0] == pytest.approx(1.0)

    def test_low_comm_observation_degraded(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.observe(TWIN_KIND_ARMOR, i, (10.0, 0.0, 0.0), confidence=0.9,
                       observer_comm=0.4)   # < 0.8 → 降级
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.confidence[s] == pytest.approx(0.9 * 0.4, abs=1e-6)

    def test_no_observation_keeps_unknown(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.UNKNOWN)


class TestDecay:
    """通道③：时间衰减。"""

    def test_stale_decays_to_unknown(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.observe(TWIN_KIND_ARMOR, i, (0.0, 0.0, 0.0), confidence=0.5)
        engine.tick(loop, 1.0)   # tick 0：观测
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        # 衰减到 UNKNOWN：置信度 0.5 → 多次 tick 后 < 0.3
        for _ in range(100):
            loop.state.tick_count += 1
            engine.tick(loop, 1.0)
            if twin.status[s] == int(TwinStatus.UNKNOWN):
                break
        assert twin.status[s] == int(TwinStatus.UNKNOWN), "超时衰减应降级 UNKNOWN"

    def test_fresh_observation_no_decay(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        for _ in range(20):
            loop.state.tick_count += 1
            engine.observe(TWIN_KIND_ARMOR, i, (0.0, 0.0, 0.0), confidence=0.9)
            engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.ACTIVE), "持续观测不应衰减"
        assert twin.confidence[s] == pytest.approx(0.9)

    def test_kia_confidence_backoff(self) -> None:
        """KIA 但置信度衰减 → 回退（误报重估，docs §2.1 ③）。"""
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.confirm_kia(TWIN_KIND_ARMOR, i, confidence=0.5)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.KIA)
        # 无观测 → 衰减 → < 0.5 → 回退 UNKNOWN
        for _ in range(200):
            loop.state.tick_count += 1
            engine.tick(loop, 1.0)
            if twin.status[s] == int(TwinStatus.UNKNOWN):
                break
        assert twin.status[s] == int(TwinStatus.UNKNOWN), "KIA 置信衰减应回退"


class TestDeathReporting:
    """死亡双通道（docs §2.3）。"""

    def test_confirm_kia_immediate(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.observe(TWIN_KIND_ARMOR, i, (0.0, 0.0, 0.0), confidence=0.9)
        engine.tick(loop, 1.0)
        engine.confirm_kia(TWIN_KIND_ARMOR, i)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.KIA)
        assert twin.confidence[s] == pytest.approx(0.95)

    def test_physical_destruction_leads_to_mia(self) -> None:
        """物理单位被毁但无人观测 → 通信中断 → MIA（信号丢失）。"""
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.observe(TWIN_KIND_ARMOR, i, (0.0, 0.0, 0.0), confidence=0.9)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        armor.destroy(i)   # 物理销毁
        loop.state.tick_count += 1
        engine.tick(loop, 1.0)
        assert twin.status[s] == int(TwinStatus.MIA), "物理失活 → 信号丢失 MIA"

    def test_mia_then_report_confirms_kia(self) -> None:
        """MIA → 后续侦察报告 → KIA（概念文档验收时序的简化）。"""
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.observe(TWIN_KIND_ARMOR, i, (0.0, 0.0, 0.0), confidence=0.9)
        engine.tick(loop, 1.0)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        armor.destroy(i)
        loop.state.tick_count += 1
        engine.tick(loop, 1.0)
        assert twin.status[s] == int(TwinStatus.MIA)
        engine.confirm_kia(TWIN_KIND_ARMOR, i, confidence=0.95)
        assert twin.status[s] == int(TwinStatus.KIA)


class TestDelayedReport:
    """通道④：延迟报告注入。"""

    def test_report_arrives_after_delay(self) -> None:
        twin, engine, armor, loop = _rig()
        i = armor.spawn(0.0, 0.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i)
        engine.submit_report(TWIN_KIND_ARMOR, i, pos=(500.0, 0.0, 0.0),
                             confidence=0.6, delay_ticks=3)
        s = twin.slot_of(TWIN_KIND_ARMOR, i)
        # 延迟期间不写
        for _ in range(2):
            loop.state.tick_count += 1
            engine.tick(loop, 1.0)
            assert twin.status[s] == int(TwinStatus.UNKNOWN)
        # 第 3 tick 到点注入
        loop.state.tick_count += 1
        engine.tick(loop, 1.0)
        assert twin.status[s] == int(TwinStatus.ACTIVE)
        assert twin.pos[s, 0] == pytest.approx(500.0)
        assert twin.confidence[s] == pytest.approx(0.6)
        assert twin.source[s] == 3   # 友军报告
