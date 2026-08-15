"""Phase 8 步骤①：TwinPool + 绑定生命周期 + 序列化（docs/phase8-datalink-design.md §十四 步骤①验收）。

覆盖：
    1:1 槽映射（armor/aircraft 无重叠）、bind 初始化（UNKNOWN + confidence 0）、
    绑定查询、destroy 保留孪生（残骸渲染）、序列化往返（TwinPool 全字段）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.entities.pools import MAX_ARMOR, AircraftPool, ArmorPool
from Airis.lulilo.entities.twin import (
    MAX_TWIN,
    TWIN_KIND_AIRCRAFT,
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)
from Airis.lulilo.simulation.serialization import (
    deserialize_battlefield,
    serialize_battlefield,
)
from Airis.lulilo.simulation.state import SimulationState


class TestSlotMapping:
    """1:1 槽映射（无 dict，AGENTS.md 4.3 缓存纪律）。"""

    def test_armor_slot_is_idx(self) -> None:
        assert TwinPool.slot_of(TWIN_KIND_ARMOR, 0) == 0
        assert TwinPool.slot_of(TWIN_KIND_ARMOR, 2047) == 2047
        assert TwinPool.slot_of(TWIN_KIND_ARMOR, 2048) == -1   # 越界
        assert TwinPool.slot_of(TWIN_KIND_ARMOR, -1) == -1

    def test_aircraft_slot_offset(self) -> None:
        assert TwinPool.slot_of(TWIN_KIND_AIRCRAFT, 0) == MAX_ARMOR
        assert TwinPool.slot_of(TWIN_KIND_AIRCRAFT, 127) == MAX_ARMOR + 127
        assert TwinPool.slot_of(TWIN_KIND_AIRCRAFT, 128) == -1

    def test_no_overlap_between_kinds(self) -> None:
        """装甲槽 [0, 2048) 与飞机槽 [2048, 2176) 不重叠。"""
        armor_slots = {TwinPool.slot_of(TWIN_KIND_ARMOR, i) for i in range(0, 100)}
        air_slots = {TwinPool.slot_of(TWIN_KIND_AIRCRAFT, i) for i in range(0, 100)}
        assert armor_slots.isdisjoint(air_slots)


class TestBindLifecycle:
    """绑定生命周期（docs §1.3）。"""

    def test_bind_initializes_unknown(self) -> None:
        twin = TwinPool()
        armor = ArmorPool(capacity=4)
        i = armor.spawn(100.0, 200.0, 0.0)
        s = twin.bind(TWIN_KIND_ARMOR, i, pos=(100.0, 200.0, 0.0))
        assert s == i
        assert twin.active[s]
        assert twin.unit_kind[s] == TWIN_KIND_ARMOR
        assert twin.twin_idx[s] == i
        assert twin.status[s] == int(TwinStatus.UNKNOWN)
        assert twin.confidence[s] == pytest.approx(0.0)
        assert twin.pos[s, 0] == pytest.approx(100.0)
        assert twin.is_bound(TWIN_KIND_ARMOR, i)

    def test_bind_aircraft(self) -> None:
        twin = TwinPool()
        aircraft = AircraftPool(capacity=4)
        i = aircraft.spawn(0.0, 0.0, 3000.0)
        s = twin.bind(TWIN_KIND_AIRCRAFT, i, pos=(0.0, 0.0, 3000.0),
                      vel=(250.0, 0.0, 0.0), init_confidence=0.8)
        assert s == MAX_ARMOR + i
        assert twin.confidence[s] == pytest.approx(0.8)
        assert twin.last_vel_known[s, 0] == pytest.approx(250.0)

    def test_destroy_keeps_twin(self) -> None:
        """物理单位销毁 → 孪生保留（残骸/失联由引擎标记，docs §1.3）。"""
        twin = TwinPool()
        armor = ArmorPool(capacity=4)
        i = armor.spawn(0.0, 0.0, 0.0)
        s = twin.bind(TWIN_KIND_ARMOR, i, pos=(0.0, 0.0, 0.0))
        armor.destroy(i)
        assert twin.active[s], "destroy 不销毁孪生（残骸渲染）"
        assert twin.active_count == 1

    def test_active_count(self) -> None:
        twin = TwinPool()
        armor = ArmorPool(capacity=8)
        for _ in range(3):
            twin.bind(TWIN_KIND_ARMOR, armor.spawn(0, 0, 0))
        assert twin.active_count == 3

    def test_slot_rebind(self) -> None:
        """同槽重绑（物理索引复用）→ 覆盖旧孪生。"""
        twin = TwinPool()
        armor = ArmorPool(capacity=4)
        i1 = armor.spawn(0.0, 0.0, 0.0)
        i2 = armor.spawn(1.0, 1.0, 0.0)
        twin.bind(TWIN_KIND_ARMOR, i1, pos=(0.0, 0.0, 0.0))
        twin.bind(TWIN_KIND_ARMOR, i2, pos=(1.0, 1.0, 0.0))
        assert twin.active_count == 2
        assert twin.twin_idx[i1] == i1
        assert twin.twin_idx[i2] == i2


class TestSerialization:
    """TwinPool 存档往返（docs §1.3）。"""

    def test_round_trip(self) -> None:
        twin = TwinPool(capacity=64)
        armor = ArmorPool(capacity=4)
        i = armor.spawn(100.0, 200.0, 0.0)
        s = twin.bind(TWIN_KIND_ARMOR, i, pos=(100.0, 200.0, 0.0))
        twin.status[s] = int(TwinStatus.ACTIVE)
        twin.confidence[s] = 0.9
        twin.pos_err[s] = 25.0
        twin.last_update[s] = 42.0
        twin.source[s] = 1

        state = SimulationState(twin_pool=twin)
        blob = serialize_battlefield(state)
        save = deserialize_battlefield(blob)
        assert save.twin is not None
        t2 = save.twin
        assert t2.capacity == twin.capacity
        j = t2.slot_of(TWIN_KIND_ARMOR, i)
        assert t2.active[j]
        assert t2.status[j] == int(TwinStatus.ACTIVE)
        assert t2.confidence[j] == pytest.approx(0.9)
        assert t2.pos_err[j] == pytest.approx(25.0)
        assert t2.last_update[j] == pytest.approx(42.0)
        assert t2.pos[j, 0] == pytest.approx(100.0)

    def test_constants(self) -> None:
        assert MAX_TWIN >= MAX_ARMOR + 128

    def test_missing_twin_old_save(self) -> None:
        """旧存档无 twin 段 → 读档 twin=None（不崩）。"""
        state = SimulationState(armor_pool=ArmorPool(capacity=2))
        blob = serialize_battlefield(state)
        save = deserialize_battlefield(blob)
        assert save.twin is None
