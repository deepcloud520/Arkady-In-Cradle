"""Phase 8 步骤④：渲染切换——WorldAdapter 读 Twin + 迷雾状态映射（docs/phase8-datalink-design.md §十四 步骤④验收）。

覆盖：
    map_twin_render_status 映射表（ACTIVE/KIA/MIA/UNKNOWN + 置信度分档）、
    from_snapshot_twin（位置/血量/置信度/状态来自孪生）、
    旧 from_snapshot 行为不变（上帝视角回归）、get_render_data 输出迷雾字段。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.entities.pools import ArmorPool
from Airis.lulilo.entities.twin import (
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)
from Airis.render.adapter import (
    UnitRenderStatus,
    WorldAdapter,
    map_twin_render_status,
)
from Airis.lulilo.simulation.state import SimulationState


class TestStatusMapping:
    """孪生状态 → 渲染状态映射（docs §4.1 表）。"""

    def test_kias_and_mia(self) -> None:
        assert map_twin_render_status(int(TwinStatus.KIA), 0.95) == UnitRenderStatus.KIA
        assert map_twin_render_status(int(TwinStatus.KIA), 0.3) == UnitRenderStatus.KIA
        assert map_twin_render_status(int(TwinStatus.MIA), 0.7) == UnitRenderStatus.MIA

    def test_unknown(self) -> None:
        assert map_twin_render_status(int(TwinStatus.UNKNOWN), 0.0) == UnitRenderStatus.UNKNOWN

    def test_active_confidence_bands(self) -> None:
        # > 0.8 → ACTIVE；0.5~0.8 → ACTIVE；< 0.5 → 模糊接触框（UNKNOWN）
        assert map_twin_render_status(int(TwinStatus.ACTIVE), 0.95) == UnitRenderStatus.ACTIVE
        assert map_twin_render_status(int(TwinStatus.ACTIVE), 0.6) == UnitRenderStatus.ACTIVE
        assert map_twin_render_status(int(TwinStatus.ACTIVE), 0.4) == UnitRenderStatus.UNKNOWN


class TestAdapterTwin:
    """WorldAdapter 孪生视图（docs §4.1）。"""

    def _rig_snapshot(self) -> tuple[SimulationState, TwinPool, ArmorPool, int, int]:
        armor = ArmorPool(capacity=8)
        i = armor.spawn(100.0, 200.0, 0.0, side=0)
        j = armor.spawn(300.0, 400.0, 0.0, side=1)
        twin = TwinPool(capacity=64)
        # 己方单位：高置信观测
        s1 = twin.bind(TWIN_KIND_ARMOR, i, pos=(100.0, 200.0, 0.0), init_confidence=0.95)
        twin.status[s1] = int(TwinStatus.ACTIVE)
        twin.health[s1] = 80.0
        # 敌方单位：低置信（迷雾）
        s2 = twin.bind(TWIN_KIND_ARMOR, j, pos=(280.0, 380.0, 0.0), init_confidence=0.4)
        twin.status[s2] = int(TwinStatus.ACTIVE)
        twin.health[s2] = 50.0
        state = SimulationState(armor_pool=armor, twin_pool=twin)
        return state, twin, armor, i, j

    def test_twin_view_position_health_confidence(self) -> None:
        state, twin, armor, i, j = self._rig_snapshot()
        snap = state.snapshot()
        adapter = WorldAdapter.from_snapshot_twin(snap, twin)
        # 位置来自孪生（敌方位置带误差：280 vs 真值 300）
        assert adapter.armor_pos[j, 0] == pytest.approx(280.0)
        assert adapter.armor_health[i] == pytest.approx(80.0)
        assert adapter.armor_confidence[i] == pytest.approx(0.95)
        assert adapter.armor_confidence[j] == pytest.approx(0.4)

    def test_twin_view_render_status(self) -> None:
        state, twin, armor, i, j = self._rig_snapshot()
        snap = state.snapshot()
        adapter = WorldAdapter.from_snapshot_twin(snap, twin)
        # 己方高置信 → ACTIVE；敌方低置信 → 模糊接触框（UNKNOWN）
        assert adapter.armor_status[i] == int(UnitRenderStatus.ACTIVE)
        assert adapter.armor_status[j] == int(UnitRenderStatus.UNKNOWN)

    def test_twin_view_kia_renders(self) -> None:
        state, twin, armor, i, j = self._rig_snapshot()
        twin.status[twin.slot_of(TWIN_KIND_ARMOR, j)] = int(TwinStatus.KIA)
        snap = state.snapshot()
        adapter = WorldAdapter.from_snapshot_twin(snap, twin)
        assert adapter.armor_status[j] == int(UnitRenderStatus.KIA)

    def test_get_render_data_includes_fog(self) -> None:
        state, twin, armor, i, j = self._rig_snapshot()
        snap = state.snapshot()
        adapter = WorldAdapter.from_snapshot_twin(snap, twin)
        data = adapter.get_render_data()
        assert "confidence" in data
        assert "status" in data
        assert data["confidence"][i] == pytest.approx(0.95)

    def test_legacy_from_snapshot_unchanged(self) -> None:
        """旧 from_snapshot 行为不变（上帝视角回归，Phase 5 契约）。"""
        state, twin, armor, i, j = self._rig_snapshot()
        snap = state.snapshot()
        adapter = WorldAdapter.from_snapshot(snap)
        assert adapter.armor_pos[i, 0] == pytest.approx(100.0)   # 真值
        assert np.all(adapter.armor_confidence == pytest.approx(1.0))
        assert np.all(adapter.armor_status == 0)   # 全 ACTIVE
