"""孪生更新引擎（Phase 8，docs/phase8-datalink-design.md §二，Python 1s tick）。

四条更新通道（§2.1）：
    ① 传感器注入：detection 层输出（友军 comm ≥ 0.8 才算「可上报」观测）——
       写 pos/vel/status=ACTIVE/confidence/last_update。
    ② 数据链航迹共享：`DataLinkNetwork.broadcast_contacts()` → 同簇单位互相
       校正孪生（多源融合取最高置信度、最小误差）（§三.3，步骤③接入）。
    ③ 时间衰减：`now - last_update > T_STALE` → confidence *= 0.99^Δt；
       < CONF_UNKNOWN → 降级 UNKNOWN；KIA 置信衰减 → 回退（误报重估）。
    ④ 延迟报告注入：pending 队列 (delay_ticks, report) 逐 tick 倒计时 → 到点写孪生。

死亡双通道报告（§2.3）：
    - 直接观测击毁（击毁者 comm ≥ 0.8）→ `confirm_kia` 立即 KIA + confidence 0.95；
    - 未观测 → 物理失活但孪生 ACTIVE → 通信中断 → 转 MIA（信号丢失）→
      后续侦察/报告 → KIA。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from Airia.lulilo.entities.twin import (
    TWIN_KIND_AIRCRAFT,
    TWIN_KIND_ARMOR,
    TwinPool,
    TwinStatus,
)

_ST_UNKNOWN = int(TwinStatus.UNKNOWN)
_ST_ACTIVE = int(TwinStatus.ACTIVE)
_ST_KIA = int(TwinStatus.KIA)
_ST_MIA = int(TwinStatus.MIA)

# 观测源（TwinPool.source）
_SRC_NONE = 0
_SRC_OWN = 1        # 本队观测
_SRC_DATALINK = 2   # 数据链共享航迹
_SRC_REPORT = 3     # 友军无线电报告
_SRC_NARRATIVE = 4  # 叙事层注入


class TwinEngine:
    """认知孪生更新引擎（四条通道 + 死亡双通道报告）。"""

    # ---- 衰减参数（docs §2.1 ③）----
    T_STALE = 10.0          # 最近观测超时秒数 → 开始衰减
    CONF_DECAY = 0.99       # 每秒置信度衰减因子
    CONF_UNKNOWN = 0.3      # 置信度低于此 → 降级 UNKNOWN
    CONF_MIA = 0.5          # 物理失活且置信度低于此 → 转 MIA（信号丢失）
    CONF_KIA_BACKOFF = 0.5  # KIA 置信度衰减低于此 → 回退（误报重估）
    OBSERVE_COMM_MIN = 0.8  # 观测者 comm ≥ 此值才算「可上报」观测（§2.1 ①）

    def __init__(
        self,
        twin: TwinPool,
        *,
        armor: Any | None = None,
        aircraft: Any | None = None,
        datalink: Any | None = None,
        events: Any | None = None,
        rng: Any | None = None,
    ) -> None:
        self.twin = twin
        self.armor = armor
        self.aircraft = aircraft
        self.datalink = datalink
        self.events = events
        self.rng = rng
        # 本 tick 累积的观测（外部 detection/AI 在 tick 前调 observe）
        self._observations: list[tuple[int, int, Any, Any, float, float, float]] = []
        # 延迟报告队列：[(remaining_ticks, (kind, idx, pos, vel, confidence, source))]
        self._pending_reports: list[list] = []
        self._now = 0.0
        self._tick_seconds = 1.0

    # ------------------------------------------------------------------
    # 观测入口（外部调用：detection/AI/防空/巡飞弹）
    # ------------------------------------------------------------------

    def observe(
        self,
        kind: int,
        idx: int,
        pos: Any,
        vel: Any | None = None,
        confidence: float = 0.9,
        observer_comm: float = 1.0,
        health: float | None = None,
    ) -> None:
        """本队观测注入（通道① 的数据源，docs §2.1 ①）。

        observer_comm < `OBSERVE_COMM_MIN` 的观测降级（confidence × comm，
        且不入数据链广播——broadcast 按簇成员 comm 再过滤）。
        health：报告血量（滞后/误差，docs §5.1「血量 = twin 报告值」）。
        """
        if observer_comm < self.OBSERVE_COMM_MIN:
            confidence *= max(observer_comm, 0.0)
        self._observations.append(
            (kind, idx, pos, vel, float(confidence), observer_comm,
             float(health) if health is not None else -1.0)
        )

    # ------------------------------------------------------------------
    # 死亡双通道报告
    # ------------------------------------------------------------------

    def confirm_kia(self, kind: int, idx: int, confidence: float = 0.95) -> None:
        """直接观测击毁（击毁者 comm ≥ 0.8 调用）→ 立即 KIA（docs §2.3）。"""
        s = self.twin.slot_of(kind, idx)
        if 0 <= s < self.twin.capacity and self.twin.active[s]:
            self.twin.status[s] = _ST_KIA
            self.twin.confidence[s] = max(self.twin.confidence[s], confidence)
            self.twin.source[s] = _SRC_OWN
            self.twin.last_update[s] = self._now

    def submit_report(
        self,
        kind: int,
        idx: int,
        *,
        pos: Any = None,
        vel: Any | None = None,
        confidence: float = 0.6,
        source: int = _SRC_REPORT,
        delay_ticks: int = 0,
    ) -> None:
        """延迟报告注入（通道④，叙事/友军报告，docs §2.1 ④ / §九）。"""
        self._pending_reports.append([
            max(delay_ticks, 0),
            (kind, idx, pos, vel, float(confidence), source),
        ])

    # ------------------------------------------------------------------
    # 主循环（每 1s 地面 Tick）
    # ------------------------------------------------------------------

    def tick(self, loop: Any, dt: float) -> None:
        self._now = float(loop.state.tick_count) if hasattr(loop, "state") else 0.0
        self._tick_seconds = float(dt)
        self._channel_sensor_inject()
        if self.datalink is not None:
            self.datalink.broadcast_contacts(self.twin)
        self._channel_decay()
        self._channel_delayed_reports()
        self._observations.clear()

    # ------------------------------------------------------------------
    # 通道① 传感器注入
    # ------------------------------------------------------------------

    def _channel_sensor_inject(self) -> None:
        for kind, idx, pos, vel, conf, _comm, health in self._observations:
            self._write(kind, idx, pos, vel, conf, _SRC_OWN, now=True, health=health)

    # ------------------------------------------------------------------
    # 通道③ 时间衰减 + 失联/死亡状态机
    # ------------------------------------------------------------------

    def _channel_decay(self) -> None:
        t = self.twin
        now = self._now
        dt = self._tick_seconds
        if not np.any(t.active):
            return
        slots = np.flatnonzero(t.active)
        st = t.status[slots].copy()
        kind = t.unit_kind[slots]
        idx = t.twin_idx[slots]

        # 物理失活掩码（批量：armor.active / aircraft.active 映射到孪生槽）
        phys_alive = np.ones(slots.size, dtype=np.bool_)
        if self.armor is not None:
            a_mask = kind == TWIN_KIND_ARMOR
            if np.any(a_mask):
                aidx = idx[a_mask]
                ok = (aidx >= 0) & (aidx < self.armor.capacity)
                alive = np.zeros(aidx.size, dtype=np.bool_)
                if np.any(ok):
                    alive[ok] = self.armor.active[aidx[ok]]
                phys_alive[a_mask] = alive
        if self.aircraft is not None:
            c_mask = kind == TWIN_KIND_AIRCRAFT
            if np.any(c_mask):
                cidx = idx[c_mask]
                ok = (cidx >= 0) & (cidx < self.aircraft.capacity)
                alive = np.zeros(cidx.size, dtype=np.bool_)
                if np.any(ok):
                    alive[ok] = self.aircraft.active[cidx[ok]]
                phys_alive[c_mask] = alive

        # MIA：物理失活 且（ACTIVE/UNKNOWN）→ 信号丢失
        mia = (~phys_alive) & np.isin(st, (_ST_ACTIVE, _ST_UNKNOWN))
        if np.any(mia):
            t.status[slots[mia]] = _ST_MIA
            t.confidence[slots[mia]] *= 0.9

        # 时间衰减（观测超时，非失活）
        age = now - t.last_update[slots]
        decay_mask = (~mia) & (age > self.T_STALE) & (st != _ST_UNKNOWN)
        if np.any(decay_mask):
            t.confidence[slots[decay_mask]] *= self.CONF_DECAY ** dt
            # KIA 置信衰减 → 回退（误报重估）
            kia_back = decay_mask & (st == _ST_KIA) \
                & (t.confidence[slots] < self.CONF_KIA_BACKOFF)
            if np.any(kia_back):
                t.status[slots[kia_back]] = _ST_UNKNOWN
            # 降级 UNKNOWN（超时）
            low = decay_mask & (t.confidence[slots] < self.CONF_UNKNOWN)
            if np.any(low):
                t.status[slots[low]] = _ST_UNKNOWN

    def _phys_alive(self, kind: int, idx: int) -> bool:
        if kind == TWIN_KIND_ARMOR and self.armor is not None:
            return 0 <= idx < self.armor.capacity and bool(self.armor.active[idx])
        if kind == TWIN_KIND_AIRCRAFT and self.aircraft is not None:
            return 0 <= idx < self.aircraft.capacity and bool(self.aircraft.active[idx])
        return True   # 无物理池引用 → 视为存活（不误判失联）

    # ------------------------------------------------------------------
    # 通道④ 延迟报告注入
    # ------------------------------------------------------------------

    def _channel_delayed_reports(self) -> None:
        for r in list(self._pending_reports):
            r[0] -= 1
            if r[0] <= 0:
                kind, idx, pos, vel, conf, src = r[1]
                self._write(kind, idx, pos, vel, conf, src, now=True)
                self._pending_reports.remove(r)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def _write(
        self, kind: int, idx: int, pos: Any, vel: Any | None,
        confidence: float, source: int, *, now: bool = False,
        health: float = -1.0,
    ) -> None:
        s = self.twin.slot_of(kind, idx)
        if s < 0 or s >= self.twin.capacity or not self.twin.active[s]:
            return
        t = self.twin
        # 多源融合：取最高置信度、最小误差（docs §2.1 ②）
        if confidence >= t.confidence[s] or t.status[s] == _ST_UNKNOWN:
            t.confidence[s] = float(confidence)
            t.source[s] = int(source)
            if pos is not None:
                t.pos[s] = pos
                if vel is not None:
                    t.vel[s] = vel
                    t.last_vel_known[s] = vel
                else:
                    # 位置更新但无速度 → 外推速度 = 位移/dt（简化）
                    pass
            if health >= 0.0:
                t.health[s] = float(health)
            t.status[s] = _ST_ACTIVE
            t.pos_err[s] = max(0.0, 50.0 * (1.0 - float(confidence)))
            if now:
                t.last_update[s] = self._now
