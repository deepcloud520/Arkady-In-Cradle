"""空中编队/单位 AI（Phase 7，docs/phase7-air-pipeline.md §四~§六）。

架构（design §〇.2/§〇.3）：
    - **编队 AI 决策，单位 AI 执行**：决策主体是 `AirFormationAI`（意图 FSM + 目标分配），
      单位 AI（`AirUnitAI`）只飞/只打——写 `order`/`target_idx`，由飞控 +
      `FiringControl.fire_air()` 落地。
    - **感知驱动，非上帝视角**：机载雷达走 `detection/sensors.evaluate_sensor_detection`
      （平台级批量检测，6b 已泛化）→ LOS 过滤 → 检测概率采样 → 编队接触表。
    - **热路径向量化，冷路径 Python**：检测内核在 sensors.py（Numba）；本模块
      gather/FSM/分配是 Python 决策层（≤128 架 × 秒级，可忽略）。

步骤④：`gather_air_contacts`（机载雷达搜索，docs §四.2）
步骤⑤：`AirFormationAI`（意图 FSM + 贪心分配 + 编队飞行，docs §五）
步骤⑥：`AirUnitAI`（状态机 + 空战评分，docs §六）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numba import njit

from Airia.lulilo.detection.sensors import (
    evaluate_sensor_detection,
    radar_detection_probability,
)
from Airia.lulilo.entities.pools import EmitterBand, ProjectileType
from Airia.lulilo.entities.twin import TWIN_KIND_ARMOR

# ---------------------------------------------------------------------------
# 枚举（追加值不重排，读档确定性）
# ---------------------------------------------------------------------------


class AirFormationIntent(IntEnum):
    """编队意图状态机（docs §5.2）。"""
    CAP = 0        # 巡逻/空战
    STRIKE = 1     # 对地打击
    SEAD = 2       # 压制防空
    ESCORT = 3     # 护航
    RTB = 4        # 返航


class AirUnitOrder(IntEnum):
    """单位 AI 订单（docs §6.1）。"""
    HOLD = 0
    CAP = 1
    ENGAGE_AIR = 2
    STRIKE_GROUND = 3
    SUPPRESS = 4
    RTB = 5
    LAND = 6


# 状态常量（entities 层 AircraftState）
_STATE_TAKEOFF = 0
_STATE_CRUISE = 1
_STATE_ENGAGE = 2
_STATE_RTB = 3
_STATE_LANDING = 4
_STATE_CRASHING = 5
_STATE_PARKED = 6

# 目标种类（AircraftPool.target_kind 语义）
TGT_AIR = 0
TGT_GROUND = 1

# 地球半径（雷达地平线修正，docs §四.2）
_EARTH_R = 6371000.0


# ---------------------------------------------------------------------------
# 接触表（步骤④产物）
# ---------------------------------------------------------------------------


@dataclass
class AirContact:
    """一次机载雷达接触（docs §四.2 输出）。"""
    tgt_idx: int            # 目标池索引
    tgt_kind: int           # TGT_AIR / TGT_GROUND
    dist: float             # 斜距 (m)
    d_alt: float            # 高度差（我 - 目标）(m)
    speed: float            # 目标速度 (m/s)
    health: float           # 目标血量（0~1 归一或原始）
    rcs: float              # 目标 RCS (m²)
    los: bool = True        # 视线清晰


def _radar_horizon(alt: float) -> float:
    """雷达地平线距离 R_h = √(2·Re·h)（docs §四.2 视距截断）。"""
    h = max(alt, 0.0)
    return math.sqrt(2.0 * _EARTH_R * h)


def gather_air_contacts(
    side: int,
    aircraft: Any,
    armor: Any,
    contacts: list[list[AirContact]],
    rng: Any = None,
    los_checker: Any = None,
    radar_power_db: float = 70.0,
) -> None:
    """逐架机载雷达搜索敌方空中/地面目标 → 写接触表（docs §四.2）。

    流程（全复用 6b 检测层，本函数零新检测公式）：
        1. 打包本 side 活跃飞机为传感器（sensor_type=0 雷达）。
        2. 打包敌方活跃飞机 + 敌方活跃地面为目标（rcs 数组）。
        3. `evaluate_sensor_detection` 批量算出 max_range（含 1.5× 门限）。
        4. 雷达地平线截断 + LOS 过滤 + 检测概率采样 → 接触条目。

    Args:
        side: 己方阵营（只收集本 side 飞机的探测结果）。
        aircraft: AircraftPool。
        armor: ArmorPool。
        contacts: 长度 ≥ aircraft.capacity 的列表；写
            contacts[air_idx] = [AirContact, ...]（无接触 → 清空列表）。
        rng: RNGManager（检测概率采样；None → 全命中，确定性测试用）。
        los_checker: LosChecker（None → 跳过 LOS 过滤，空战测试用）。
        radar_power_db: 机载雷达功率（默认 70dBm，与 AircraftConfig 对齐）。
    """
    # 清空本 side 传感器槽（不碰其他 side——多 side 顺序 gather 时互不清，
    # 否则先 gather 的 side 接触被后一次清空）
    air_idx = np.flatnonzero(
        aircraft.active
        & (aircraft.side == side)
        & (aircraft.state != _STATE_CRASHING)
        & (aircraft.state != _STATE_PARKED)
    )
    for i in air_idx:
        contacts[int(i)].clear()
    if air_idx.size == 0:
        return

    # 敌方飞机（活跃、非坠毁/驻停）
    e_air = np.flatnonzero(
        aircraft.active
        & (aircraft.side != side)
        & (aircraft.state != _STATE_CRASHING)
        & (aircraft.state != _STATE_PARKED)
    )
    # 敌方地面（活跃装甲）
    e_gr = np.flatnonzero(armor.active & (armor.side != side))

    n_t = e_air.size + e_gr.size
    if n_t == 0:
        return

    t_x = np.concatenate([aircraft.pos[e_air, 0], armor.pos[e_gr, 0]]).astype(np.float32)
    t_y = np.concatenate([aircraft.pos[e_air, 1], armor.pos[e_gr, 1]]).astype(np.float32)
    t_z = np.concatenate([aircraft.pos[e_air, 2], armor.pos[e_gr, 2]]).astype(np.float32)
    t_rcs = np.concatenate([aircraft.rcs[e_air], armor.rcs[e_gr]]).astype(np.float32)
    t_ir = np.zeros(n_t, dtype=np.float32)
    t_active = np.ones(n_t, dtype=np.bool_)
    t_kind = np.concatenate(
        [np.full(e_air.size, TGT_AIR, np.int8),
         np.full(e_gr.size, TGT_GROUND, np.int8)]
    )

    n_s = air_idx.size
    s_x = aircraft.pos[air_idx, 0].astype(np.float32)
    s_y = aircraft.pos[air_idx, 1].astype(np.float32)
    s_z = aircraft.pos[air_idx, 2].astype(np.float32)
    s_type = np.zeros(n_s, dtype=np.int32)
    s_active = np.ones(n_s, dtype=np.bool_)

    max_range_cache = np.zeros((n_s, n_t), dtype=np.float32)
    evaluate_sensor_detection(
        s_x, s_y, s_z, t_x, t_y, t_z,
        s_type, t_rcs, t_ir, s_active, t_active, max_range_cache,
    )

    # 距离矩阵（重算：内核只写 max_range 门限）
    dx = t_x[None, :] - s_x[:, None]
    dy = t_y[None, :] - s_y[:, None]
    dz = t_z[None, :] - s_z[:, None]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    # 逐传感器写接触（Python 决策层，秒级）
    for si, a_i in enumerate(air_idx):
        a_i = int(a_i)
        sensor_alt = float(aircraft.pos[a_i, 2])
        horizon = _radar_horizon(sensor_alt)
        for ti in range(n_t):
            mr = float(max_range_cache[si, ti])
            if mr <= 0.0:
                continue
            d = float(dist[si, ti])
            # 距离门限与 evaluate_sensor_detection 内核一致（1.5× max_r）
            if d > mr * 1.5:
                continue
            # 雷达地平线截断（低空时 R_h < R_radar → 生效）
            if d > horizon:
                continue
            # 检测概率采样
            if rng is not None:
                p = radar_detection_probability(d, mr)
                if rng.uniform() >= p:
                    continue
            # LOS 过滤（可选）
            if los_checker is not None and not los_checker.check(
                float(aircraft.pos[a_i, 0]), float(aircraft.pos[a_i, 1]), sensor_alt,
                float(t_x[ti]), float(t_y[ti]), float(t_z[ti]),
            ):
                continue
            if ti < e_air.size:
                t_idx = int(e_air[ti])
                contacts[a_i].append(AirContact(
                    tgt_idx=t_idx, tgt_kind=TGT_AIR,
                    dist=d, d_alt=float(aircraft.pos[a_i, 2]) - float(t_z[ti]),
                    speed=float(np.linalg.norm(aircraft.vel[t_idx])),
                    health=float(aircraft.health[t_idx]),
                    rcs=float(t_rcs[ti]), los=True,
                ))
            else:
                gi = ti - e_air.size
                g_idx = int(e_gr[gi])
                contacts[a_i].append(AirContact(
                    tgt_idx=g_idx, tgt_kind=TGT_GROUND,
                    dist=d, d_alt=float(aircraft.pos[a_i, 2]) - float(t_z[ti]),
                    speed=float(np.linalg.norm(armor.vel[g_idx])),
                    health=0.0,   # 装甲血量在 BoxPool.box_hp，接触表不深挖
                    rcs=float(t_rcs[ti]), los=True,
                ))


# ---------------------------------------------------------------------------
# 空战评分内核（步骤⑥ §6.3；编队 AI 并发轮消费）
# ---------------------------------------------------------------------------


@njit(cache=True)
def score_air_targets(
    dist: np.ndarray,          # (K,) float32 斜距
    d_alt: np.ndarray,         # (K,) float32 高度差（我 − 目标，正=我在上方）
    tgt_health: np.ndarray,    # (K,) float32 目标血量 0~1
    los_x: np.ndarray,         # (K,) float32 我→目标视线单位向量 x
    los_y: np.ndarray,
    los_z: np.ndarray,
    tvx: np.ndarray,           # (K,) float32 目标速度 x（未归一）
    tvy: np.ndarray,
    tvz: np.ndarray,
    max_range: float,          # 归一化参考距离
    w_r: float,
    w_alt: float,
    w_aspect: float,
    w_energy: float,
    out: np.ndarray,           # (K,) float32 输出威胁分（越高越优先）
) -> None:
    """空战目标评分（docs §6.3）：距离 + 高度（比能） + 尾追角 + 威胁度。

    与地面 `score_targets`（ai.py）正交：空战的「位置优势」本质是
    尾追角 + 高度（比能），而非速度/穿深。
    """
    k = dist.shape[0]
    for i in range(k):
        tv = np.sqrt(tvx[i] * tvx[i] + tvy[i] * tvy[i] + tvz[i] * tvz[i])
        tail = 0.0
        if tv > 1e-6:
            tail = (los_x[i] * tvx[i] + los_y[i] * tvy[i] + los_z[i] * tvz[i]) / tv
        # +1=目标背离我（我追尾，尾追优势）；-1=目标迎头。高度优势：d_alt>0（我高）加分
        out[i] = (
            w_r * (1.0 - dist[i] / max_range)
            + w_alt * (d_alt[i] / max_range)
            + w_aspect * (1.0 + tail) * 0.5
            + w_energy * (1.0 - tgt_health[i])
        )


# ---------------------------------------------------------------------------
# 编队 AI（步骤⑤，docs §五）
# ---------------------------------------------------------------------------

# 队形槽位（相对长机水平偏移，docs §5.4；按 slot 索引查表）
_SLOT_WEDGE = [(0.0, 0.0), (-350.0, -220.0), (350.0, 220.0),
               (-700.0, -440.0), (700.0, 440.0)]     # CAP 楔形
_SLOT_COLUMN = [(0.0, 0.0), (0.0, -350.0), (0.0, -700.0), (0.0, -1050.0)]  # STRIKE 纵队
_SLOT_LINE = [(0.0, 0.0), (-450.0, 0.0), (450.0, 0.0), (-900.0, 0.0)]      # SEAD 松散横队

_INTENT_SLOTS = {
    int(AirFormationIntent.CAP): _SLOT_WEDGE,
    int(AirFormationIntent.STRIKE): _SLOT_COLUMN,
    int(AirFormationIntent.SEAD): _SLOT_LINE,
    int(AirFormationIntent.ESCORT): _SLOT_WEDGE,
    int(AirFormationIntent.RTB): _SLOT_COLUMN,
}

# 意图 FSM 阈值（docs §5.2）
_AAM_RATIO_ADVANTAGE = 1.2   # 兵力占优
_AAM_RATIO_WORSE = 0.7       # 兵力劣势
_ALIVE_FRACTION_WORSE = 0.4  # 存活比劣势


class AirFormationAI:
    """空中编队 AI：意图 FSM + 贪心目标分配 + 编队飞行（docs §五）。

    决策主体：只写编队意图与成员 `order/target_idx/target_kind`，
    不直接改运动态（单位 AI + 飞控落地）。
    """

    def __init__(
        self,
        aircraft: Any,
        formations: Any,
        aero: Any,
        armor: Any | None = None,
        rng: Any | None = None,
        los_checker: Any | None = None,
        # ---- Phase 8 协同规避（docs §7.3）----
        flight: Any | None = None,      # FlightControl（机动原语）
        emitters: Any | None = None,    # EmitterPool（协同诱饵）
    ) -> None:
        self.aircraft = aircraft
        self.formations = formations
        self.aero = aero
        self.armor = armor
        self.rng = rng
        self.los_checker = los_checker
        self.flight = flight
        self.emitters = emitters
        self._contacts: list[list[AirContact]] = [[] for _ in range(aircraft.capacity)]
        # 编队级命令（玩家/上级注入）：fidx → 意图覆盖（None = 按 FSM 自主）
        self.commanded_intent: dict[int, int | None] = {}
        # 自由开火令（weapons_free）：fidx → bool（玩家/上级注入）
        self.weapons_free: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # 主循环（每 ai tick 一次）
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """刷新接触 → 逐编队 FSM/分配/队形。"""
        # 1. 机载雷达搜索（按 side 去重）
        if self.armor is not None:
            sides: set[int] = set()
            for f in self.formations.active_indices():
                sides.add(int(self.formations.side[f]))
            for s in sides:
                gather_air_contacts(
                    s, self.aircraft, self.armor, self._contacts,
                    rng=self.rng, los_checker=self.los_checker,
                )
        # 2. 逐编队决策
        for f in self.formations.active_indices():
            f = int(f)
            sit = self.situation_of(f)
            self.update_intent(f, sit)
            self.assign_targets(f)
            self.execute_formation_geometry(f)
            self._tick_coordinated_evasion(f, dt)   # Phase 8 协同规避（docs §7.3）

    def _tick_coordinated_evasion(self, fidx: int, dt: float) -> None:
        """编队级协同规避（docs §7.3）：RWR 锁定告警 → 协同诱饵 + 机动脱离。

        - 威胁成员（rwr_state==3，来袭 AAM/SAM 锁定）：编队选「诱饵机」
          （弹药耗尽者优先——牺牲优先级最低，docs §7.3）抛热焰吸引导引头；
        - 其余受威胁成员 `kinematic_defeat` 急转弯下降脱离；
        - 数据链共享的威胁航迹 = 全体成员 rwr_state 汇总（本机 RWR 已含
          data link 告警——`RWRDetector` 泛化读 EmitterPool）。
        """
        a = self.aircraft
        members = self._members(fidx)
        if not members or self.flight is None:
            return
        threatened = [m for m in members if int(a.rwr_state[m]) == 3]
        if not threatened:
            return
        # 诱饵机：弹药耗尽者优先（无弹可打 → 当诱饵），否则取首个受威胁成员
        decoy = None
        for m in members:
            if int(a.aam_left[m]) == 0 and int(a.agm_left[m]) == 0:
                decoy = m
                break
        if decoy is None:
            decoy = threatened[0]
        # 诱饵抛热焰（高功率瞬时 IR 源，_find_ir_hot 锁最热源 → 吸引导引头）
        if self.emitters is not None:
            self.emitters.deploy_decoy(
                float(a.pos[decoy, 0]), float(a.pos[decoy, 1]), float(a.pos[decoy, 2]),
                band=int(EmitterBand.IR), power=800.0,
            )
        # 其余受威胁成员机动脱离
        for m in threatened:
            if m != decoy:
                self.flight.kinematic_defeat(a, m, dir_sign=1.0)

    # ------------------------------------------------------------------
    # 态势评估（§5.1 步骤 2）
    # ------------------------------------------------------------------

    def _members(self, fidx: int) -> list[int]:
        """编队存活成员（按索引升序；长机见 `_leader_idx`）。"""
        a = self.aircraft
        return [
            int(i) for i in a.active_indices()
            if int(a.formation_id[i]) == fidx
            and int(a.state[i]) not in (_STATE_CRASHING, _STATE_PARKED)
        ]

    def _leader_idx(self, fidx: int) -> int:
        """编队长机 aircraft 索引（未登记 → 回退最小索引成员）。"""
        lidx = int(self.formations.leader_idx[fidx])
        if lidx >= 0:
            return lidx
        members = self._members(fidx)
        return members[0] if members else -1

    def situation_of(self, fidx: int) -> dict:
        """兵力比（AAM 剩余 + 存活）、敌接触概览。"""
        a = self.aircraft
        members = self._members(fidx)
        alive = len(members)
        aam_total = int(sum(int(a.aam_left[m]) for m in members))
        # 汇总本编队全部接触（去重按 (kind, idx)）
        seen: dict[tuple[int, int], AirContact] = {}
        for m in members:
            for c in self._contacts[m]:
                key = (c.tgt_kind, c.tgt_idx)
                if key not in seen or c.dist < seen[key].dist:
                    seen[key] = c
        contacts = list(seen.values())
        n_air = sum(1 for c in contacts if c.tgt_kind == TGT_AIR)
        n_ground = sum(1 for c in contacts if c.tgt_kind == TGT_GROUND)
        return {
            "members": members,
            "alive": alive,
            "total": len(members),
            "aam_total": aam_total,
            "contacts": contacts,
            "n_air": n_air,
            "n_ground": n_ground,
        }

    # ------------------------------------------------------------------
    # 意图 FSM（§5.2 迁移表）
    # ------------------------------------------------------------------

    def update_intent(self, fidx: int, sit: dict) -> None:
        f = self.formations
        cur = int(f.intent[fidx])
        a = self.aircraft
        # 玩家/上级命令覆盖（STRIKE/SEAD/ESCORT/RTB）
        cmd = self.commanded_intent.get(fidx)
        if cmd is not None:
            if cmd != cur:
                f.intent[fidx] = cmd
                self.commanded_intent[fidx] = None   # 一次性命令
            return

        n_air = sit["n_air"]
        aam = sit["aam_total"]
        alive = sit["alive"]

        if cur == int(AirFormationIntent.CAP):
            if n_air > 0:
                # 劣势检查优先于交战（无论是否自由开火）
                if alive > 0 and aam / max(alive, 1) < _AAM_RATIO_WORSE:
                    f.intent[fidx] = int(AirFormationIntent.RTB)
                elif alive < _ALIVE_FRACTION_WORSE * sit["total"]:
                    f.intent[fidx] = int(AirFormationIntent.RTB)
                elif self.weapons_free.get(fidx, False):
                    pass  # 接敌且自由开火 → CAP 交战中（单位 AI 自主 OODA）
        elif cur == int(AirFormationIntent.STRIKE):
            if all(int(a.agm_left[m]) == 0 for m in sit["members"]):
                f.intent[fidx] = int(AirFormationIntent.RTB)
        elif cur == int(AirFormationIntent.SEAD):
            if n_air == 0 and n_ground == 0 or all(
                int(a.agm_left[m]) == 0 for m in sit["members"]
            ):
                f.intent[fidx] = int(AirFormationIntent.RTB)
        elif cur == int(AirFormationIntent.ESCORT):
            if aam == 0:
                f.intent[fidx] = int(AirFormationIntent.RTB)
        elif cur == int(AirFormationIntent.RTB):
            pass  # 到达基地空域 → 分组降落（步骤⑨）

    # ------------------------------------------------------------------
    # 目标分配（§5.3 贪心两轮）
    # ------------------------------------------------------------------

    def assign_targets(self, fidx: int) -> None:
        a = self.aircraft
        sit = self.situation_of(fidx)
        members = sit["members"]
        intent = int(self.formations.intent[fidx])
        contacts = sit["contacts"]
        air_targets = [c for c in contacts if c.tgt_kind == TGT_AIR]
        ground_targets = [c for c in contacts if c.tgt_kind == TGT_GROUND]

        # 交战授权：CAP 需自由开火令；STRIKE/SEAD/ESCORT 任务编队自带授权
        weapons_free = self.weapons_free.get(fidx, False)
        if not weapons_free and intent == int(AirFormationIntent.CAP):
            air_targets = []

        # 先清空分配
        for m in members:
            a.order[m] = int(AirUnitOrder.HOLD)
            a.target_idx[m] = -1

        # ---- 覆盖轮：每个敌接触指派 1 个「AAM 最多且未指派」成员 ----
        assigned: set[int] = set()
        for t in air_targets:
            cand = [m for m in members if m not in assigned]
            if not cand:
                break
            m = max(cand, key=lambda x: int(a.aam_left[x]))
            assigned.add(m)
            a.order[m] = int(AirUnitOrder.ENGAGE_AIR)
            a.target_idx[m] = t.tgt_idx
            a.target_kind[m] = TGT_AIR

        # ---- 并发轮：剩余成员集中火力打最高威胁目标（§5.3）----
        remaining = [m for m in members if m not in assigned]
        if remaining and air_targets:
            scored = self._score_target_list(fidx, air_targets)
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[0][1]
            for m in remaining:
                a.order[m] = int(AirUnitOrder.ENGAGE_AIR)
                a.target_idx[m] = top.tgt_idx
                a.target_kind[m] = TGT_AIR

        # ---- 对地（STRIKE/SEAD）----
        if intent in (int(AirFormationIntent.STRIKE), int(AirFormationIntent.SEAD)):
            unassigned = [m for m in members if int(a.order[m]) == int(AirUnitOrder.HOLD)]
            for m, t in zip(unassigned, ground_targets):
                a.order[m] = int(AirUnitOrder.STRIKE_GROUND)
                a.target_idx[m] = t.tgt_idx
                a.target_kind[m] = TGT_GROUND
            # 没有地面目标 → 巡航到目标区（STRIKE 无目标时保持 HOLD→巡航）

    def _score_target_list(
        self, fidx: int, targets: list[AirContact],
    ) -> list[tuple[float, AirContact]]:
        """用 score_air_targets 内核给目标清单打分（成员=长机参考点）。"""
        a = self.aircraft
        members = self._members(fidx)
        if not members:
            return [(0.0, t) for t in targets]
        ref = self._leader_idx(fidx)
        rx, ry, rz = a.pos[ref]
        k = len(targets)
        dist = np.zeros(k, np.float32)
        d_alt = np.zeros(k, np.float32)
        health = np.zeros(k, np.float32)
        los_x = np.zeros(k, np.float32)
        los_y = np.zeros(k, np.float32)
        los_z = np.zeros(k, np.float32)
        tvx = np.zeros(k, np.float32)
        tvy = np.zeros(k, np.float32)
        tvz = np.zeros(k, np.float32)
        max_r = 1.0
        for i, t in enumerate(targets):
            if t.tgt_kind == TGT_AIR:
                tx, ty, tz = a.pos[t.tgt_idx]
                tvx[i], tvy[i], tvz[i] = a.vel[t.tgt_idx]
                health[i] = t.health / 100.0
            else:
                tx, ty, tz = self.armor.pos[t.tgt_idx] if self.armor is not None else (0, 0, 0)
                tvx[i], tvy[i], tvz[i] = (0.0, 0.0, 0.0)
                health[i] = 0.5
            dx = tx - rx
            dy = ty - ry
            dz = tz - rz
            dist[i] = math.sqrt(dx * dx + dy * dy + dz * dz)
            d_alt[i] = rz - tz
            inv = 1.0 / max(dist[i], 1e-6)
            los_x[i] = dx * inv
            los_y[i] = dy * inv
            los_z[i] = dz * inv
            max_r = max(max_r, dist[i])
        out = np.zeros(k, np.float32)
        score_air_targets(
            dist, d_alt, health, los_x, los_y, los_z,
            tvx, tvy, tvz, max_r,
            1.0, 0.5, 1.5, 1.0, out,
        )
        return [(float(out[i]), targets[i]) for i in range(k)]

    # ------------------------------------------------------------------
    # 编队飞行（§5.4：僚机追长机实时位置）
    # ------------------------------------------------------------------

    def execute_formation_geometry(self, fidx: int) -> None:
        members = self._members(fidx)
        if len(members) < 2:
            return
        a = self.aircraft
        leader = self._leader_idx(fidx)
        if leader < 0:
            return
        intent = int(self.formations.intent[fidx])
        table = _INTENT_SLOTS.get(intent, _SLOT_WEDGE)
        wingmen = [m for m in members if m != leader]
        for k, m in enumerate(wingmen, start=1):
            off = table[min(k, len(table) - 1)]
            a.set_waypoints(m, [(
                float(a.pos[leader, 0]) + off[0],
                float(a.pos[leader, 1]) + off[1],
                float(a.pos[leader, 2]),
            )])

    # ------------------------------------------------------------------
    # 供上层/测试查询
    # ------------------------------------------------------------------

    def member_contacts(self, air_idx: int) -> list[AirContact]:
        return list(self._contacts[air_idx])


# ---------------------------------------------------------------------------
# 单位 AI（步骤⑥，docs §六）
# ---------------------------------------------------------------------------

# 空战参数（docs §6.2：BVR/WVR 简化）
_AAM_MAX_RANGE = 25000.0    # AAM 最大射程判定（进入占位追击）
_AAM_LAUNCH_RANGE = 18000.0  # AAM 发射距离（雷达锁定 + 射程内）
_FIRE_INTERVAL = 2.0        # 每架两次发射间隔 (s)
_STRIKE_ALT = 1600.0        # 对地投弹高度（低于此 → 拉起返航）


class AirUnitAI:
    """空中单位 AI：消费编队订单 → 机动原语 + fire_air（docs §六）。

    只写飞控输入（经 FlightControl）与开火（FiringControl.fire_air），
    不直改运动态（AI 只决策不执行，docs §〇.3）。
    """

    def __init__(
        self,
        aircraft: Any,
        aero: Any,
        flight: Any,
        firing: Any,
        armor: Any | None = None,
        aam_ammo_id: str = "aim120_amraam",
        agm_ammo_id: str = "gb_1000kg",
        # ---- Phase 8：数据链共享航迹发射（docs §7.2）----
        contact_confidence: Any | None = None,   # Callable[[target_idx], float]（None=本机雷达=1.0）
    ) -> None:
        self.aircraft = aircraft
        self.aero = aero
        self.flight = flight
        self.firing = firing
        self.armor = armor
        self.aam_ammo_id = aam_ammo_id
        self.agm_ammo_id = agm_ammo_id
        self._contact_confidence = contact_confidence
        self._fire_timer: dict[int, float] = {}

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """逐架消费订单（编队 AI 已写 order/target_idx/target_kind）。"""
        a = self.aircraft
        for k in list(self._fire_timer):
            self._fire_timer[k] = max(0.0, self._fire_timer[k] - dt)
        for i in a.active_indices():
            i = int(i)
            st = int(a.state[i])
            if st in (_STATE_CRASHING, _STATE_PARKED, _STATE_LANDING):
                continue
            order = int(a.order[i])
            if order == int(AirUnitOrder.ENGAGE_AIR):
                self._engage_air(i)
            elif order == int(AirUnitOrder.STRIKE_GROUND):
                self._strike_ground(i)
            elif order == int(AirUnitOrder.SUPPRESS):
                self._strike_ground(i)   # 压制 = 对地攻击（SEAD 打雷达/防空）
            elif order == int(AirUnitOrder.RTB):
                self._rtb(i)
            elif order == int(AirUnitOrder.LAND):
                pass   # 降落由起降流程接管（步骤⑨）
            else:
                self._cap(i)   # HOLD / CAP：巡航监视

    # ------------------------------------------------------------------
    # 空战（docs §6.2 OODA：发现→判定→攻击→战果→规避→返航）
    # ------------------------------------------------------------------

    def _engage_air(self, i: int) -> None:
        a = self.aircraft
        t = int(a.target_idx[i])
        if t < 0 or not a.active[t] or int(a.state[t]) in (_STATE_CRASHING, _STATE_PARKED):
            a.order[i] = int(AirUnitOrder.CAP)   # 目标失效 → 回巡逻
            a.target_idx[i] = -1
            return
        tpos = a.pos[t]
        tvel = a.vel[t]
        d = math.sqrt(
            (float(a.pos[i, 0]) - float(tpos[0])) ** 2
            + (float(a.pos[i, 1]) - float(tpos[1])) ** 2
            + (float(a.pos[i, 2]) - float(tpos[2])) ** 2
        )
        # 占位机动（lead pursuit：指向拦截点）
        self.flight.lead_pursuit(a, i, tpos, tvel)
        # 发射：射程内 + 有弹 + 冷却就绪
        if d < _AAM_LAUNCH_RANGE and int(a.aam_left[i]) > 0 \
                and self._fire_timer.get(i, 0.0) <= 0.0:
            if self.firing is not None:
                # Phase 8：数据链共享航迹发射（confidence 折扣进散布，docs §7.2）
                conf = 1.0
                if self._contact_confidence is not None:
                    conf = float(self._contact_confidence(t))
                self.firing.fire_air(
                    i, tpos, ammo_id=self.aam_ammo_id,
                    target_kind=0, target_idx=t, confidence=conf,
                )
                self._fire_timer[i] = _FIRE_INTERVAL
        # 燃油/弹药强制返航（docs §6.2 最高优先）
        if float(a.fuel[i]) < float(a.bingo[i]) or int(a.aam_left[i]) == 0:
            a.order[i] = int(AirUnitOrder.RTB)

    # ------------------------------------------------------------------
    # 对地（STRIKE/SUPPRESS，docs §六.2 俯冲-投弹-拉起）
    # ------------------------------------------------------------------

    def _strike_ground(self, i: int) -> None:
        a = self.aircraft
        t = int(a.target_idx[i])
        if self.armor is None or t < 0 or not self.armor.active[t]:
            a.order[i] = int(AirUnitOrder.RTB)
            return
        tpos = self.armor.pos[t]
        # 俯冲投弹 → 投后拉起返航
        if float(a.pos[i, 2]) > _STRIKE_ALT:
            self.flight.dive(a, i)
            d = math.sqrt(
                (float(a.pos[i, 0]) - float(tpos[0])) ** 2
                + (float(a.pos[i, 1]) - float(tpos[1])) ** 2
            )
            if d < 3000.0 and int(a.agm_left[i]) > 0 \
                    and self._fire_timer.get(i, 0.0) <= 0.0:
                if self.firing is not None:
                    self.firing.fire_air(
                        i, tpos, ammo_id=self.agm_ammo_id,
                        target_kind=1, target_idx=t,
                    )
                    self._fire_timer[i] = _FIRE_INTERVAL
        else:
            self.flight.pullup(a, i)
            a.order[i] = int(AirUnitOrder.RTB)   # 投弹完成 → 返航

    # ------------------------------------------------------------------
    # 返航 / 巡逻
    # ------------------------------------------------------------------

    def _rtb(self, i: int) -> None:
        """返航：飞控追基地航路点（wp 由上层/场景设置），低空慢速。"""
        a = self.aircraft
        a.cruise_alt[i] = 2000.0
        a.cruise_spd[i] = 200.0
        if int(a.wp_count[i]) == 0:
            # 无航路点（测试/简化）：盘旋等待
            pass

    def _cap(self, i: int) -> None:
        """CAP 巡逻：定高定航巡航（飞控 cruise_control 默认行为）。"""
        pass


# ---------------------------------------------------------------------------
# 巡飞弹侦察上报（步骤⑦，docs §5.4）
# ---------------------------------------------------------------------------


class LoiterRecon:
    """巡飞弹侦察上报（docs §5.4）：LOITER 弹丸把附近敌方真值上报孪生。

    - 巡飞弹（`ProjectileType.LOITER`）= 可消耗无人机：LOITER 期间扫描
      scan_radius 内敌方装甲 → `TwinEngine.observe`（source=1 本队观测，
      confidence 由观测质量定）；
    - 被击落/燃料耗尽 → 弹丸销毁 → 上报自然中断 → 孪生无更新 → 衰减回
      UNKNOWN（战争迷雾「无人机没了，情报就断了」）。
    """

    def __init__(
        self,
        twin_engine: Any,
        armor: Any,
        projectiles: Any,
        *,
        side: int = 0,
        scan_radius: float = 300.0,
        confidence: float = 0.85,
    ) -> None:
        self.twin_engine = twin_engine
        self.armor = armor
        self.projectiles = projectiles
        self.side = int(side)
        self.scan_radius = float(scan_radius)
        self.confidence = float(confidence)

    def tick(self, dt: float) -> None:
        p = self.projectiles
        for proj in p.active_indices():
            proj = int(proj)
            if int(p.kind[proj]) != int(ProjectileType.LOITER):
                continue
            px = float(p.pos[proj, 0])
            py = float(p.pos[proj, 1])
            pz = float(p.pos[proj, 2])
            a = self.armor
            r2 = self.scan_radius * self.scan_radius
            for i in a.active_indices():
                i = int(i)
                if int(a.side[i]) == self.side:
                    continue   # 只上报敌方
                dx = float(a.pos[i, 0]) - px
                dy = float(a.pos[i, 1]) - py
                dz = float(a.pos[i, 2]) - pz
                if dx * dx + dy * dy + dz * dz > r2:
                    continue
                # 上报孪生（source=1 本队观测，TV 制导图像筛选目标）
                self.twin_engine.observe(
                    TWIN_KIND_ARMOR, i, a.pos[i], a.vel[i],
                    confidence=self.confidence, observer_comm=1.0,
                )


# ---------------------------------------------------------------------------
# 起降 / 基地（步骤⑨，docs §九）
# ---------------------------------------------------------------------------


class AirbaseConfig:
    """基地配置（docs §九：闭环但简化，不建模跑道物理）。

    x/y/z: 跑道位置（z = 跑道高度，多为 0）；run_heading_deg: 跑道航向；
    radius: 基地空域半径（RTB 到达判定）。
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        run_heading_deg: float = 0.0,
        radius: float = 2000.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.run_heading_deg = float(run_heading_deg)
        self.radius = float(radius)

    @property
    def run_heading(self) -> float:
        return math.radians(self.run_heading_deg)


class AirfieldOps:
    """起降闭环（docs §九）：起飞（PARKED→TAKOFF→CRUISE）+ 降落（RTB→LANDING→PARKED）。

    - 起飞：航向对齐跑道 + 满油门加速（FlightControl.takeoff_control）→ 离地爬升 → CRUISE。
    - 降落（分组）：RTB 到基地空域 → 依到达顺序排队 LANDING（梯队间隔由上层控制）→
      程序化进近（landing_control 降高度减速）→ 触地 → 滑跑 → PARKED。
    - 基地抽象：跑道 = 配置坐标，不建模跑道物理（docs §九「基地抽象」）。
    """

    def __init__(self, aircraft: Any) -> None:
        self.aircraft = aircraft
        self.bases: list[AirbaseConfig] = []
        self._takeoff_queue: list[int] = []
        self._landing_queue: list[int] = []

    # ------------------------------------------------------------------
    # 注册 / 查询
    # ------------------------------------------------------------------

    def add_base(self, base: AirbaseConfig) -> None:
        self.bases.append(base)

    def nearest_base(self, x: float, y: float) -> AirbaseConfig | None:
        best = None
        best_d = 1e18
        for b in self.bases:
            d = (b.x - x) ** 2 + (b.y - y) ** 2
            if d < best_d:
                best_d = d
                best = b
        return best

    def in_airspace(self, air_idx: int, base: AirbaseConfig) -> bool:
        a = self.aircraft
        dx = float(a.pos[air_idx, 0]) - base.x
        dy = float(a.pos[air_idx, 1]) - base.y
        return dx * dx + dy * dy <= base.radius * base.radius

    # ------------------------------------------------------------------
    # 起飞
    # ------------------------------------------------------------------

    def request_takeoff(self, air_idx: int, base: AirbaseConfig) -> None:
        """PARKED → TAKEOFF：对齐跑道航向，航路点 = 跑道外 + 巡航高度。"""
        a = self.aircraft
        a.state[air_idx] = _STATE_TAKEOFF
        a.heading[air_idx] = base.run_heading
        a.cruise_alt[air_idx] = max(float(a.cruise_alt[air_idx]), 1500.0)
        h = base.run_heading
        a.set_waypoints(air_idx, [(
            base.x + 3000.0 * math.cos(h),
            base.y + 3000.0 * math.sin(h),
            float(a.cruise_alt[air_idx]),
        )])

    # ------------------------------------------------------------------
    # 降落（分组）
    # ------------------------------------------------------------------

    def request_landing(self, air_idx: int, base: AirbaseConfig) -> None:
        """RTB 到基地空域 → LANDING：进近航路点 = 跑道中心前方 + 进近高度。

        依到达顺序排队（分组降落：编队成员逐个进近，梯队间隔由
        `spacing` 控制——AI 层每 dt 放行一个）。
        """
        a = self.aircraft
        if air_idx not in self._landing_queue:
            self._landing_queue.append(air_idx)
        h = base.run_heading
        a.set_waypoints(air_idx, [(
            base.x + 400.0 * math.cos(h),
            base.y + 400.0 * math.sin(h),
            max(float(base.z) + 300.0, 300.0),
        )])
        a.cruise_alt[air_idx] = max(float(base.z) + 300.0, 300.0)
        a.cruise_spd[air_idx] = 150.0
        a.state[air_idx] = _STATE_LANDING

    # ------------------------------------------------------------------
    # 每帧状态机推进
    # ------------------------------------------------------------------

    def tick(self, dt: float, release_interval: float = 2.0) -> None:
        """推进起降状态：起飞离地 → CRUISE；降落触地 → PARKED。

        Args:
            dt: 帧时长。
            release_interval: 排队放行间隔（分组降落梯队间隔，s）。
        """
        a = self.aircraft

        # 起飞：离地（z 高于跑道 + 50m）且速度达标 → CRUISE
        for i in list(self._takeoff_queue):
            if not a.active[i]:
                self._takeoff_queue.remove(i)
                continue
            v = np.linalg.norm(a.vel[i])
            if float(a.pos[i, 2]) > 50.0 and v > 60.0:
                a.state[i] = _STATE_CRUISE
                self._takeoff_queue.remove(i)

        # 降落：进近保持 300m → 跑道附近拉平下压 → 触地驻停
        for i in list(self._landing_queue):
            if not a.active[i]:
                self._landing_queue.remove(i)
                continue
            base = self.nearest_base(float(a.pos[i, 0]), float(a.pos[i, 1]))
            ground = base.z if base is not None else 0.0
            if base is not None:
                dx = float(a.pos[i, 0]) - base.x
                dy = float(a.pos[i, 1]) - base.y
                if dx * dx + dy * dy < 200.0 * 200.0:
                    a.gamma_cmd[i] = -0.25   # 拉平下压触地
                    a.cruise_alt[i] = 0.0    # 取消高度保持（下压接管）
            if float(a.pos[i, 2]) <= ground + 2.0:
                a.vel[i] = 0.0
                a.state[i] = _STATE_PARKED
                self._landing_queue.remove(i)

    def takeoff_in_queue(self, air_idx: int) -> None:
        if air_idx not in self._takeoff_queue:
            self._takeoff_queue.append(air_idx)


# ---------------------------------------------------------------------------
# 地面防空激活（步骤⑩，docs §十）
# ---------------------------------------------------------------------------


class AirDefenseController:
    """地面防空（role=3，UnitRole.AA）激活：检测空中目标 → 发射 SAM（docs §十）。

    复用既有 `FiringControl.fire()`（射手=装甲）+ 制导弹丸：
    SAM 追飞机只需把 `aircraft.pos` 写 `guid_target_pos`、`aircraft.vel` 写
    `guid_target_vel`、`guid_target_kind=1`（docs §7.2 已支持，零内核改动）。
    检测 = 距离 + 高度（防空雷达射程判定，非上帝视角简化——Phase 8 换孪生接触源）。

    空中对抗 SAM 闭环（docs §十末）：SEAD 单位 RWR 告警 → 规避机动 + 反辐射弹
    （`AirUnitAI.STRIKE_GROUND` 已可打防空单位本体）。
    """

    def __init__(
        self,
        armor: Any,
        aircraft: Any,
        firing: Any,
        *,
        sam_ammo_id: str = "sam",
        max_range_m: float = 12000.0,
        min_alt_m: float = 50.0,
    ) -> None:
        self.armor = armor
        self.aircraft = aircraft
        self.firing = firing
        self.sam_ammo_id = sam_ammo_id
        self.max_range_m = float(max_range_m)
        self.min_alt_m = float(min_alt_m)
        self._cooldown: dict[int, float] = {}

    def tick(self, dt: float) -> None:
        """逐防空单位：射程内敌机 → 发射一发 SAM。"""
        for k in list(self._cooldown):
            self._cooldown[k] = max(0.0, self._cooldown[k] - dt)
        a = self.armor
        for aa in a.active_indices():
            aa = int(aa)
            if int(a.role[aa]) != 3:   # UnitRole.AA
                continue
            if self._cooldown.get(aa, 0.0) > 0.0:
                continue
            ax = float(a.pos[aa, 0])
            ay = float(a.pos[aa, 1])
            az = float(a.pos[aa, 2])
            aside = int(a.side[aa])
            for foe in self.aircraft.active_indices():
                foe = int(foe)
                if int(self.aircraft.side[foe]) == aside:
                    continue
                if int(self.aircraft.state[foe]) in (_STATE_CRASHING, _STATE_PARKED):
                    continue
                fx = float(self.aircraft.pos[foe, 0])
                fy = float(self.aircraft.pos[foe, 1])
                fz = float(self.aircraft.pos[foe, 2])
                if fz < self.min_alt_m:
                    continue
                d = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2 + (fz - az) ** 2)
                if d > self.max_range_m:
                    continue
                rec = self.firing.fire(aa, -1, aim_at=self.aircraft.pos[foe],
                                       ammo_id=self.sam_ammo_id)
                if rec is not None:
                    # SAM 制导字段：目标 = 飞机（kind=1 + vel 前置量，docs §7.2）
                    p = self.firing.projectiles
                    p.guid_target_kind[rec.proj_idx] = 1
                    p.guid_target_vel[rec.proj_idx] = self.aircraft.vel[foe]
                    self._cooldown[aa] = 5.0   # 单发间隔（防空导弹装填）
                break   # 每防空单位每次一发

