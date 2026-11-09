"""编队/单位 AI（Phase 6c 步骤 ②，docs/phase6-ai-design.md §二/§三）。

编队 AI 是**决策主体**（意图 FSM + 指挥链 + 目标分配），单位 AI 是**执行者**
（只执行编队下达的 `order`/`target_idx`，不做独立决策）。

三条红线（docs §〇）：
    1. 编队 AI 决策，单位 AI 执行——编队写成员 `order`/`target_idx`，单位只执行。
    2. AI 只决策，不执行——通过既有接口（`FiringControl.fire` / `ArmorPool.wp`）
       行动，**绝不直接改 SoA 数组**（与叙事层「永不直接修改模拟状态」同构）。
    3. 热路径只做向量化批量评估——目标评分/威胁排序用 `score_targets`（@njit）；
       编队 FSM 是纯 Python 逐编队遍历（≤256 编队 × 简单分支 < 0.1ms）。

接触来源（docs §〇.3）：Phase 6c 用简化「全知接触」接口（默认从 armor 池读敌方），
Phase 8 战争迷雾接入后换成孪生数据。

依赖方向（AGENTS.md 4.1）：simulation → combat/movement/entities（严格向下）；
combat 不反向依赖本模块。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
from numba import njit

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.detection.sensors import esm_detection_range
from Airia.lulilo.entities.pools import (
    EMITTER_KIND_ILLUMINATOR,
    EMITTER_KIND_PLATFORM_RADAR,
    EMITTER_KIND_SEEKER,
    EMITTER_PARENT_AIRCRAFT,
    EMITTER_PARENT_ARMOR,
    EMITTER_PARENT_PROJECTILE,
    ArmorPool,
    FormationAIPool,
    PartType,
)
from Airia.lulilo.entities.side import Side, hostile
from Airia.lulilo.movement.pathfinding import GraphPathfinder
from Airia.lulilo.simulation.commands import Command, CommandType
from Airia.lulilo.simulation.loop import SimulationLoop

logger = get_logger(__name__)

# 默认初始血量（spawn 默认 100.0；无配置时的态势健康基准）
_DEFAULT_MAX_HEALTH = 100.0


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class FormationIntent(IntEnum):
    """编队意图 FSM（docs §二.1）。"""
    MANEUVER = 0   # 机动（向任务目标区行进，途中遭遇才交战）
    ADVANCE = 1    # 进攻（主动推进 + 分配打击任务）
    DEFEND = 2     # 防御（固守，只对进入射程目标分配射击）
    RETREAT = 3    # 撤退（脱离接触，向安全区机动）
    HOLD = 4       # 待命（静默，辐射管制）


class UnitOrder(IntEnum):
    """编队下达的执行指令（写成员 `order`，docs §三.1）。"""
    ENGAGE = 0     # 进攻：向 target_idx 机动 + 自由射击
    DEFEND = 1     # 防御：固守阵位，只对进入射程的分配目标射击
    MANEUVER = 2   # 机动：向编队 target_zone 寻路移动
    RETREAT = 3    # 撤退：脱离接触，向安全区机动
    HOLD = 4       # 待命：静默（不移动不射击，辐射管制）


class UnitRole(IntEnum):
    """单位/编队角色（ArmorPool.role / FormationAIPool.role，docs §1.1）。"""
    TANK = 0             # 坦克
    ARTILLERY = 1        # 炮兵（曲射，Phase 6c 步骤⑤）
    SCOUT = 2            # 侦察
    AA = 3               # 防空
    TOP_ATTACK_ATGM = 4  # 攻顶 ATGM 发射器（步骤④ 延后）
    AIR = 5              # 空中（Phase 7）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Contact:
    """一条敌接触（编队 AI 汇总用，Python 层非热路径）。

    speed / weapon_pen 为威胁评分输入（Phase 6c 恒 0，Phase 7/8 细化）。
    Phase 8：confidence / pos_err 战争迷雾字段（孪生接触源注入，docs §5.1）。
    """
    idx: int          # 装甲池索引
    side: int         # 绝对阵营（Side 枚举）
    x: float
    y: float
    health: float
    speed: float = 0.0
    weapon_pen: float = 0.0
    kind: int = 0     # 目标类型：0=装甲 1=辐射源 2=光斑 3=地形点
    confidence: float = 1.0   # Phase 8：接触置信度 0~1（迷雾；1=全知模型）
    pos_err: float = 0.0      # Phase 8：位置误差半径 (m)
    twin_slot: int = -1       # Phase 8：孪生槽位（支援请求回链）


@dataclass
class FormationSituation:
    """编队态势（FSM 转移输入，docs §二.1）。"""
    friendly_active: int = 0   # 本编队存活成员数
    enemy_contacts: int = 0    # 汇总的敌接触数
    enemy_known: int = 0       # 已知敌总数（情报）
    ammo_ratio: float = 1.0    # 编队平均弹药 0~1
    health_ratio: float = 1.0  # 编队平均血量 0~1
    under_artillery: bool = False
    under_rwr: bool = False
    task_reached: bool = False


@dataclass
class ArtilleryMission:
    """一条炮兵火力任务（Phase 6c 步骤⑤，docs/phase6-ai-pipeline.md §六.2）。

    Command.params 表达 coords / ammo / target_kind / rounds（万物皆单位，
    不建 FireMission 数据结构）。priority：0=友军请求 1=玩家指令（玩家覆盖
    编队 AI 决策，§六.2「优先级最高」）。
    """
    coords: tuple[float, float]   # 落点世界坐标 (x, y)
    ammo: str | None = None       # 弹药 id（None 用单位配置默认）
    rounds: int = 1               # 齐射数（每炮位每 tick 一发，多炮位分摊）
    priority: int = 0             # 0 友军请求 / 1 玩家指令（高者先执行）
    target_idx: int = -1          # 关联目标装甲索引（-1 纯地形点）


@dataclass
class MuzzleFlash:
    """一条炮口闪光记录（反炮暴露源，Phase 6c 步骤⑥，docs §六.3）。

    炮兵开火 → 炮口闪光暴露阵地。这里用 Python 侧日志（而非 SoA EmitterPool 红外
    emitter）建模「开火暴露」——本质是「敌方侦察可检测到我的发射位置」，寿命短
    （max_age_s 内可被反炮），无需物理红外残热模型。
    """
    side: int      # 开火方绝对阵营
    x: float       # 炮口世界坐标
    y: float
    z: float
    born: float = 0.0  # 登记时的共享时钟（MuzzleFlashLog._now）
    id: int = 0    # 单调递增 id（反炮去重：每闪光只反一次）


class MuzzleFlashLog:
    """炮口闪光共享日志（反炮闭环的暴露源，docs §六.3「开火→暴露→反炮→转移」）。

    双方炮兵编队 AI 共享同一实例：开火方 `add()`，敌方 `enemy_flashes()` 检测后
    反炮。时钟（`tick(dt)`）由**单点**（场景装配器/测试循环）驱动一次/帧，避免
    双方 AI 各 tick 一次导致闪光双重老化；`enemy_flashes` 惰性修剪过期记录。
    """

    def __init__(self, max_age_s: float = 30.0) -> None:
        self.max_age_s = float(max_age_s)
        self._flashes: list[MuzzleFlash] = []
        self._next_id = 0
        self._now = 0.0

    def add(self, side: int, x: float, y: float, z: float = 0.0) -> MuzzleFlash:
        """登记一条炮口闪光（开火暴露）。"""
        f = MuzzleFlash(
            side=int(side),
            x=float(x),
            y=float(y),
            z=float(z),
            born=self._now,
            id=self._next_id,
        )
        self._next_id += 1
        self._flashes.append(f)
        return f

    def tick(self, dt: float) -> None:
        """推进共享时钟并丢弃超过 max_age_s 的记录（单点驱动一次/帧）。"""
        self._now += float(dt)
        self._prune()

    def _prune(self) -> None:
        self._flashes = [
            f for f in self._flashes if self._now - f.born < self.max_age_s
        ]

    def enemy_flashes(self, side: int) -> list[MuzzleFlash]:
        """返回仍活跃的敌方闪光（hostile(side, f.side)），惰性修剪过期记录。"""
        self._prune()
        return [f for f in self._flashes if hostile(side, f.side)]

    def __len__(self) -> int:
        return len(self._flashes)


# ---------------------------------------------------------------------------
# 目标评分内核（Numba，向量化批量；供编队 AI 与自保复用，docs §三.4）
# ---------------------------------------------------------------------------


@njit(cache=True)
def score_targets(
    dist: np.ndarray,          # (K,) float32 距离
    speed: np.ndarray,         # (K,) float32 目标速度
    weapon_pen: np.ndarray,    # (K,) float32 敌武器威胁
    health: np.ndarray,        # (K,) float32 目标血量（残血优先补刀）
    confidence: np.ndarray,    # (K,) float32 接触置信度 0~1（Phase 8 迷雾折扣）
    w_range: float,
    w_speed: float,
    w_weapon: float,
    w_health: float,
    max_range: float,
    out: np.ndarray,           # (K,) float32 威胁分（原地写）
) -> None:
    """威胁评分：threat = (w_range·(1−d/max_range) + w_speed·speed
                            + w_weapon·weapon_pen + w_health·(1−health)) × confidence。

    距离越近越危险、敌方速度越快越危险、敌武器威胁越大越危险、
    残血目标优先补刀（易击杀）。Phase 8：**confidence 折扣**——看不清的目标
    不该被优先攻击（迷雾影响决策，docs §5.1）。
    """
    k = dist.shape[0]
    for i in range(k):
        out[i] = (
            w_range * (1.0 - dist[i] / max_range)
            + w_speed * speed[i]
            + w_weapon * weapon_pen[i]
            + w_health * (1.0 - health[i])
        ) * confidence[i]


# ---------------------------------------------------------------------------
# 孪生接触源（Phase 8，docs §5.1：contact_provider 换孪生源）
# ---------------------------------------------------------------------------


class TwinContactProvider:
    """孪生接触源：从认知孪生生成 Contact 列表（docs §5.1「换源」）。

    AI 不再读 Real——目标位置 = `twin.pos`（带误差）、血量 = `twin.health`
    （滞后）、confidence/pos_err 进入 Contact（评分折扣 + 支援请求回链）。
    KIA 目标不报（已确认击毁）；UNKNOWN 报模糊接触（低置信，评分折扣后自然低优先）。
    """

    def __init__(self, twin: Any, armor: ArmorPool, *, min_confidence: float = 0.0) -> None:
        self.twin = twin
        self.armor = armor
        self.min_confidence = float(min_confidence)

    def __call__(self, side: int) -> list[Contact]:
        """生成 side 阵营视角的敌接触（敌方孪生槽 → Contact）。"""
        from Airia.lulilo.entities.twin import TWIN_KIND_ARMOR, TwinStatus

        twin = self.twin
        armor = self.armor
        contacts: list[Contact] = []
        for s in twin.active_slots():
            s = int(s)
            kind = int(twin.unit_kind[s])
            idx = int(twin.twin_idx[s])
            if kind != TWIN_KIND_ARMOR:
                continue
            if not (0 <= idx < armor.capacity) or not armor.active[idx]:
                continue
            if not hostile(side, int(armor.side[idx])):
                continue
            if int(twin.status[s]) == int(TwinStatus.KIA):
                continue
            conf = float(twin.confidence[s])
            if conf < self.min_confidence:
                continue
            contacts.append(Contact(
                idx=idx,
                side=int(armor.side[idx]),
                x=float(twin.pos[s, 0]),
                y=float(twin.pos[s, 1]),
                health=float(twin.health[s]),
                confidence=conf,
                pos_err=float(twin.pos_err[s]),
                twin_slot=s,
            ))
        return contacts


# ---------------------------------------------------------------------------
# 通信网络（去中心化数据共享，docs §七；Phase 6c 步骤⑨）
# ---------------------------------------------------------------------------


class CommunicationNetwork:
    """通信簇：把 `comm` 抽象为去中心化系统参量——中央集群 vs 独立建网（docs §七）。

    Phase 6c 简化：以 comm 标量 + 阵营分组（不做点对点链路全图）；Phase 8 战争迷雾
    接入时换 networkx 连通分量 + Twin 同步（docs/phase8-datalink-design.md）。

    规则（docs §七.2/七.3）：
        - 编队 comm ≥ threshold → 加入中央集群（大面板可见、可协同、可收玩家指令）。
        - 编队 comm < threshold → 与中央断开，按阵营归入「独立网络」
          （内部仍可自主协同，但大面板不可见在干什么）。
    """

    def __init__(
        self,
        armor: ArmorPool,
        formations: FormationAIPool,
        *,
        comm_threshold: float = 0.5,
        central_side: int = Side.NATO,
    ) -> None:
        self.armor = armor
        self.formations = formations
        self.comm_threshold = float(comm_threshold)
        self.central_side = int(central_side)

    def is_central(self, fidx: int) -> bool:
        """编队是否在中央集群（comm ≥ 阈值）。"""
        if not self.formations.active[fidx]:
            return False
        return float(self.formations.comm[fidx]) >= self.comm_threshold

    def visible_formations(self) -> list[int]:
        """大面板可见的己方编队 = central_side 且 comm ≥ 阈值。"""
        return [
            int(i)
            for i in self.formations.active_indices()
            if int(self.formations.side[i]) == self.central_side and self.is_central(int(i))
        ]

    def hidden_formations(self) -> list[int]:
        """失联（大面板不可见）的己方编队 = central_side 且 comm < 阈值。"""
        return [
            int(i)
            for i in self.formations.active_indices()
            if int(self.formations.side[i]) == self.central_side and not self.is_central(int(i))
        ]

    def independent_networks(self) -> list[list[int]]:
        """独立建网分组：失联（comm < 阈值）的己方编队按阵营归组。

        Phase 6c 简化：己方失联编队无点对点链路模型，同阵营编队视为一个独立网络
        （内部可自主协同，大面板不可见）；Phase 8 换 networkx 连通分量（docs §七）。
        """
        groups: dict[int, list[int]] = {}
        for fidx in self.hidden_formations():
            groups.setdefault(int(self.formations.side[fidx]), []).append(fidx)
        return list(groups.values())

    def can_share_contacts(self, a_fidx: int, b_fidx: int) -> bool:
        """两编队是否同簇可共享接触（同中央集群，或同独立网络同阵营）。"""
        if self.is_central(a_fidx) and self.is_central(b_fidx):
            return True
        if (
            not self.is_central(a_fidx)
            and not self.is_central(b_fidx)
            and int(self.formations.side[a_fidx]) == int(self.formations.side[b_fidx])
        ):
            return True
        return False

    def can_receive_command(self, fidx: int) -> bool:
        """编队能否接收玩家/上级指令（须在中央集群）。"""
        return self.is_central(fidx)


# ---------------------------------------------------------------------------
# RWR 告警（复用 detection，不建 RWR 模块，docs §八.1；Phase 6c 步骤⑨）
# ---------------------------------------------------------------------------


class RWRDetector:
    """RWR / 激光告警：复用 `detection` 被动接收，判定「我被照射了吗」。

    docs §八「复用 detection，不建 RWR 模块」——本类是把 `EmitterPool` 的敌方
    RF emitter（平台雷达/主动导引头）与 LASER 照射器翻译成「告警」的薄封装，
    供编队 AI 消费（AI 决策不进 Numba 热路径，逐单位 Python 遍历即可）。

    - 射频告警：敌方 PLATFORM_RADAR / SEEKER 开机且距我方单位 ≤ ESM 探测距离。
    - 激光告警：敌方 ILLUMINATOR 照射（SAL 来袭前兆，直线可及）。
    """

    LASER_RANGE_M = 20000.0  # 激光直线可及（同 evaluate_passive_emitter_detection）

    def __init__(
        self,
        armor: ArmorPool,
        emitters: Any,
        projectiles: Any | None = None,
        *,
        events: Any | None = None,
        aircraft: Any | None = None,   # Phase 7：RWR 泛化到飞机（docs §十一.1）
    ) -> None:
        self.armor = armor
        self.emitters = emitters
        self.projectiles = projectiles
        self.events = events
        self.aircraft = aircraft

    def _emitter_side(self, e: int) -> int:
        """辐射源所属绝对阵营（装甲父 → 阵营；弹丸父 → owner 阵营；飞机父 → 阵营）。"""
        pk = int(self.emitters.parent_kind[e])
        pi = int(self.emitters.parent_idx[e])
        if pk == EMITTER_PARENT_ARMOR:
            if 0 <= pi < self.armor.capacity:
                return int(self.armor.side[pi])
            return -1
        if pk == EMITTER_PARENT_PROJECTILE and self.projectiles is not None:
            owner = int(self.projectiles.owner[pi])
            if 0 <= owner < self.armor.capacity:
                return int(self.armor.side[owner])
            # AAM owner 是飞机（Phase 7）：AAM 主动导引头暴露 → 触发敌方 RWR
            if self.aircraft is not None and 0 <= owner < self.aircraft.capacity:
                return int(self.aircraft.side[owner])
            return -1
        if pk == EMITTER_PARENT_AIRCRAFT and self.aircraft is not None:  # Phase 7
            if 0 <= pi < self.aircraft.capacity:
                return int(self.aircraft.side[pi])
            return -1
        return -1

    def tick(self, dt: float) -> list[dict[str, Any]]:
        """检测我方单位被敌方 RF / LASER 照射，返回告警列表（并 emit rwr_warning）。"""
        warnings: list[dict[str, Any]] = []
        a = self.armor
        for uid in a.active_indices():
            uid = int(uid)
            uside = int(a.side[uid])
            ux = float(a.pos[uid, 0])
            uy = float(a.pos[uid, 1])
            for e in self.emitters.active_indices():
                e = int(e)
                kind = int(self.emitters.kind[e])
                if kind not in (
                    EMITTER_KIND_PLATFORM_RADAR,
                    EMITTER_KIND_SEEKER,
                    EMITTER_KIND_ILLUMINATOR,
                ):
                    continue
                eside = self._emitter_side(e)
                if eside < 0 or not hostile(uside, eside):
                    continue
                ex = float(self.emitters.pos[e, 0])
                ey = float(self.emitters.pos[e, 1])
                d = math.hypot(ex - ux, ey - uy)
                rng = (
                    self.LASER_RANGE_M
                    if kind == EMITTER_KIND_ILLUMINATOR
                    else esm_detection_range(float(self.emitters.power_db[e]))
                )
                if d <= rng:
                    warnings.append(
                        {
                            "unit": uid,
                            "threat_kind": kind,
                            "band": int(self.emitters.band[e]),
                            "direction": (ex - ux, ey - uy),
                            "range": d,
                        }
                    )
                    break  # 每单位一个告警即可
        for w in warnings:
            if self.events is not None:
                self.events.emit("rwr_warning", dict(w))

        # Phase 7：飞机 RWR（docs §十一.1：泛化到飞机，写 aircraft.rwr_state）
        if self.aircraft is not None:
            self._aircraft_rwr()
        return warnings

    def _aircraft_rwr(self) -> None:
        """飞机 RWR：敌方 RF 辐射源（平台雷达/主动导引头）对飞机的告警分档。

        rwr_state：0 静默 / 1 搜索（敌方雷达开机但未锁定）/ 2 跟踪 / 3 锁定（来袭）。
        判定公式复用 `esm_detection_range`（零新公式，docs §十一.1）。
        """
        a = self.aircraft
        for uid in a.active_indices():
            uid = int(uid)
            if int(a.state[uid]) in (5, 6):   # CRASHING / PARKED 不告警
                a.rwr_state[uid] = 0
                continue
            uside = int(a.side[uid])
            ux = float(a.pos[uid, 0])
            uy = float(a.pos[uid, 1])
            level = 0
            for e in self.emitters.active_indices():
                e = int(e)
                kind = int(self.emitters.kind[e])
                if kind not in (EMITTER_KIND_PLATFORM_RADAR, EMITTER_KIND_SEEKER):
                    continue
                eside = self._emitter_side(e)
                if eside < 0 or not hostile(uside, eside):
                    continue
                ex = float(self.emitters.pos[e, 0])
                ey = float(self.emitters.pos[e, 1])
                d = math.hypot(ex - ux, ey - uy)
                if d <= esm_detection_range(float(self.emitters.power_db[e])):
                    # 主动导引头（来袭 AAM/SAM）→ 锁定级；平台雷达 → 搜索/跟踪级
                    level = 3 if kind == EMITTER_KIND_SEEKER else max(level, 1)
            a.rwr_state[uid] = level


# ---------------------------------------------------------------------------
# FormationAI：编队决策 + 单位执行
# ---------------------------------------------------------------------------


class FormationAI:
    """编队 AI 决策 + 单位 AI 执行（Phase 6c 步骤 ②）。

    用法::

        ai = FormationAI(armor, formations, fc=fc, pathfinder=gpf)
        loop.attach("ai", ai.tick)   # hook(loop, dt)，1s 地面 Tick 分频
    """

    # ---- 默认阈值（可构造覆盖）----
    COMM_THRESHOLD = 0.5          # 成员 comm 低于此值 → 收不到编队指令，自主自保
    LOSS_RETREAT = 0.5            # 编队血量损失超此比例 → 撤退
    MAX_CONCURRENT_FIRE = 3       # 每目标并发开火上限（防过度杀伤）
    SELF_PRESERVE_HEALTH = 0.25   # 自保：血量低于 25% → 撤退
    HIT_DEFENSE_HEALTH = 0.5      # 受击面防御：进攻编队中血量低于 50% 的成员转 DEFEND
    # Phase 6c 步骤⑨：RWR 规避 + 诱饵（docs §八.2/八.3）
    ECM_RAISE = 0.5               # 收到 RWR 告警时 ECM 提升量（clamp 到 1.0）
    EVADE_RADIUS_M = 100.0        # 规避机动偏移距离
    MISSILE_WARNING_RADIUS_M = 1500.0  # 来袭制导弹距此距离内 → 抛诱饵 + 规避
    DECOY_POWER = 500.0           # 诱饵额定强度（高功率 → 吸引导引头）
    DECOY_DECAY_PER_S = 2.0       # 诱饵衰减速率（短寿命）
    # Phase 6c 步骤⑦：队形保持 + 射界规避（docs §二.4）
    SLOT_SPACING_M = 30.0         # 楔形队形槽位间距
    FIELD_OF_FIRE_HALF_ANGLE_DEG = 30.0  # 敌方直射火力射界半角
    # ---- Phase 8 支援请求链（docs §5.2）----
    SUPPORT_RANGE_MULT = 1.5      # 接触距离超成员最大射程 × 此倍数 → 请求炮兵支援
    SUPPORT_MIN_CONFIDENCE = 0.3  # 低于此置信度的接触不请求（看不清不打）
    SUPPORT_MAX_REQUESTS = 2      # 每 tick 每编队最多请求数
    # ---- 部位毁伤行为（Phase 演示：引擎毁失机动 / 履带毁减速）----
    TRACK_SPEED_CAP = 2.0         # 履带毁 → 减速上限 (m/s，演示正常速度 5)

    def __init__(
        self,
        armor: ArmorPool,
        formations: FormationAIPool,
        *,
        fc: Any | None = None,    # FiringControl（开火接口，None=不可开火）
        boxes: Any | None = None,  # BoxPool（读 HULL 部位血；None 时回退 fc.boxes）
        pathfinder: GraphPathfinder | None = None,
        contact_provider: Callable[[], list[Contact]] | None = None,
        events: Any | None = None,
        artillery: "ArtilleryAI | None" = None,  # Phase 6c 步骤⑤：炮兵编队 AI（tick 内驱动）
        # ---- Phase 6c 步骤⑨：RWR / 通信簇 ----
        rwr: RWRDetector | None = None,
        comm_network: CommunicationNetwork | None = None,
        speed: Any | None = None,   # 可选：per-armor 速度数组（履带毁减速，演示装配传入）
    ) -> None:
        self.armor = armor
        self.formations = formations
        self.fc = fc
        self.boxes = boxes
        self.pathfinder = pathfinder
        self._contact_provider = contact_provider
        self.events = events
        self.artillery = artillery
        self.rwr = rwr
        self.comm_network = comm_network
        self.speed = None  # 可选：per-armor 移动速度数组（履带毁减速用，演示装配传入）
        self._members_cache: dict[int, list[int]] | None = None  # 每 tick 惰性缓存（性能）
        self._direct_fire_threats: list[tuple] | None = None    # 每 tick 预计算直射威胁

    def _hull_hp(self, i: int) -> float:
        """某装甲整车血量（HULL 部位血；未绑定 HULL 或无可读池 → 0）。

        docs/part-health-design.md：血量权威来源 = BoxPool.hp，armor.hull_idx 是指针。
        """
        hj = int(self.armor.hull_idx[i])
        if hj < 0:
            return 0.0
        boxes = self.fc.boxes if self.fc is not None else self.boxes
        if boxes is None:
            return 0.0
        return float(boxes.hp[hj])

    # ------------------------------------------------------------------
    # Hook 入口（每 1s 地面 Tick）
    # ------------------------------------------------------------------

    def tick(self, loop: SimulationLoop, dt: float) -> None:
        """AI hook：逐编队（汇总接触 → FSM → 目标分配）+ 单位执行 + 受击面防御
        + RWR 规避/诱饵（步骤⑨）。

        决策频率由 loop 分频（`set_rate("ai", TICK_GROUND)`），dt 为累计时长。
        """
        # 预计算各阵营敌接触 + 编队成员缓存 + 直射威胁缓存
        # （O(units) 一次，避免逐编队/逐成员重复扫描）
        contacts_by_side = self._precompute_contacts()
        self._members_cache = self._build_members_cache()
        self._direct_fire_threats = self._build_direct_fire_threats()
        try:
            for fidx in self.formations.active_indices():
                fidx = int(fidx)
                self._gather_contacts(fidx, contacts_by_side)
                situation = self._situation_of(fidx)
                self._update_intent(fidx, situation)
                self._maybe_request_support(fidx)   # Phase 8：支援请求链（docs §5.2）
                self._assign_targets(fidx)
                self._apply_hit_defense(fidx)
            self._execute_unit_orders()
            # Phase 6c 步骤⑦：射界规避（覆盖机动目标，docs §二.4）
            self._apply_field_of_fire_avoidance_all()
            # Phase 6c 步骤⑨：RWR 告警 → 规避 + 导弹来袭 → 诱饵（docs §八）
            self._tick_rwr_evasion(dt)
            # Phase 6c 步骤⑤：炮兵编队 AI（火力任务分派，docs §六）
            if self.artillery is not None:
                self.artillery.tick(loop, dt)
        finally:
            self._members_cache = None
            self._direct_fire_threats = None

    # ------------------------------------------------------------------
    # 编队 AI：汇总接触
    # ------------------------------------------------------------------

    def _precompute_contacts(self) -> dict[int, list[Contact]] | None:
        """预计算各阵营敌接触表（默认全知模型）。

        注入 contact_provider 时返回 None（仍逐编队调用 provider）；否则每阵营
        只扫一次 armor（O(units)），同阵营编队复用同一接触表——避免 256 编队 ×
        2048 单位的 O(units×formations) 重复扫描。
        """
        if self._contact_provider is not None:
            return None
        sides = {int(self.formations.side[f]) for f in self.formations.active_indices()}
        return {s: self._default_contacts(s) for s in sides}

    def _gather_contacts(
        self, fidx: int, contacts_by_side: dict[int, list[Contact]] | None = None
    ) -> None:
        """汇总敌接触表：默认全知（读 armor 敌方），可注入 contact_provider。

        Phase 8：`TwinContactProvider` 注入（带 side 参数）；6c 旧 provider
        （无参）兼容（TypeError 回退）。
        """
        if self._contact_provider is not None:
            provider = self._contact_provider
            try:
                contacts = provider(int(self.formations.side[fidx]))
            except TypeError:
                contacts = provider()
        elif contacts_by_side is not None:
            contacts = contacts_by_side.get(int(self.formations.side[fidx]), [])
        else:
            contacts = self._default_contacts(int(self.formations.side[fidx]))
        self.formations.contact_set[fidx] = contacts

    def _default_contacts(self, side: int) -> list[Contact]:
        """全知接触：读 armor 池中敌对（hostile(side, ·)）活跃单位。"""
        a = self.armor
        contacts: list[Contact] = []
        for i in a.active_indices():
            if not hostile(side, int(a.side[i])):
                continue
            contacts.append(
                Contact(
                    idx=int(i),
                    side=int(a.side[i]),
                    x=float(a.pos[i, 0]),
                    y=float(a.pos[i, 1]),
                    health=self._hull_hp(int(i)),
                )
            )
        return contacts

    # ------------------------------------------------------------------
    # 编队 AI：态势 + 意图 FSM
    # ------------------------------------------------------------------

    def _situation_of(self, fidx: int) -> FormationSituation:
        members = self._members_of(fidx)
        n = len(members)
        health_sum = sum(self._hull_hp(i) for i in members)
        return FormationSituation(
            friendly_active=n,
            enemy_contacts=len(self.formations.contact_set[fidx] or []),
            health_ratio=health_sum / (_DEFAULT_MAX_HEALTH * n) if n > 0 else 1.0,
            ammo_ratio=self._ammo_ratio(members),
        )

    def _update_intent(self, fidx: int, situation: FormationSituation) -> None:
        """编队意图 FSM（docs §二.1 转移表）。"""
        f = self.formations
        intent = int(f.intent[fidx])

        # 编队级命令优先（玩家/上级下达 HOLD → 待命，直接覆盖）
        if int(f.order[fidx]) == int(UnitOrder.HOLD):
            f.intent[fidx] = int(FormationIntent.HOLD)
            return

        if intent == int(FormationIntent.MANEUVER):
            if situation.enemy_contacts > 0:
                # 兵力优势判定：存活成员 ≥ 敌接触数 → 进攻，否则防御
                if situation.friendly_active >= situation.enemy_contacts:
                    f.intent[fidx] = int(FormationIntent.ADVANCE)
                else:
                    f.intent[fidx] = int(FormationIntent.DEFEND)
        elif intent in (int(FormationIntent.ADVANCE), int(FormationIntent.DEFEND)):
            # 损失超阈值 → 撤退
            if situation.health_ratio < 1.0 - self.LOSS_RETREAT:
                f.intent[fidx] = int(FormationIntent.RETREAT)
        elif intent == int(FormationIntent.RETREAT):
            # 到达安全区（无接触且血量恢复）→ 转防御
            if situation.enemy_contacts == 0 and situation.health_ratio > 0.8:
                f.intent[fidx] = int(FormationIntent.DEFEND)

    # ------------------------------------------------------------------
    # 编队 AI：支援请求链（Phase 8，docs §5.2）
    # ------------------------------------------------------------------

    def _max_weapon_range(self, members: list[int]) -> float:
        """成员最大武器射程（fc 配置；无 fc/配置 → 默认 4000）。"""
        if self.fc is None:
            return 4000.0
        best = 4000.0
        for m in members:
            cfg = self.fc.get_unit_config(m)
            if cfg is not None:
                best = max(best, float(cfg.max_range_m))
        return best

    def _maybe_request_support(self, fidx: int) -> None:
        """支援请求链（docs §5.2）：目标超射程 / 高价值且火力不足 →
        发 support_request 事件（炮兵编队订阅，docs §六.1）。

        条件：进攻/防御意图 + 接触距离 > 成员最大射程 × SUPPORT_RANGE_MULT
        + 接触置信度 ≥ SUPPORT_MIN_CONFIDENCE（迷雾决策：看不清不打）。
        请求方须在通信簇内（comm ≥ COMM_THRESHOLD，玩家指令/数据链可达）。
        """
        if self.events is None:
            return
        intent = int(self.formations.intent[fidx])
        if intent not in (int(FormationIntent.ADVANCE), int(FormationIntent.DEFEND)):
            return
        contacts = self.formations.contact_set[fidx] or []
        if not contacts:
            return
        members = self._members_of(fidx)
        if not members:
            return
        if float(self.formations.comm[fidx]) < self.COMM_THRESHOLD:
            return   # 失联编队：数据链不可达 → 不请求（自保）
        max_range = self._max_weapon_range(members)
        sent = 0
        for c in contacts:
            if sent >= self.SUPPORT_MAX_REQUESTS:
                break
            d = math.hypot(c.x - float(self.armor.pos[members[0], 0]),
                           c.y - float(self.armor.pos[members[0], 1]))
            if d > max_range * self.SUPPORT_RANGE_MULT \
                    and c.confidence >= self.SUPPORT_MIN_CONFIDENCE:
                self.events.emit("support_request", {
                    "kind": "artillery",
                    "requester": fidx,
                    "target_twin": c.twin_slot,
                    "x": c.x,
                    "y": c.y,
                    "priority": 5,
                    "confidence": c.confidence,
                })
                sent += 1

    # ------------------------------------------------------------------
    # 编队 AI：贪心火力分配（指挥链）
    # ------------------------------------------------------------------

    def _assign_targets(self, fidx: int) -> None:
        """贪心火力分配（docs §二.2）：对每个敌接触选最合适成员，写 order/target_idx。

        约束：每目标并发开火上限（防过度杀伤）+ 每个成员最多分配一个目标
        （避免后写覆盖前写导致部分目标落空）。
        """
        contacts = self.formations.contact_set[fidx] or []
        members = self._members_of(fidx)
        if not contacts or not members:
            return
        intent = int(self.formations.intent[fidx])
        # Phase 8：低置信接触（模糊）不分配——看不清的目标不该打（docs §5.1）
        contacts = [c for c in contacts if c.confidence >= self.SUPPORT_MIN_CONFIDENCE]
        if not contacts:
            return

        # 重置成员目标（每 tick 重新分配）
        for m in members:
            self.armor.target_idx[m] = -1

        fire_count: dict[int, int] = {}
        available = list(members)  # 尚未分配目标的成员

        def assign_one(contact: Contact) -> bool:
            """把 contact 分给「最近且有弹药」的可用成员，返回是否成功。"""
            best = self._best_assignee(available, contact)
            if best is None:
                return False
            self.armor.target_idx[best] = contact.idx
            self.armor.target_kind[best] = 0  # 装甲目标
            self.armor.order[best] = self._order_for_intent(intent)
            fire_count[contact.idx] = fire_count.get(contact.idx, 0) + 1
            available.remove(best)
            return True

        # 第一轮：每个目标覆盖一个成员（目标数 ≥ 成员数时全覆盖，避免挤单目标）
        for contact in contacts:
            if not available:
                break
            if not assign_one(contact):
                break  # 无成员可分配（无弹药/无可用）→ 后续接触同样无法分配
        # 第二轮：补充并发（成员多目标少时，每目标最多 MAX_CONCURRENT_FIRE）
        for contact in contacts:
            while (
                fire_count.get(contact.idx, 0) < self.MAX_CONCURRENT_FIRE
                and available
            ):
                if not assign_one(contact):
                    break

    def _order_for_intent(self, intent: int) -> int:
        """编队意图 → 成员执行指令（docs §三.1 表）。"""
        if intent == int(FormationIntent.ADVANCE):
            return int(UnitOrder.ENGAGE)
        if intent == int(FormationIntent.DEFEND):
            return int(UnitOrder.DEFEND)
        if intent == int(FormationIntent.RETREAT):
            return int(UnitOrder.RETREAT)
        return int(UnitOrder.MANEUVER)  # MANEUVER / HOLD

    def _best_assignee(self, members: list[int], contact: Contact) -> int | None:
        """选「距目标最近 + 有弹药」的成员（距离评分简化版）。"""
        a = self.armor
        best: int | None = None
        best_d = 1e18
        for m in members:
            if self.fc is not None and not self._has_ammo(m):
                continue
            dx = a.pos[m, 0] - contact.x
            dy = a.pos[m, 1] - contact.y
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best = int(m)
        return best

    def _has_ammo(self, m: int) -> bool:
        if self.fc is None or self.fc.ammo is None:
            return True
        return float(self.fc.ammo.state.reserve[m]) > 0.0

    # ------------------------------------------------------------------
    # 编队 AI：受击面防御（进攻中的局部防御）
    # ------------------------------------------------------------------

    def _apply_hit_defense(self, fidx: int) -> None:
        """进攻编队命令受击严重的成员转 DEFEND（docs §二.2 第 4 点）。"""
        if int(self.formations.intent[fidx]) != int(FormationIntent.ADVANCE):
            return
        for m in self._members_of(fidx):
            if self._hull_hp(m) < self.HIT_DEFENSE_HEALTH * _DEFAULT_MAX_HEALTH:
                self.armor.order[m] = int(UnitOrder.DEFEND)

    # ------------------------------------------------------------------
    # 单位 AI：执行
    # ------------------------------------------------------------------

    def _part_active(self, i: int, part: int) -> bool:
        """部位箱是否存活（无该部位箱 → True 不限制；毁 → False）。

        部位毁伤行为基础：TURRET 毁 → can_fire（FiringControl 侧）；
        ENGINE 毁 → 失机动；TRACK 毁 → 减速。
        """
        boxes = self.fc.boxes if self.fc is not None else self.boxes
        if boxes is None:
            return True
        m = (boxes.parent == i) & (boxes.part == int(part))
        if not np.any(m):
            return True   # 无该部位箱（简化配置）→ 不限制
        return bool(np.any(boxes.active[m]))

    def _mobility_ok(self, i: int) -> bool:
        """引擎毁 → 失机动（不能发起机动，原地作战）。"""
        return self._part_active(i, int(PartType.ENGINE))

    def _apply_part_mobility(self, i: int) -> None:
        """履带毁 → 减速（speed 上限钳制，每 tick 应用幂等）。"""
        if self.speed is None or i >= len(self.speed):
            return
        if not self._part_active(i, int(PartType.TRACK)):
            self.speed[i] = min(float(self.speed[i]), self.TRACK_SPEED_CAP)

    def _execute_unit_orders(self) -> None:
        """单位 AI 执行：全部活跃装甲按 order 行动（无编队/通信断裂 → 自保）。

        炮兵成员（role=ARTILLERY）跳过——曲射由 ArtilleryAI 驱动（步骤⑤），
        不走直射/机动执行（docs §五.1「不直接参与直射」）。
        部位毁伤行为：引擎毁 → 失机动（跳过机动）；履带毁 → 减速（每 tick 钳制）。
        """
        a = self.armor
        for i in a.active_indices():
            i = int(i)
            if int(a.role[i]) == int(UnitRole.ARTILLERY):
                continue  # 炮兵编队 AI 负责（fire_indirect）
            self._apply_part_mobility(i)   # 履带毁 → 减速
            fid = int(a.formation_id[i])
            # 无编队 或 编队已销毁 → 自保
            if fid < 0 or fid >= self.formations.capacity or not self.formations.active[fid]:
                self._self_preserve(i)
                continue
            # 通信断裂（收不到编队指令）→ 自主自保
            if float(a.comm[i]) < self.COMM_THRESHOLD:
                self._self_preserve(i)
                continue

            order = int(a.order[i])
            if order == int(UnitOrder.ENGAGE):
                self._execute_engage(i)
            elif order == int(UnitOrder.DEFEND):
                self._execute_defend(i)
            elif order == int(UnitOrder.MANEUVER):
                if not self._mobility_ok(i):
                    continue   # 引擎毁 → 失机动
                fid = int(a.formation_id[i])
                # 队形保持（步骤⑦，docs §二.4）：成员按槽位偏移楔形展开，
                # 锚点 = target_zone；单成员槽位 0 → 偏移 (0,0)，直达目标区
                members = self._members_of(fid)
                slot = members.index(i) if i in members else 0
                ox, oy = self._formation_offset(slot, len(members))
                self._move_to(
                    i,
                    float(self.formations.target_zone[fid, 0]) + ox,
                    float(self.formations.target_zone[fid, 1]) + oy,
                )
            elif order == int(UnitOrder.RETREAT):
                if self._mobility_ok(i):
                    self._move_retreat(i)
            # HOLD：静默（不移动不射击）

    def _execute_engage(self, i: int) -> None:
        """进攻：目标可打则开火，否则机动靠近目标（引擎毁 → 原地射击）。"""
        tid = int(self.armor.target_idx[i])
        if tid >= 0 and self.armor.active[tid]:
            if self._try_fire(i, tid):
                return
            if self._mobility_ok(i):
                self._move_toward_target(i, tid)

    def _execute_defend(self, i: int) -> None:
        """防御：只对进入射程的分配目标射击，不机动。"""
        tid = int(self.armor.target_idx[i])
        if tid >= 0 and self.armor.active[tid]:
            self._try_fire(i, tid)

    def _try_fire(self, i: int, tid: int) -> bool:
        """尝试开火：can_fire + 射程内 → fc.fire。返回是否成功。"""
        if self.fc is None or not self.fc.can_fire(i):
            return False
        if not self._in_range(i, tid):
            return False
        rec = self.fc.fire(i, tid)
        if rec is not None and self.events is not None:
            self.events.emit("ai_fire", {"shooter": i, "target": tid})
        return rec is not None

    def _in_range(self, i: int, tid: int) -> bool:
        a = self.armor
        dx = a.pos[tid, 0] - a.pos[i, 0]
        dy = a.pos[tid, 1] - a.pos[i, 1]
        return float(np.hypot(dx, dy)) <= self._max_range_of(i)

    def _max_range_of(self, i: int) -> float:
        """单位最大射程（读 FiringControl 配置；缺省 500m）。"""
        if self.fc is None:
            return 500.0
        cfg = getattr(self.fc, "_configs", {}).get(i)
        if cfg is None:
            return 500.0
        return float(getattr(cfg, "max_range_m", 500.0))

    # ------------------------------------------------------------------
    # 机动
    # ------------------------------------------------------------------

    def _move_to(self, i: int, tx: float, ty: float) -> None:
        """寻路移动到 (tx, ty)，写 armor.wp（无 pathfinder 时直线）。"""
        a = self.armor
        sx = float(a.pos[i, 0])
        sy = float(a.pos[i, 1])
        if self.pathfinder is not None:
            wps = self.pathfinder.find_path((sx, sy), (tx, ty))
            if wps:
                a.set_waypoints(i, wps + [(tx, ty)])
                return
        a.set_waypoints(i, [(tx, ty)])  # 直线回退

    def _move_toward_target(self, i: int, tid: int) -> None:
        tx = float(self.armor.pos[tid, 0])
        ty = float(self.armor.pos[tid, 1])
        self._move_to(i, tx, ty)

    def _move_retreat(self, i: int) -> None:
        """撤退：远离最近威胁方向机动 100m。"""
        nearest = self._nearest_enemy(i)
        if nearest is None:
            return
        a = self.armor
        ex = float(a.pos[nearest, 0])
        ey = float(a.pos[nearest, 1])
        ix = float(a.pos[i, 0])
        iy = float(a.pos[i, 1])
        dx = ix - ex
        dy = iy - ey
        n = max(float(np.hypot(dx, dy)), 1e-6)
        self._move_to(i, ix + dx / n * 100.0, iy + dy / n * 100.0)

    # ------------------------------------------------------------------
    # 队形保持 + 射界规避（Phase 6c 步骤⑦，docs §二.4）
    # ------------------------------------------------------------------

    def _formation_offset(self, slot: int, count: int) -> tuple[float, float]:
        """楔形（wedge）队形槽位偏移：slot 0 = 锚点，其余左右交替向后展开。

        docs §二.4「队形偏移由编队槽位号决定」——Phase 6c 用固定楔形
        （菱形/横队留后续参数化）；返回 (横向 dx, 纵向 dy) 世界偏移。
        """
        if slot <= 0 or count <= 1:
            return (0.0, 0.0)
        row = (slot + 1) // 2                      # 1,1,2,2,3,3...
        side = 1.0 if slot % 2 == 1 else -1.0       # 左/右交替
        dx = side * row * self.SLOT_SPACING_M
        dy = -row * self.SLOT_SPACING_M * 0.6       # 向后（y 负方向）
        return (dx, dy)

    def _apply_field_of_fire_avoidance_all(self) -> None:
        """逐编队射界规避（覆盖机动目标，docs §二.4「退出有效射程/绕出射界」）。"""
        for fidx in self.formations.active_indices():
            self._apply_field_of_fire_avoidance(int(fidx))

    def _apply_field_of_fire_avoidance(self, fidx: int) -> None:
        """成员落入敌方直射火力射界 → 命令绕出（机动规避）。"""
        for m in self._members_of(fidx):
            threat = self._in_enemy_field_of_fire(m)
            if threat is not None:
                self._avoid_field_of_fire(m, threat)

    def _build_direct_fire_threats(self) -> dict[int, list[tuple]]:
        """预计算敌方直射威胁（每 tick 一次）：{成员阵营 → [(idx, x, y, cos_yaw, sin_yaw, max_range)]}。

        把每成员都会重复的 `role` 过滤 / `hostile` / `_max_range_of` / `cos(yaw)`/
        `sin(yaw)` 提前算好，射界判定只做廉价算术——避免 O(成员×单位) 的重复重算。
        """
        a = self.armor
        direct: list[tuple] = []
        for j in a.active_indices():
            j = int(j)
            if int(a.role[j]) not in (int(UnitRole.TANK), int(UnitRole.TOP_ATTACK_ATGM)):
                continue
            yaw = float(a.yaw[j])
            direct.append(
                (
                    j,
                    float(a.pos[j, 0]),
                    float(a.pos[j, 1]),
                    math.cos(yaw),
                    math.sin(yaw),
                    self._max_range_of(j),
                    int(a.side[j]),
                )
            )
        # 以活跃装甲阵营为键（无编队单位也受射界规避保护；tick 外直接调用同样可用）
        member_sides = {int(a.side[i]) for i in a.active_indices()}
        return {
            s: [t for t in direct if hostile(s, t[6])]
            for s in member_sides
        }

    def _in_enemy_field_of_fire(self, m: int) -> int | None:
        """返回威胁 m 的敌方直射单位（坦克/攻顶 ATGM 发射器），无则 None。

        威胁判定：敌 role ∈ {TANK, TOP_ATTACK_ATGM} + 敌对 + 距离 ≤ 敌射程 +
        m 位于敌朝向 ±FIELD_OF_FIRE_HALF_ANGLE_DEG 的射界锥内。
        """
        threats = self._direct_fire_threats
        if threats is None:
            threats = self._build_direct_fire_threats()  # tick 外直接调用的兜底
        a = self.armor
        mx = float(a.pos[m, 0])
        my = float(a.pos[m, 1])
        mside = int(a.side[m])
        cos_half = math.cos(math.radians(self.FIELD_OF_FIRE_HALF_ANGLE_DEG))
        for j, ex, ey, fx, fy, max_range, _eside in threats.get(mside, []):
            dx = mx - ex
            dy = my - ey
            d2 = dx * dx + dy * dy
            if d2 > max_range * max_range:
                continue
            dn = math.sqrt(d2)
            if dn < 1e-6:
                return j  # 重叠：必被威胁
            if (fx * dx + fy * dy) / dn >= cos_half:
                return j
        return None

    def _avoid_field_of_fire(self, m: int, threat: int) -> None:
        """绕出射界：向敌朝向垂直方向机动（选择远离敌人的一侧）。"""
        a = self.armor
        ex = float(a.pos[threat, 0])
        ey = float(a.pos[threat, 1])
        yaw = float(a.yaw[threat])
        px = -math.sin(yaw)
        py = math.cos(yaw)
        mx = float(a.pos[m, 0])
        my = float(a.pos[m, 1])
        dx = mx - ex
        dy = my - ey
        if dx * px + dy * py < 0.0:  # 选远离敌人的垂直侧
            px, py = -px, -py
        self._move_to(m, mx + px * self.EVADE_RADIUS_M, my + py * self.EVADE_RADIUS_M)
        if self.events is not None:
            self.events.emit("field_of_fire_evade", {"unit": m, "threat": threat})

    # ------------------------------------------------------------------
    # 自主自保（通信断裂 / 无编队）
    # ------------------------------------------------------------------

    def _self_preserve(self, i: int) -> None:
        """简单自保：血量过低 → 撤退；否则对最近威胁还击（不主动进攻）。"""
        a = self.armor
        if self._hull_hp(i) < self.SELF_PRESERVE_HEALTH * _DEFAULT_MAX_HEALTH:
            self._move_retreat(i)
            return
        threat = self._nearest_enemy(i)
        if threat is not None:
            self._try_fire(i, threat)

    def _nearest_enemy(self, i: int) -> int | None:
        """返回距 i 最近的敌方装甲索引（全知）。"""
        a = self.armor
        best: int | None = None
        best_d = 1e18
        for j in a.active_indices():
            if not hostile(int(a.side[i]), int(a.side[j])):
                continue
            dx = a.pos[j, 0] - a.pos[i, 0]
            dy = a.pos[j, 1] - a.pos[i, 1]
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best = int(j)
        return best

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _build_members_cache(self) -> dict[int, list[int]]:
        """一次遍历把活跃装甲按 formation_id 分组（每 tick 调用一次）。"""
        cache: dict[int, list[int]] = {}
        for i in self.armor.active_indices():
            fid = int(self.armor.formation_id[i])
            if fid >= 0:
                cache.setdefault(fid, []).append(int(i))
        return cache

    def _members_of(self, fidx: int) -> list[int]:
        """编队成员（formation_id == fidx 的活跃装甲）。

        有 tick 级缓存时 O(1)（`_members_cache`）；否则全扫（供 tick 外直接调用）。
        """
        if self._members_cache is not None:
            return self._members_cache.get(fidx, [])
        a = self.armor
        return [
            int(i)
            for i in a.active_indices()
            if int(a.formation_id[i]) == fidx
        ]

    def _ammo_ratio(self, members: list[int]) -> float:
        if self.fc is None or self.fc.ammo is None or not members:
            return 1.0
        st = self.fc.ammo.state
        ratios = []
        for m in members:
            cap = float(st.capacity[m])
            if cap > 0:
                ratios.append(float(st.reserve[m]) / cap)
        return float(np.mean(ratios)) if ratios else 1.0

    # ------------------------------------------------------------------
    # RWR 规避 + 诱饵（Phase 6c 步骤⑨，docs §八）
    # ------------------------------------------------------------------

    def _ensure_rwr(self) -> RWRDetector | None:
        """惰性装配 RWR 探测器（有 fc.emitters 才可用）。"""
        if self.rwr is None:
            if self.fc is None or self.fc.emitters is None:
                return None
            self.rwr = RWRDetector(
                self.armor,
                self.fc.emitters,
                self.fc.projectiles,
                events=self.events,
            )
        return self.rwr

    def _tick_rwr_evasion(self, dt: float) -> None:
        """RWR 告警 → 规避（开 ECM + 机动 + 烟雾）；导弹来袭 → 抛诱饵。"""
        rwr = self._ensure_rwr()
        if rwr is None:
            return
        for w in rwr.tick(dt):
            self._apply_rwr_evasion(w)
        self._apply_missile_defense(dt)

    def _apply_rwr_evasion(self, w: dict[str, Any]) -> None:
        """收到照射告警 → 开 ECM + 机动规避 + 抛烟雾（docs §八.2）。"""
        uid = int(w["unit"])
        # 1. ECM 提升（压制敌方射频导引头检测；clamp 到 1.0）
        self.armor.ecm[uid] = min(1.0, float(self.armor.ecm[uid]) + self.ECM_RAISE)
        # 2. 机动规避：向告警方向反方向偏移（走既有 wp 接口）
        dx, dy = w["direction"]
        n = math.hypot(dx, dy)
        if n < 1e-6:
            dx, dy, n = 1.0, 0.0, 1.0
        px = float(self.armor.pos[uid, 0])
        py = float(self.armor.pos[uid, 1])
        self._move_to(uid, px - dx / n * self.EVADE_RADIUS_M, py - dy / n * self.EVADE_RADIUS_M)
        # 3. 烟雾（事件标记；实体烟雾遮蔽 Phase 7/8 落地）
        if self.events is not None:
            self.events.emit("smoke", {"unit": uid, "reason": "rwr"})

    def _apply_missile_defense(self, dt: float) -> None:
        """来袭制导弹（敌 owner + guid_law≥0）距我单位 ≤ 阈值 → 抛诱饵 + 规避。"""
        if self.fc is None or self.fc.projectiles is None:
            return
        p = self.fc.projectiles
        guided = [int(i) for i in p.active_indices() if p.guid_law[int(i)] >= 0]
        if not guided:
            return
        for uid in self.armor.active_indices():
            uid = int(uid)
            for proj in guided:
                owner = int(p.owner[proj])
                if owner < 0 or not hostile(int(self.armor.side[uid]), int(self.armor.side[owner])):
                    continue
                dx = float(p.pos[proj, 0]) - float(self.armor.pos[uid, 0])
                dy = float(p.pos[proj, 1]) - float(self.armor.pos[uid, 1])
                if math.hypot(dx, dy) <= self.MISSILE_WARNING_RADIUS_M:
                    self._deploy_decoy(uid)
                    break

    def _deploy_decoy(self, uid: int) -> None:
        """抛诱饵（flare/chaff）：在单位位置部署瞬时高功率 emitter（docs §八.3）。"""
        if self.fc is None or self.fc.emitters is None:
            return
        pos = self.armor.pos[uid]
        self.fc.emitters.deploy_decoy(
            float(pos[0]), float(pos[1]), float(pos[2]),
            power=self.DECOY_POWER,
            decay_per_s=self.DECOY_DECAY_PER_S,
        )
        if self.events is not None:
            self.events.emit("decoy_deployed", {"unit": uid})


# ---------------------------------------------------------------------------
# ArtilleryAI：炮兵编队 AI（Phase 6c 步骤⑤，docs/phase6-ai-pipeline.md §六）
# ---------------------------------------------------------------------------


class ArtilleryAI:
    """炮兵编队 AI（曲射火控，Phase 6c 步骤⑤）。

    决策主体：接收友军/玩家 STRIKE 指令 → 按优先级排队 → 分派给炮位
    （role=ARTILLERY 成员）→ `fc.fire_indirect`（弹药消耗/冷却由开火链管理）。

    红线同 FormationAI（docs §〇）：
        1. 只决策、只走既有接口（`FiringControl.fire_indirect` / CommandQueue），
           不直接改 SoA 数组。
        2. 指挥链门控：编队 comm < 阈值 → 不收/不执行指令（§九 完整分簇 Phase 8）。
        3. 玩家指令优先级最高（`handle_command` 入队 priority=1，先于友军请求执行）。

    装配（scenario ⑩ 落地）::

        artillery = ArtilleryAI(armor, formations, fc=fc, events=events)
        loop.command_handler = artillery.handle_command   # 玩家 STRIKE 直达
        ai = FormationAI(armor, formations, fc=fc, artillery=artillery)  # tick 内驱动
    """

    COMM_THRESHOLD = 0.5     # 编队 comm 低于此值 → 不接收/不执行任务

    # 反炮闭环默认参数（Phase 6c 步骤⑥，docs §六.3）
    COUNTER_BATTERY_RANGE_M = 20000.0   # 反炮探测距离（敌方闪光在此距离内才反）
    WARNING_RADIUS_M = 150.0            # 来袭弹落点距炮位此距离内 → 告警转移
    DISPLACE_RADIUS_M = 300.0           # 疏散转移距离（远离来袭弹落点方向）

    def __init__(
        self,
        armor: ArmorPool,
        formations: FormationAIPool,
        *,
        fc: Any | None = None,      # FiringControl（fire_indirect 接口）
        events: Any | None = None,  # EventBus（artillery_fire 事件）
        comm_threshold: float = COMM_THRESHOLD,
        # ---- Phase 6c 步骤⑥：反炮闭环 ----
        side: int | None = None,       # 本 AI 代表阵营；None=驱动所有炮兵编队（旧行为）
        muzzle_flashes: MuzzleFlashLog | None = None,  # 炮口闪光共享日志（暴露源）
        counter_battery_range_m: float = COUNTER_BATTERY_RANGE_M,
        warning_radius_m: float = WARNING_RADIUS_M,
        displace_radius_m: float = DISPLACE_RADIUS_M,
    ) -> None:
        self.armor = armor
        self.formations = formations
        self.fc = fc
        self.events = events
        self.comm_threshold = float(comm_threshold)
        self._missions: list[ArtilleryMission] = []
        # 反炮闭环
        self.side = int(side) if side is not None else None
        self.muzzle_flashes = muzzle_flashes
        self.counter_battery_range_m = float(counter_battery_range_m)
        self.warning_radius_m = float(warning_radius_m)
        self.displace_radius_m = float(displace_radius_m)
        self._countered: set[int] = set()      # 已反炮的闪光 id（每闪光只反一次）
        self._displacing: dict[int, float] = {}  # 编队 idx → 疏散冷却剩余秒（防每 tick 重复转移）

    # ------------------------------------------------------------------
    # 指令入口（docs §六.2：统一走既有 CommandQueue / 装配层直连）
    # ------------------------------------------------------------------

    def handle_command(self, cmd: Command) -> bool:
        """玩家/叙事层 STRIKE 指令（优先级最高）。返回是否被本模块消费。

        params 支持：coords=(x, y[, z]) 或 target_idx（活目标自动取坐标）+
        ammo / rounds。
        """
        if cmd.type != CommandType.STRIKE:
            return False
        params = cmd.params
        coords = params.get("coords")
        target_idx = int(params.get("target_idx", -1))
        if coords is None:
            a = self.armor
            if 0 <= target_idx < a.capacity and a.active[target_idx]:
                p = a.pos[target_idx]
                coords = (float(p[0]), float(p[1]), float(p[2]))
            else:
                return False
        self._push(
            coords,
            ammo=params.get("ammo"),
            rounds=int(params.get("rounds", 1)),
            priority=1,
            target_idx=target_idx,
        )
        return True

    def request_strike(
        self,
        coords: tuple[float, float, float] | tuple[float, float],
        *,
        ammo: str | None = None,
        target_idx: int = -1,
        rounds: int = 1,
    ) -> bool:
        """友军请求（装甲/侦察发现目标下单；优先级低于玩家指令）。"""
        return self._push(
            coords, ammo=ammo, rounds=rounds, priority=0, target_idx=target_idx
        )

    def _push(
        self,
        coords: tuple[float, float, float] | tuple[float, float],
        *,
        ammo: str | None,
        rounds: int,
        priority: int,
        target_idx: int,
    ) -> bool:
        if rounds < 1:
            return False
        self._missions.append(
            ArtilleryMission(
                coords=(float(coords[0]), float(coords[1])),
                ammo=ammo,
                rounds=max(1, int(rounds)),
                priority=int(priority),
                target_idx=int(target_idx),
            )
        )
        return True

    @property
    def pending_count(self) -> int:
        """待执行火力任务数。"""
        return len(self._missions)

    # ------------------------------------------------------------------
    # Hook 入口（每 1s 地面 Tick，由 FormationAI.tick 内驱动）
    # ------------------------------------------------------------------

    def tick(self, loop: SimulationLoop, dt: float) -> None:
        """处理炮兵编队火力任务 + 反炮闭环（步骤⑥）+ 自主接单（Phase 8 §6.1）。

        - 火力任务：每 tick 每炮位最多打一发（射速冷却限速），玩家指令
          （priority=1）先于友军请求（priority=0）分配炮位。
        - 反炮闭环（docs §六.3）：先检测敌方炮口闪光 → 反炮 STRIKE；再检测
          来袭弹 → 疏散转移（即使无火力任务也执行）。
        - 自主接单（Phase 8）：drain support_request 事件（装甲编队呼叫，docs §5.2）
          → request_strike（**目标坐标来自孪生，带误差**——迷雾影响落点，设计意图）。
        """
        self._handle_support_events()
        # 反炮闭环：无任务也运行（开火暴露 / 来袭告警与火力任务队列无关）
        self._tick_counter_battery(dt)
        self._tick_incoming_warning(dt)

        if self.fc is None or not self._missions:
            return
        for fidx in self._artillery_formations():
            if float(self.formations.comm[fidx]) < self.comm_threshold:
                continue  # 指挥链断裂：不收指令（§九 完整分簇 Phase 8 接入）
            guns = self._guns_of(fidx)
            if not guns:
                continue
            for gun in guns:
                if not self._missions:
                    break
                if not self.fc.can_fire(gun):
                    continue  # 冷却 / 无弹药 / 无炮塔
                mission = max(self._missions, key=lambda m: m.priority)
                rec = self.fc.fire_indirect(
                    gun,
                    mission.coords[0],
                    mission.coords[1],
                    ammo_id=mission.ammo,
                    target_idx=mission.target_idx,
                )
                if rec is None:
                    continue
                self._register_muzzle_flash(gun)
                mission.rounds -= 1
                if mission.rounds <= 0:
                    self._missions.remove(mission)
                if self.events is not None:
                    self.events.emit(
                        "artillery_fire",
                        {
                            "shooter": gun,
                            "proj": rec.proj_idx,
                            "x": mission.coords[0],
                            "y": mission.coords[1],
                            "ammo": rec.ammo_id,
                            "priority": mission.priority,
                        },
                    )

    # ------------------------------------------------------------------
    # 自主接单（Phase 8，docs §6.1：订阅 support_request 事件）
    # ------------------------------------------------------------------

    def _handle_support_events(self) -> None:
        """订阅 EventBus 的 support_request（装甲编队呼叫支援，docs §5.2）。

        - 只接 kind=artillery 的请求；目标坐标来自孪生（带误差 → 落点散布，
          迷雾直接影响炮兵精度——设计意图 docs §6.1）；
        - 指挥链门控：编队 comm < 阈值 → 不收指令（§3.4）。
        """
        if self.events is None:
            return
        for e in self.events.drain_filter({"support_request"}):
            d = e.data
            if d.get("kind") != "artillery":
                continue
            # 友军请求级（priority=0，低于玩家 STRIKE）；目标坐标来自孪生（带误差）
            self.request_strike(
                (float(d.get("x", 0.0)), float(d.get("y", 0.0))),
                rounds=1,
            )
            if self.events is not None:
                self.events.emit("support_accepted", {
                    "requester": d.get("requester"),
                    "x": d.get("x"),
                    "y": d.get("y"),
                })

    # ------------------------------------------------------------------
    # 反炮闭环（Phase 6c 步骤⑥，docs §六.3「开火→暴露→反炮→转移」）
    # ------------------------------------------------------------------

    def _register_muzzle_flash(self, gun: int) -> None:
        """开火后登记炮口闪光（暴露）。side 未指定时按编队阵营推导。"""
        if self.muzzle_flashes is None:
            return
        side = self.side
        if side is None:
            fid = int(self.armor.formation_id[gun])
            if 0 <= fid < self.formations.capacity:
                side = int(self.formations.side[fid])
            else:
                side = 0
        p = self.armor.pos[gun]
        self.muzzle_flashes.add(side, float(p[0]), float(p[1]), float(p[2]))

    def _tick_counter_battery(self, dt: float) -> None:
        """检测敌方炮口闪光 → 反炮 STRIKE（自动下单，priority=0 友军请求级）。

        距离门控：闪光距我方任一炮位 ≤ counter_battery_range_m 才反；每闪光
        只反一次（_countered 去重）。闪光老化由共享日志 tick() 驱动。
        """
        if self.fc is None or self.muzzle_flashes is None or self.side is None:
            return
        my_guns = self._my_guns()
        if not my_guns:
            return
        for flash in self.muzzle_flashes.enemy_flashes(self.side):
            if flash.id in self._countered:
                continue
            if not self._within_range_of_any(flash.x, flash.y, my_guns, self.counter_battery_range_m):
                continue
            self._countered.add(flash.id)
            self._push((flash.x, flash.y), ammo=None, rounds=1, priority=0, target_idx=-1)
            if self.events is not None:
                self.events.emit(
                    "counter_battery",
                    {"side": flash.side, "x": flash.x, "y": flash.y, "z": flash.z},
                )

    def _tick_incoming_warning(self, dt: float) -> None:
        """检测来袭敌方曲射弹 → 命令我方炮位疏散转移（远离落点）。

        扫描 fc.projectiles 中「owner 敌对 + indirect」的弹丸，其落点参考
        （guid_target_pos）距我方炮位 ≤ warning_radius_m → 疏散该编队。
        """
        if self.fc is None or self.side is None:
            return
        p = self.fc.projectiles
        for fidx in self._artillery_formations():
            if float(self.formations.comm[fidx]) < self.comm_threshold:
                continue
            guns = self._guns_of(fidx)
            if not guns:
                continue
            # 疏散冷却：转移后若干秒内不再重复触发
            cd = self._displacing.get(fidx, 0.0) - dt
            if cd > 0.0:
                self._displacing[fidx] = cd
                continue
            threatened = None
            for proj in p.active_indices():
                if not p.indirect[int(proj)]:
                    continue
                owner = int(p.owner[int(proj)])
                if owner < 0 or not hostile(self.side, int(self.armor.side[owner])):
                    continue
                tpos = p.guid_target_pos[int(proj)]
                if self._within_range_of_any(float(tpos[0]), float(tpos[1]), guns, self.warning_radius_m):
                    threatened = (float(tpos[0]), float(tpos[1]))
                    break
            if threatened is not None:
                self._displace_formation(fidx, guns, threatened)

    def _displace_formation(
        self, fidx: int, guns: list[int], threat: tuple[float, float]
    ) -> None:
        """命令炮兵编队疏散转移：各炮位沿「远离威胁落点」方向机动 + 编队记疏散意图。"""
        for gun in guns:
            gp = self.armor.pos[gun]
            dx = float(gp[0]) - threat[0]
            dy = float(gp[1]) - threat[1]
            n = math.hypot(dx, dy)
            if n < 1e-6:
                dx, dy, n = 1.0, 0.0, 1.0  # 恰在落点：默认向东
            tx = float(gp[0] + dx / n * self.displace_radius_m)
            ty = float(gp[1] + dy / n * self.displace_radius_m)
            self.armor.set_waypoints(gun, [(tx, ty)])
        self.formations.order[fidx] = int(UnitOrder.MANEUVER)
        self._displacing[fidx] = 10.0  # 疏散冷却 10s（防每 tick 重复转移）
        if self.events is not None:
            self.events.emit(
                "artillery_displace", {"formation": fidx, "guns": list(guns)}
            )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _artillery_formations(self) -> list[int]:
        """活跃的炮兵编队（编队 role == ARTILLERY；side 指定时按阵营过滤）。"""
        f = self.formations
        out = []
        for i in f.active_indices():
            i = int(i)
            if int(f.role[i]) != int(UnitRole.ARTILLERY):
                continue
            if self.side is not None and int(f.side[i]) != self.side:
                continue
            out.append(i)
        return out

    def _my_guns(self) -> list[int]:
        """我方全部活跃炮兵成员（反炮距离门控用）。"""
        out: list[int] = []
        for fidx in self._artillery_formations():
            out.extend(self._guns_of(fidx))
        return out

    def _within_range_of_any(self, x: float, y: float, guns: list[int], radius: float) -> bool:
        """坐标 (x, y) 距任一炮位 ≤ radius（读 self.armor 的炮位位置）。"""
        a = self.armor
        r2 = float(radius) * float(radius)
        for gun in guns:
            dx = float(a.pos[gun, 0]) - x
            dy = float(a.pos[gun, 1]) - y
            if dx * dx + dy * dy <= r2:
                return True
        return False

    def _guns_of(self, fidx: int) -> list[int]:
        """编队内活跃炮兵成员（单位 role == ARTILLERY）。"""
        a = self.armor
        return [
            int(i)
            for i in a.active_indices()
            if int(a.formation_id[i]) == fidx
            and int(a.role[i]) == int(UnitRole.ARTILLERY)
        ]
