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

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
from numba import njit

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.entities.pools import ArmorPool, FormationAIPool
from Airis.lulilo.movement.pathfinding import GraphPathfinder
from Airis.lulilo.simulation.loop import SimulationLoop

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


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Contact:
    """一条敌接触（编队 AI 汇总用，Python 层非热路径）。

    speed / weapon_pen 为威胁评分输入（Phase 6c 恒 0，Phase 7/8 细化）。
    """
    idx: int          # 装甲池索引
    team: int         # 阵营
    x: float
    y: float
    health: float
    speed: float = 0.0
    weapon_pen: float = 0.0
    kind: int = 0     # 目标类型：0=装甲 1=辐射源 2=光斑 3=地形点


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


# ---------------------------------------------------------------------------
# 目标评分内核（Numba，向量化批量；供编队 AI 与自保复用，docs §三.4）
# ---------------------------------------------------------------------------


@njit(cache=True)
def score_targets(
    dist: np.ndarray,          # (K,) float32 距离
    speed: np.ndarray,         # (K,) float32 目标速度
    weapon_pen: np.ndarray,    # (K,) float32 敌武器威胁
    health: np.ndarray,        # (K,) float32 目标血量（残血优先补刀）
    w_range: float,
    w_speed: float,
    w_weapon: float,
    w_health: float,
    max_range: float,
    out: np.ndarray,           # (K,) float32 威胁分（原地写）
) -> None:
    """威胁评分：threat = w_range·(1−d/max_range) + w_speed·speed
                            + w_weapon·weapon_pen + w_health·(1−health)。

    距离越近越危险、敌方速度越快越危险、敌武器威胁越大越危险、
    残血目标优先补刀（易击杀）。
    """
    k = dist.shape[0]
    for i in range(k):
        out[i] = (
            w_range * (1.0 - dist[i] / max_range)
            + w_speed * speed[i]
            + w_weapon * weapon_pen[i]
            + w_health * (1.0 - health[i])
        )


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
    ) -> None:
        self.armor = armor
        self.formations = formations
        self.fc = fc
        self.boxes = boxes
        self.pathfinder = pathfinder
        self._contact_provider = contact_provider
        self.events = events

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
        """AI hook：逐编队（汇总接触 → FSM → 目标分配）+ 单位执行 + 受击面防御。

        决策频率由 loop 分频（`set_rate("ai", TICK_GROUND)`），dt 为累计时长。
        """
        for fidx in self.formations.active_indices():
            fidx = int(fidx)
            self._gather_contacts(fidx)
            situation = self._situation_of(fidx)
            self._update_intent(fidx, situation)
            self._assign_targets(fidx)
            self._apply_hit_defense(fidx)
        self._execute_unit_orders()

    # ------------------------------------------------------------------
    # 编队 AI：汇总接触
    # ------------------------------------------------------------------

    def _gather_contacts(self, fidx: int) -> None:
        """汇总敌接触表：默认全知（读 armor 敌方），可注入 contact_provider。"""
        if self._contact_provider is not None:
            contacts = self._contact_provider()
        else:
            contacts = self._default_contacts(int(self.formations.team[fidx]))
        self.formations.contact_set[fidx] = contacts

    def _default_contacts(self, team: int) -> list[Contact]:
        """全知接触：读 armor 池中敌方（team 相反）活跃单位（双阵营简化 0↔1）。"""
        a = self.armor
        enemy_team = 1 - team
        contacts: list[Contact] = []
        for i in a.active_indices():
            if int(a.team[i]) != enemy_team:
                continue
            contacts.append(
                Contact(
                    idx=int(i),
                    team=int(a.team[i]),
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
            assign_one(contact)
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

    def _execute_unit_orders(self) -> None:
        """单位 AI 执行：全部活跃装甲按 order 行动（无编队/通信断裂 → 自保）。"""
        a = self.armor
        for i in a.active_indices():
            i = int(i)
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
                fid = int(a.formation_id[i])
                self._move_to(
                    i,
                    float(self.formations.target_zone[fid, 0]),
                    float(self.formations.target_zone[fid, 1]),
                )
            elif order == int(UnitOrder.RETREAT):
                self._move_retreat(i)
            # HOLD：静默（不移动不射击）

    def _execute_engage(self, i: int) -> None:
        """进攻：目标可打则开火，否则机动靠近目标。"""
        tid = int(self.armor.target_idx[i])
        if tid >= 0 and self.armor.active[tid]:
            if self._try_fire(i, tid):
                return
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
        enemy_team = 1 - int(a.team[i])
        best: int | None = None
        best_d = 1e18
        for j in a.active_indices():
            if int(a.team[j]) != enemy_team:
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

    def _members_of(self, fidx: int) -> list[int]:
        """编队成员（formation_id == fidx 的活跃装甲）。"""
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
