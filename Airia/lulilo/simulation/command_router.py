"""玩家/叙事层地面指令路由（AGENTS.md 3.7 的**消费端**扩展）。

背景
----
`SimulationLoop._apply_command`（`loop.py:156-169`）对未被 `command_handler`
消费的指令**只 emit 一个 `command` 事件**；`ArtilleryAI.handle_command`
（`ai.py:1341-1348`）只认 `STRIKE`。于是叙事层 HOLD/REDEPLOY 选项在生产路径上
「零实际效果」——玩家选择只是日志。

> 依据：`docs/story-architecture-review.md` §三.5；修复后「玩家选择改变战局」成立。

指令语义（与 `FormationIntent` / `UnitOrder` 对齐，`ai.py:59-74`）
----------------------------------------------------------------
| Command   | 语义 | 落点 |
|-----------|------|------|
| HOLD      | 固守：停止机动，仍可对进入射程目标开火 | `order`/`intent` = DEFEND；清路标点 + 速度 0 |
| REDEPLOY  | 转移/推进：设任务目标区 + 逐车路标点 | `order` = MANEUVER，`intent` = ADVANCE；恢复速度 |
| RESUPPLY  | 申请补给：按单位下单并结算 | `AmmoManager`/`FuelManager.request_supply` + `process_requests` |
| DISENGAGE | 脱离接触：背离敌重心机动 | `order`/`intent` = RETREAT |

> 注意：`FormationIntent.HOLD`/`UnitOrder.HOLD` 在引擎中语义是「待命（静默，
> 辐射管制，不移动不射击）」，与叙事层「固守（停止机动但仍开火）」不同——
> 故 HOLD 指令映射到 **DEFEND**，而非同名枚举（docs 语义对齐，见上表）。

`target_id` 解析顺序
--------------------
1. 场景别名（`register_alias`，如 `"7th_armor"`）——叙事层 YAML 用叙事名，
   装配层把叙事名绑到真实编队索引，避免引擎认识剧情专名；
2. 单位名（`set_unit_names`，如 `"M1A2 #03"`）；
3. `"formation:<idx>"`；
4. 未匹配 / 空 → **本方全部地面（非炮兵）编队**。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.simulation.ai import FormationIntent, UnitOrder, UnitRole
from Airia.lulilo.simulation.commands import Command, CommandType

logger = get_logger(__name__)


class GroundCommandRouter:
    """把叙事层地面指令落到编队/单位（HOLD/REDEPLOY/RESUPPLY/DISENGAGE）。

    纯逻辑（无 Qt / 无 UI 依赖），可在无头测试中直接构造。
    """

    HANDLED: tuple[CommandType, ...] = (
        CommandType.HOLD,
        CommandType.REDEPLOY,
        CommandType.RESUPPLY,
        CommandType.DISENGAGE,
    )
    DEFAULT_SPEED: float = 5.0      # m/s（与演示编队一致）
    WITHDRAW_DIST: float = 900.0    # DISENGAGE 脱离距离（m）

    def __init__(
        self,
        armor: Any,
        formations: Any,
        speed: Any | None = None,
        *,
        pathfinder: Any | None = None,
        ammo: Any | None = None,
        fuel: Any | None = None,
        events: Any | None = None,
        player_side: int = 0,
    ) -> None:
        """初始化。

        Args:
            armor: `ArmorPool`（读写 wp/target_idx/formation_id）。
            formations: `FormationAIPool`（读写 order/intent/target_zone）。
            speed: 每单位速度数组（movement hook 消费；None = 不改速度）。
            pathfinder: 可选寻路服务（`find_path((sx,sy),(tx,ty))`）；缺失走直线。
            ammo: 可选 `AmmoManager`。
            fuel: 可选 `FuelManager`。
            events: 可选 `EventBus`（发 command_* 事件供 HUD/记录）。
            player_side: 本方绝对阵营（`Side` 枚举值）；解析回退与敌重心用。
        """
        self.armor = armor
        self.formations = formations
        self.speed = speed
        self.pathfinder = pathfinder
        self.ammo = ammo
        self.fuel = fuel
        self.events = events
        self.player_side = int(player_side)
        self._aliases: dict[str, int] = {}
        self._name_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 装配期注册
    # ------------------------------------------------------------------

    def register_alias(self, alias: str, formation_id: int) -> None:
        """绑定叙事名 → 编队索引（叙事层 YAML 的 target_id 用叙事名）。"""
        self._aliases[str(alias)] = int(formation_id)

    def set_unit_names(self, unit_names: dict[int, str]) -> None:
        """注入单位名表（armor_idx → 名称），供 target_id 精确匹配。"""
        self._name_to_idx = {
            str(name): int(idx) for idx, name in unit_names.items()
        }

    # ------------------------------------------------------------------
    # 消费
    # ------------------------------------------------------------------

    def handle_command(self, cmd: Command) -> bool:
        """消费一条指令；返回是否被本路由处理（False = loop 回落到通用事件）。"""
        if cmd.type == CommandType.HOLD:
            return self._hold(cmd)
        if cmd.type == CommandType.REDEPLOY:
            return self._redeploy(cmd)
        if cmd.type == CommandType.RESUPPLY:
            return self._resupply(cmd)
        if cmd.type == CommandType.DISENGAGE:
            return self._disengage(cmd)
        return False   # STRIKE 等由 ArtilleryAI/其他子系统处理

    # ------------------------------------------------------------------
    # HOLD：固守（停止机动，仍可开火）
    # ------------------------------------------------------------------

    def _hold(self, cmd: Command) -> bool:
        targets = self._resolve_formations(cmd.target_id)
        if not targets:
            return False
        for fidx in targets:
            self.formations.order[fidx] = int(UnitOrder.DEFEND)
            self.formations.intent[fidx] = int(FormationIntent.DEFEND)
            for i in self._members(fidx):
                self.armor.set_waypoints(i, [])      # 清路标点 → 停止机动
                self.armor.target_idx[i] = -1        # 下一 AI tick 重新分配射程内目标
                if self.speed is not None:
                    self.speed[i] = 0.0
        self._emit("command_hold", {
            "formations": targets,
            "target_id": cmd.target_id,
            "formation": cmd.params.get("formation"),
        })
        return True

    # ------------------------------------------------------------------
    # REDEPLOY：转移 / 推进
    # ------------------------------------------------------------------

    def _redeploy(self, cmd: Command) -> bool:
        targets = self._resolve_formations(cmd.target_id)
        if not targets:
            return False
        point = self._redeploy_point(cmd, targets)
        if point is None:
            return False
        tx, ty = point
        speed_v = float(cmd.params.get("speed", self.DEFAULT_SPEED))
        for fidx in targets:
            self.formations.order[fidx] = int(UnitOrder.MANEUVER)
            self.formations.intent[fidx] = int(FormationIntent.ADVANCE)
            self.formations.target_zone[fidx] = (tx, ty, 0.0)
            for i in self._members(fidx):
                self._set_route(i, tx, ty)
                if self.speed is not None:
                    self.speed[i] = speed_v
        self._emit("command_redeploy", {
            "formations": targets,
            "target_id": cmd.target_id,
            "waypoint": [tx, ty],
            "formation": cmd.params.get("formation"),
        })
        return True

    def _redeploy_point(
        self, cmd: Command, targets: list[int]
    ) -> tuple[float, float] | None:
        """目标点：params.waypoint / coords（**世界坐标**）→ 否则编队既有目标区。"""
        wp = cmd.params.get("waypoint")
        if wp is None:
            wp = cmd.params.get("coords")
        if wp is not None:
            try:
                return float(wp[0]), float(wp[1])
            except (TypeError, ValueError, IndexError):
                logger.warning("REDEPLOY 目标点非法（需世界坐标 [x, y]）: %r", wp)
                return None
        zone = self.formations.target_zone[targets[0]]
        return float(zone[0]), float(zone[1])

    # ------------------------------------------------------------------
    # RESUPPLY：补给
    # ------------------------------------------------------------------

    def _resupply(self, cmd: Command) -> bool:
        if self.ammo is None and self.fuel is None:
            logger.warning("RESUPPLY 指令无补给管理器（装配层未注入 ammo/fuel）")
            return False
        targets = self._resolve_formations(cmd.target_id)
        if not targets:
            return False
        ratio = float(cmd.params.get("ratio", 1.0))
        supply_capacity = float(cmd.params.get("supply_capacity", 1e9))
        requested = 0
        for fidx in targets:
            for i in self._members(fidx):
                requested += self._request_one(self.ammo, i, ratio)
                requested += self._request_one(self.fuel, i, ratio)
        if requested == 0:
            self._emit("resupply", {"units": 0, "delivered": 0, "note": "已满"})
            return True
        delivered = 0
        if self.ammo is not None:
            delivered += int(self.ammo.process_requests(supply_capacity=supply_capacity))
        if self.fuel is not None:
            delivered += int(self.fuel.process_requests(supply_capacity=supply_capacity))
        self._emit("resupply", {"units": requested, "delivered": delivered})
        return True

    @staticmethod
    def _request_one(manager: Any | None, idx: int, ratio: float) -> int:
        """向一个后勤管理器下「补满到 cap 的 ratio 比例」请求；返回 1/0。"""
        if manager is None:
            return 0
        state = manager.state
        if not bool(state.active[idx]):
            return 0
        need = max(0.0, float(state.capacity[idx]) - float(state.reserve[idx])) * ratio
        if need <= 0.0:
            return 0
        manager.request_supply(idx, need, priority=1)
        return 1

    # ------------------------------------------------------------------
    # DISENGAGE：脱离接触
    # ------------------------------------------------------------------

    def _disengage(self, cmd: Command) -> bool:
        targets = self._resolve_formations(cmd.target_id)
        if not targets:
            return False
        speed_v = float(cmd.params.get("speed", self.DEFAULT_SPEED))
        dist = float(cmd.params.get("distance", self.WITHDRAW_DIST))
        ex, ey = self._enemy_centroid()
        for fidx in targets:
            members = self._members(fidx)
            if not members:
                continue
            cx = float(np.mean(self.armor.pos[members, 0]))
            cy = float(np.mean(self.armor.pos[members, 1]))
            self.formations.order[fidx] = int(UnitOrder.RETREAT)
            self.formations.intent[fidx] = int(FormationIntent.RETREAT)
            dx, dy = cx - ex, cy - ey
            norm = float(np.hypot(dx, dy))
            if norm < 1e-6:                      # 与敌重心重合 → 任意方向撤离
                dx, dy, norm = 1.0, 0.0, 1.0
            tx = cx + dx / norm * dist
            ty = cy + dy / norm * dist
            self.formations.target_zone[fidx] = (tx, ty, 0.0)
            for i in members:
                self._set_route(i, tx, ty)
                if self.speed is not None:
                    self.speed[i] = speed_v
        self._emit("command_disengage", {
            "formations": targets,
            "target_id": cmd.target_id,
        })
        return True

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _members(self, fidx: int) -> list[int]:
        """编队成员装甲索引（活跃）。"""
        a = self.armor
        sel = np.flatnonzero(a.active & (a.formation_id == int(fidx)))
        return [int(i) for i in sel]

    def _resolve_formations(self, target_id: str) -> list[int]:
        """target_id → 编队索引列表（见模块 docstring 解析顺序）。"""
        f = self.formations
        tid = (target_id or "").strip()
        if tid:
            if tid in self._aliases:
                fidx = self._aliases[tid]
                if 0 <= fidx < f.capacity and bool(f.active[fidx]):
                    return [fidx]
            if tid.startswith("formation:"):
                try:
                    fidx = int(tid.split(":", 1)[1])
                except ValueError:
                    fidx = -1
                if 0 <= fidx < f.capacity and bool(f.active[fidx]):
                    return [fidx]
            idx = self._name_to_idx.get(tid)
            if idx is not None and bool(self.armor.active[idx]):
                fidx = int(self.armor.formation_id[idx])
                if 0 <= fidx < f.capacity and bool(f.active[fidx]):
                    return [fidx]
            logger.warning("指令 target_id 无法解析: %r（回退本方编队）", target_id)
        return self._own_ground_formations()

    def _own_ground_formations(self) -> list[int]:
        """本方全部地面编队（排除炮兵——炮兵机动由 ArtilleryAI 管）。"""
        f = self.formations
        out: list[int] = []
        for raw in f.active_indices():
            fidx = int(raw)
            if int(f.side[fidx]) != self.player_side:
                continue
            if int(f.role[fidx]) == int(UnitRole.ARTILLERY):
                continue
            out.append(fidx)
        return out

    def _enemy_centroid(self) -> tuple[float, float]:
        """敌方（非本方）活跃单位重心——脱离方向参考。"""
        a = self.armor
        sel = np.flatnonzero(a.active & (a.side != self.player_side))
        if sel.size == 0:
            return 0.0, 0.0
        return float(a.pos[sel, 0].mean()), float(a.pos[sel, 1].mean())

    def _set_route(self, idx: int, tx: float, ty: float) -> None:
        """设一条到 (tx, ty) 的路由（优先寻路，失败降级直线）。"""
        a = self.armor
        sx = float(a.pos[idx, 0])
        sy = float(a.pos[idx, 1])
        if self.pathfinder is not None:
            try:
                wps = self.pathfinder.find_path((sx, sy), (tx, ty))
            except Exception:
                logger.exception("寻路失败，回退直线: (%s,%s) → (%s,%s)", sx, sy, tx, ty)
                wps = None
            if wps:
                a.set_waypoints(idx, list(wps) + [(tx, ty)])
                return
        a.set_waypoints(idx, [(tx, ty)])

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(kind, data)
