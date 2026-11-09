"""地面指令路由测试（`simulation/command_router.py`）。

依据：`docs/story-architecture-review.md` §三.5 —— 修复前叙事层 HOLD/REDEPLOY
「零实际效果」（`loop._apply_command` 只 emit 一个 command 事件），本测试锁定修复。

覆盖：
    - HOLD：停止机动（清路标点 + 速度 0）但仍保留 DEFEND（可开火）
    - HOLD 未解析 target_id → 回退本方地面编队（排除炮兵/敌方）
    - REDEPLOY：设路标点 + 任务目标区 + 恢复速度
    - RESUPPLY：下单并结算（弹药/燃料补满）
    - DISENGAGE：背离敌重心机动（RETREAT）
    - STRIKE 不由本路由消费（返回 False，交 ArtilleryAI）
    - target_id 解析：别名 / 单位名 / formation:<idx>
"""

from __future__ import annotations

import numpy as np

from Airia.lulilo.entities.pools import ArmorPool, FormationAIPool
from Airia.lulilo.entities.side import Side
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.logistics.fuel import FuelManager
from Airia.lulilo.simulation.ai import FormationIntent, UnitOrder, UnitRole
from Airia.lulilo.simulation.command_router import GroundCommandRouter
from Airia.lulilo.simulation.commands import Command, CommandType


class _Events:
    """最小 EventBus 替身（记录 emit）。"""

    def __init__(self) -> None:
        self.log: list[tuple[str, dict]] = []

    def emit(self, kind: str, data: dict | None = None) -> None:
        self.log.append((kind, dict(data or {})))


def _spawn_formation(
    armor: ArmorPool,
    fidx: int,
    side: int,
    count: int,
    origin: tuple[float, float],
    speed: np.ndarray,
    waypoint: tuple[float, float] = (7500.0, 7500.0),
) -> list[int]:
    """为一个编队生成 count 辆装甲（位置顺序排开，带路标点与速度）。"""
    idxs: list[int] = []
    for k in range(count):
        aidx = armor.spawn(
            x=origin[0] + k * 6.0, y=origin[1] + k * 6.0, z=0.0,
            half_x=1.5, half_y=3.0, half_z=0.8, yaw=0.0,
            role=int(UnitRole.TANK),
        )
        armor.side[aidx] = side
        armor.formation_id[aidx] = fidx
        armor.set_waypoints(aidx, [waypoint])
        speed[aidx] = 5.0
        idxs.append(aidx)
    return idxs


def make_world() -> tuple[ArmorPool, FormationAIPool, np.ndarray, dict, _Events]:
    """构建两个地面编队（北约/华约）+ 一个北约炮兵编队 + 后勤。"""
    armor = ArmorPool(capacity=64)
    formations = FormationAIPool(capacity=8)
    speed = np.zeros(armor.capacity, dtype=np.float32)
    events = _Events()

    nato_f = formations.spawn(side=int(Side.NATO), role=int(UnitRole.TANK),
                              target_zone=(7500.0, 7500.0, 0.0))
    pact_f = formations.spawn(side=int(Side.PACT), role=int(UnitRole.TANK),
                              target_zone=(7500.0, 7500.0, 0.0))
    arty_f = formations.spawn(side=int(Side.NATO), role=int(UnitRole.ARTILLERY),
                              target_zone=(7100.0, 6900.0, 0.0))

    nato_units = _spawn_formation(armor, nato_f, int(Side.NATO), 4, (7400.0, 7400.0), speed)
    pact_units = _spawn_formation(armor, pact_f, int(Side.PACT), 2, (7600.0, 7600.0), speed)
    arty_units = _spawn_formation(armor, arty_f, int(Side.NATO), 2, (7100.0, 6900.0), speed)
    formations.member_count[nato_f] = len(nato_units)
    formations.member_count[pact_f] = len(pact_units)
    formations.member_count[arty_f] = len(arty_units)

    ammo = AmmoManager(capacity=armor.capacity)
    fuel = FuelManager(capacity=armor.capacity)
    for idx in nato_units + arty_units:
        ammo.state.set_unit(idx, cap=40.0, initial=20.0)   # 半载 → 可补给
        fuel.state.set_unit(idx, cap=100.0, initial=50.0)

    ctx = {
        "nato_f": nato_f, "pact_f": pact_f, "arty_f": arty_f,
        "nato_units": nato_units, "pact_units": pact_units, "arty_units": arty_units,
        "ammo": ammo, "fuel": fuel,
    }
    return armor, formations, speed, ctx, events


def make_router(armor, formations, speed, ctx, events, *, pathfinder=None):
    router = GroundCommandRouter(
        armor, formations, speed,
        pathfinder=pathfinder, ammo=ctx["ammo"], fuel=ctx["fuel"],
        events=events, player_side=int(Side.NATO),
    )
    router.register_alias("7th_armor", ctx["nato_f"])
    return router


# ---------------------------------------------------------------------------
# HOLD
# ---------------------------------------------------------------------------


class TestHold:
    def test_hold_stops_movement_but_keeps_defend(self) -> None:
        """HOLD = 固守：清路标点 + 速度 0，order/intent 落 DEFEND（仍可开火）。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)

        handled = router.handle_command(Command(
            CommandType.HOLD, "7th_armor", {"formation": "defensive_line"}
        ))

        assert handled is True
        fidx = ctx["nato_f"]
        assert formations.order[fidx] == int(UnitOrder.DEFEND)
        assert formations.intent[fidx] == int(FormationIntent.DEFEND)
        for i in ctx["nato_units"]:
            assert armor.wp_count[i] == 0        # 停止机动
            assert speed[i] == 0.0
            assert armor.target_idx[i] == -1     # 目标清空，下一 tick 重新分配
        # 敌方编队不受影响
        assert formations.order[ctx["pact_f"]] == 0
        kinds = [k for k, _ in events.log]
        assert "command_hold" in kinds

    def test_hold_unknown_target_falls_back_to_own_ground(self) -> None:
        """target_id 不可解析 → 回退本方地面编队（排除炮兵与敌方）。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)

        assert router.handle_command(Command(CommandType.HOLD, "7th_armor_company")) is True

        assert formations.intent[ctx["nato_f"]] == int(FormationIntent.DEFEND)
        # 炮兵编队不动（机动由 ArtilleryAI 管）
        assert formations.intent[ctx["arty_f"]] == 0
        # 敌方编队不动
        assert formations.intent[ctx["pact_f"]] == 0


# ---------------------------------------------------------------------------
# REDEPLOY
# ---------------------------------------------------------------------------


class TestRedeploy:
    def test_redeploy_sets_route_zone_and_speed(self) -> None:
        """REDEPLOY：设路标点 + 任务目标区 + 恢复速度 + ADVANCE。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)

        handled = router.handle_command(Command(
            CommandType.REDEPLOY, "7th_armor",
            {"waypoint": [7300.0, 7200.0], "formation": "wedge"},
        ))

        assert handled is True
        fidx = ctx["nato_f"]
        assert formations.intent[fidx] == int(FormationIntent.ADVANCE)
        assert formations.order[fidx] == int(UnitOrder.MANEUVER)
        assert tuple(formations.target_zone[fidx][:2]) == (7300.0, 7200.0)
        for i in ctx["nato_units"]:
            assert armor.wp_count[i] >= 1
            assert armor.wp[i, armor.wp_count[i] - 1, 0] == 7300.0
            assert speed[i] == router.DEFAULT_SPEED
        assert "command_redeploy" in [k for k, _ in events.log]

    def test_redeploy_without_waypoint_uses_formation_zone(self) -> None:
        """无 waypoint → 用编队既有任务目标区。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        formations.target_zone[ctx["nato_f"]] = (7123.0, 7456.0, 0.0)

        assert router.handle_command(Command(CommandType.REDEPLOY, "7th_armor")) is True
        for i in ctx["nato_units"]:
            assert armor.wp[i, armor.wp_count[i] - 1, 0] == 7123.0

    def test_redeploy_illegal_waypoint_rejected(self) -> None:
        """waypoint 非法 → 返回 False（不消费，交由 loop 记通用事件）。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        assert router.handle_command(
            Command(CommandType.REDEPLOY, "7th_armor", {"waypoint": "north"})
        ) is False

    def test_redeploy_uses_pathfinder_when_available(self) -> None:
        """有寻路服务 → 路标点包含寻路结果 + 终点。"""
        class _Path:
            def find_path(self, start, end):
                return [(start[0] + 10.0, start[1] + 10.0)]

        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events, pathfinder=_Path())
        router.handle_command(Command(
            CommandType.REDEPLOY, "7th_armor", {"waypoint": [7300.0, 7200.0]}
        ))
        i = ctx["nato_units"][0]
        assert armor.wp_count[i] == 2
        assert armor.wp[i, 1, 0] == 7300.0


# ---------------------------------------------------------------------------
# RESUPPLY
# ---------------------------------------------------------------------------


class TestResupply:
    def test_resupply_tops_up_ammo_and_fuel(self) -> None:
        """RESUPPLY：按单位下单并结算 → 弹药/燃料补满。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        ammo, fuel = ctx["ammo"], ctx["fuel"]
        i = ctx["nato_units"][0]
        before_a = float(ammo.state.reserve[i])
        before_f = float(fuel.state.reserve[i])

        handled = router.handle_command(Command(CommandType.RESUPPLY, "7th_armor"))

        assert handled is True
        assert float(ammo.state.reserve[i]) > before_a
        assert float(fuel.state.reserve[i]) > before_f
        kinds = {k for k, _ in events.log}
        assert "resupply" in kinds

    def test_resupply_without_managers_returns_false(self) -> None:
        """未注入 ammo/fuel → 不消费（返回 False）。"""
        armor, formations, speed, ctx, events = make_world()
        router = GroundCommandRouter(armor, formations, speed,
                                     events=events, player_side=int(Side.NATO))
        router.register_alias("7th_armor", ctx["nato_f"])
        assert router.handle_command(Command(CommandType.RESUPPLY, "7th_armor")) is False


# ---------------------------------------------------------------------------
# DISENGAGE
# ---------------------------------------------------------------------------


class TestDisengage:
    def test_disengage_withdraws_away_from_enemy(self) -> None:
        """DISENGAGE：RETREAT + 目标点离敌重心更远。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        fidx = ctx["nato_f"]

        def dist_to_enemy_centroid() -> float:
            sel = np.flatnonzero(armor.active & (armor.side == int(Side.PACT)))
            ex, ey = armor.pos[sel, 0].mean(), armor.pos[sel, 1].mean()
            cx = armor.pos[ctx["nato_units"], 0].mean()
            cy = armor.pos[ctx["nato_units"], 1].mean()
            return float(np.hypot(cx - ex, cy - ey))

        before = dist_to_enemy_centroid()
        assert router.handle_command(Command(CommandType.DISENGAGE, "7th_armor")) is True

        assert formations.intent[fidx] == int(FormationIntent.RETREAT)
        assert formations.order[fidx] == int(UnitOrder.RETREAT)
        tx, ty = formations.target_zone[fidx][:2]
        sel = np.flatnonzero(armor.active & (armor.side == int(Side.PACT)))
        ex, ey = armor.pos[sel, 0].mean(), armor.pos[sel, 1].mean()
        after = float(np.hypot(tx - ex, ty - ey))
        assert after > before
        for i in ctx["nato_units"]:
            assert armor.wp_count[i] >= 1


# ---------------------------------------------------------------------------
# 解析与边界
# ---------------------------------------------------------------------------


class TestResolution:
    def test_strike_not_handled(self) -> None:
        """STRIKE 交 ArtilleryAI，本路由不消费。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        assert router.handle_command(
            Command(CommandType.STRIKE, "7th_armor", {"coords": (1.0, 2.0)})
        ) is False

    def test_resolve_by_unit_name(self) -> None:
        """单位名精确匹配 → 解析到其所属编队。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        first = ctx["nato_units"][0]
        router.set_unit_names({first: "M1A2 #00", ctx["pact_units"][0]: "T-14 #01"})

        assert router.handle_command(Command(CommandType.HOLD, "M1A2 #00")) is True
        assert formations.intent[ctx["nato_f"]] == int(FormationIntent.DEFEND)
        # 按单位名解析到敌方编队同样生效（不限于本方）
        assert router.handle_command(Command(CommandType.HOLD, "T-14 #01")) is True
        assert formations.order[ctx["pact_f"]] == int(UnitOrder.DEFEND)

    def test_resolve_by_formation_index_string(self) -> None:
        """`formation:<idx>` 形式直接命中编队。"""
        armor, formations, speed, ctx, events = make_world()
        router = make_router(armor, formations, speed, ctx, events)
        cmd = Command(CommandType.HOLD, f"formation:{ctx['nato_f']}")
        assert router.handle_command(cmd) is True
        assert formations.intent[ctx["nato_f"]] == int(FormationIntent.DEFEND)
