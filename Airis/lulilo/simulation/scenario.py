"""场景装配器（Phase 6c 步骤⑩，docs/phase6-ai-pipeline.md §十二.1）。

把 Phase 6c 各子系统（编队 AI / 单位 AI / 炮兵 / 反炮 / 通信簇 / RWR / 诱饵 /
爆炸提交池）装配成一个可无头运行的 `SimulationLoop`，供集成测试与压测复用。

数据流（docs §一）：指令 → CommandQueue → 调度链（movement→detection→ai→combat→
logistics）；AI 只决策不执行，开火/毁伤走 FiringControl + 伤害提交池。

⚠️ Phase 10 前不新增地图（docs §十二.1）：本装配器默认「无地形」——movement 用
简易二维路标点推进（无 WorldMesh/悬挂）；需要地形时由调用方注入
`world_mesh` + `speed`，本装配器改用 `build_armor_movement_hook`（真实悬挂管线）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel

from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.simulation.ai import (
    ArtilleryAI,
    CommunicationNetwork,
    FormationAI,
    MuzzleFlashLog,
    UnitRole,
)
from Airis.lulilo.simulation.loop import EventBus, SimulationLoop
from Airis.lulilo.simulation.state import SimulationState


# 简化碰撞箱（车体 + 炮塔；够开火链 + 部位毁伤用，与 phase6 测试同构）
_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]

# 角色 → 默认武器/弹药（无显式配置时）
_ROLE_DEFAULT: dict[int, tuple[str, str | None]] = {
    int(UnitRole.TANK): ("m256_120mm", None),
    int(UnitRole.ARTILLERY): ("m109_155mm", "m107_155mm_he_frag"),
    int(UnitRole.SCOUT): ("m256_120mm", None),
}


class FormationSpec(BaseModel):
    """一条编队配置（YAML/代码定义，docs §十二.1）。"""

    side: int = 0
    role: int = 0                            # UnitRole（编队角色）
    target_zone: tuple[float, float, float] = (0.0, 0.0, 0.0)
    comm: float = 1.0                        # 编队通信质量（中央集群门控）
    intent: int = 0                          # FormationIntent 初始意图


class UnitSpec(BaseModel):
    """一条单位配置（属编队，formation = 本场景 formations 列表索引）。"""

    side: int = 0
    role: int = 0                            # UnitRole
    x: float = 0.0
    y: float = 0.0
    formation: int = -1                      # formations 列表索引（-1 未编队）
    weapon_id: str | None = None             # None → 按 role 默认
    ammo_id: str | None = None               # None → 按 role 默认
    ammo: float = 40.0                       # 初始弹药量
    armor_mm: float = 600.0
    health: float = 100.0
    speed: float = 5.0                       # 单位机动速度 (m/s)


class ScenarioConfig(BaseModel):
    """一个完整战役场景（docs §十二.1：单位编成/初始位置/编队意图/胜利条件）。"""

    name: str = "scenario"
    seed: int = 0
    gravity: float = 9.81
    central_side: int = 0                    # 玩家阵营（中央集群 / 大面板可见方）
    formations: list[FormationSpec] = field(default_factory=list)
    units: list[UnitSpec] = field(default_factory=list)


@dataclass
class ScenarioRuntime:
    """装配产物：完整无头仿真运行时（pools + AI + loop）。"""

    config: ScenarioConfig
    armor: ArmorPool
    boxes: BoxPool
    projectiles: ProjectilePool
    ammo: AmmoManager
    fc: FiringControl
    events: EventBus
    formations: FormationAIPool
    muzzle_flashes: MuzzleFlashLog
    comm_network: CommunicationNetwork
    ai: FormationAI
    artilleries: list[ArtilleryAI]
    loop: SimulationLoop
    speed: np.ndarray
    formation_ids: list[int] = field(default_factory=list)   # config.formations 索引 → 池 fidx


def build_scenario(
    config: ScenarioConfig,
    *,
    world_mesh: Any | None = None,
) -> ScenarioRuntime:
    """装配一个可无头运行的场景（docs §十二.1）。

    - AI hook 组合：FormationAI.tick（编队决策 + 单位执行 + RWR/诱饵）+ 各阵营
      ArtilleryAI.tick（火力任务 + 反炮闭环）。
    - command_handler = 中央阵营炮兵 AI.handle_command（玩家 STRIKE 直达）。
    - movement：无 world_mesh 时用简易二维路标点推进；有 world_mesh 时走真实
      悬挂管线（build_armor_movement_hook）。
    """
    armor = ArmorPool(capacity=256)
    boxes = BoxPool(capacity=2048)
    projectiles = ProjectilePool(capacity=4096)
    ammo = AmmoManager(capacity=256)
    events = EventBus()
    formations = FormationAIPool(capacity=64)

    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, events=events,
        rng=RNGManager(seed=config.seed),
        gravity=config.gravity,
    )

    # ---- 编队 ----
    formation_ids: list[int] = []
    for fspec in config.formations:
        fidx = formations.spawn(
            side=fspec.side,
            role=fspec.role,
            target_zone=fspec.target_zone,
        )
        formations.comm[fidx] = float(fspec.comm)
        formations.intent[fidx] = int(fspec.intent)
        formation_ids.append(fidx)

    # ---- 单位 ----
    speed = np.zeros(armor.capacity, dtype=np.float32)
    for uspec in config.units:
        fidx = formation_ids[uspec.formation] if 0 <= uspec.formation < len(formation_ids) else -1
        role = int(uspec.role)
        idx = armor.spawn(
            uspec.x, uspec.y, 0.0,
            half_x=1.5, half_y=3.0, half_z=1.0,
            side=uspec.side, role=role, formation_id=fidx,
        )
        speed[idx] = float(uspec.speed)
        for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
            bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
            if int(part) == int(PartType.HULL):
                boxes.hp[bidx] = float(uspec.health)
                boxes.hp_max[bidx] = float(uspec.health)
                armor.bind_hull(idx, bidx)
        weapon_id, ammo_id = _ROLE_DEFAULT.get(role, ("m256_120mm", None))
        if uspec.weapon_id is not None:
            weapon_id = uspec.weapon_id
        if uspec.ammo_id is not None:
            ammo_id = uspec.ammo_id
        fc.set_unit_config(
            idx,
            UnitCombatConfig(
                weapon_id=weapon_id,
                ammo_id=ammo_id,
                armor_mm=float(uspec.armor_mm),
                crew_skill=0.9,
            ),
        )
        ammo.state.set_unit(idx, float(uspec.ammo), float(uspec.ammo))

    # ---- Phase 6c 子系统 ----
    muzzle_flashes = MuzzleFlashLog()
    comm_network = CommunicationNetwork(
        armor, formations,
        comm_threshold=0.5, central_side=config.central_side,
    )

    # 每个存在炮兵编队的阵营 → 一个 ArtilleryAI（共享闪光日志，反炮闭环）
    artilleries: list[ArtilleryAI] = []
    artillery_sides = {
        int(formations.side[f]) for f in formations.active_indices()
        if int(formations.role[f]) == int(UnitRole.ARTILLERY)
    }
    for side in sorted(artillery_sides):
        artilleries.append(
            ArtilleryAI(
                armor, formations, fc=fc, events=events,
                side=side, muzzle_flashes=muzzle_flashes,
            )
        )

    ai = FormationAI(
        armor, formations, fc=fc, boxes=boxes, events=events,
        artillery=None,  # 由组合 hook 统一驱动多阵营炮兵，避免重复 tick
        comm_network=comm_network,
    )

    # ---- loop 装配 ----
    loop = SimulationLoop(
        state=SimulationState(armor_pool=armor, formation_pool=formations)
    )
    loop.configure_default_rates()

    # 组合 ai hook：编队/单位 AI + 各阵营炮兵 AI
    def ai_hook(l: SimulationLoop, dt: float) -> None:
        ai.tick(l, dt)
        for art in artilleries:
            art.tick(l, dt)

    loop.attach("ai", ai_hook)
    loop.attach("combat", lambda l, dt: fc.tick(dt))
    loop.attach("movement", _make_movement_hook(armor, speed, world_mesh))

    # 玩家 STRIKE → 中央阵营炮兵 AI（若存在）
    if artilleries:
        central = next(
            (a for a in artilleries if a.side == config.central_side),
            artilleries[0],
        )
        loop.command_handler = central.handle_command

    return ScenarioRuntime(
        config=config,
        armor=armor,
        boxes=boxes,
        projectiles=projectiles,
        ammo=ammo,
        fc=fc,
        events=events,
        formations=formations,
        muzzle_flashes=muzzle_flashes,
        comm_network=comm_network,
        ai=ai,
        artilleries=artilleries,
        loop=loop,
        speed=speed,
        formation_ids=formation_ids,
    )


def _make_movement_hook(
    armor: ArmorPool,
    speed: np.ndarray,
    world_mesh: Any | None,
) -> Callable:
    """movement hook：无地形 → 简易二维路标点推进；有地形 → 真实悬挂管线。"""
    if world_mesh is not None:
        from Airis.lulilo.movement.integrate import build_armor_movement_hook

        return build_armor_movement_hook(armor, world_mesh, speed)

    def hook(_loop: Any, dt: float) -> None:
        for i in armor.active_indices():
            i = int(i)
            px = float(armor.pos[i, 0])
            py = float(armor.pos[i, 1])
            if int(armor.wp_count[i]) == 0:
                armor.vel[i] = (0.0, 0.0, 0.0)
                continue
            wi = int(armor.wp_idx[i])
            if wi >= int(armor.wp_count[i]):
                armor.wp_count[i] = 0
                armor.vel[i] = (0.0, 0.0, 0.0)
                continue
            tx = float(armor.wp[i, wi, 0])
            ty = float(armor.wp[i, wi, 1])
            dx = tx - px
            dy = ty - py
            d = math.hypot(dx, dy)
            step = float(speed[i]) * float(dt)
            if d <= step or d < 1e-6:
                armor.pos[i, 0] = tx
                armor.pos[i, 1] = ty
                armor.wp_idx[i] += 1
            else:
                armor.pos[i, 0] = px + dx / d * step
                armor.pos[i, 1] = py + dy / d * step
            # 速度 = 本 tick 位移 / dt（火控前置量读 armor.vel；z 不参与前置量）
            armor.vel[i, 0] = (float(armor.pos[i, 0]) - px) / float(dt)
            armor.vel[i, 1] = (float(armor.pos[i, 1]) - py) / float(dt)
            armor.vel[i, 2] = 0.0

    return hook
