"""模拟状态聚合与快照（AGENTS.md Phase 3：simulation/state.py）。

职责（AGENTS.md 3.1 三叉戟图中的「Simulation.get_state()」）：
    - SimulationState：持有引擎全部运行时状态（各 SoA 池 + 子系统管理器）
    - snapshot()：生成只读快照，供叙事层 / 渲染层消费，绝不暴露写路径

设计要点：
    - 叙事层通过 get_state() 读取，模拟层不感知叙事层存在（AGENTS.md 3.7）
    - 快照是「浅视图」：直接引用 SoA 数组（避免每帧深拷贝的拷贝开销），
      只读语义由约定保证；正式迭代可用 np.ndarray.setflags(write=False)
    - 挂载子系统为可选引用（骨架期未装配的子系统为 None）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationState:
    """引擎运行时状态的聚合容器（各子系统引用的所有者）。

    骨架期字段：
        armor_pool / box_pool / projectile_pool : entities.pools 的对象池
        ammo / fuel                              : logistics 管理器
        sim_time                                 : 已推进的模拟时间 (s)
        tick_count                               : 已执行的 Tick 数
        extra                                    : 子系统扩展挂载点（dict）
    """

    sim_time: float = 0.0
    tick_count: int = 0

    armor_pool: Any = None
    box_pool: Any = None
    projectile_pool: Any = None

    ammo: Any = None
    fuel: Any = None

    extra: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> "SimulationSnapshot":
        """生成只读快照（供叙事层 / 渲染层消费）。"""
        return SimulationSnapshot(
            sim_time=self.sim_time,
            tick_count=self.tick_count,
            pools={
                "armor": self.armor_pool,
                "box": self.box_pool,
                "projectile": self.projectile_pool,
            },
            logistics={
                "ammo": self.ammo,
                "fuel": self.fuel,
            },
            extra=dict(self.extra),
        )


@dataclass(frozen=True)
class SimulationSnapshot:
    """只读快照视图（AGENTS.md 3.7：叙事层只读模拟状态）。

    Note:
        - pools/logistics 为对象引用；调用方应只读其属性
        - extra 为深拷贝 dict，避免叙事层意外改动引擎挂载点
    """

    sim_time: float
    tick_count: int
    pools: dict[str, Any]
    logistics: dict[str, Any]
    extra: dict[str, Any]
