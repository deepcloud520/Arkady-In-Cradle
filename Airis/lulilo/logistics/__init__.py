"""engine.logistics 层：弹药 / 燃料消耗与补给请求—— Phase 3 骨架。

模块：
    ammo.py  弹药 SoA 状态（AmmoState）+ 消耗/补给（AmmoManager，Numba 批量）
    fuel.py  燃料 SoA 状态（FuelState）+ 消耗/补给（FuelManager，Numba 批量）

请求模型：SupplyRequest 入队，由 SimulationLoop 的 Tick 间隙处理。
"""

from Airis.lulilo.logistics.ammo import AmmoManager, AmmoState, SupplyRequest
from Airis.lulilo.logistics.fuel import FuelManager, FuelState

__all__ = [
    "AmmoManager",
    "AmmoState",
    "FuelManager",
    "FuelState",
    "SupplyRequest",
]
