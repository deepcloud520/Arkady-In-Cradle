"""爆炸/毁伤提交池（Phase 6c 步骤 ③：combat/damage_submission.py）。

对应 docs/phase6-ai-pipeline.md §七「爆炸/毁伤提交池」：
    所有弹药的伤害模型（细锥射流/锥形射流/球形爆破/破片锥）先提交到一个池，
    再统一批量结算——既加速运算（Numba 向量化），又为渲染提供统一的爆炸数据源。

设计要点（docs §七.1/§七.2）：
    - 池是纯 SoA 数据结构（本模块不 import combat/engagement，被其调用）
    - 结算 = Numba 批量内核（球形/破片锥区域毁伤 O(E×N)，原地扣 armor.health）
      + Python 逐事件顶级毁伤（锥形/细锥直接目标，由 FiringControl._resolve_top_attack 处理）
    - 爆炸数据（drain_explosions）与伤害结算解耦：submit 时记录，渲染层随时消费
    - 扩容只在 Tick 间隙触发（同 ArmorPool._expand 惯例）

依赖方向：combat/engagement.py 调用本模块；本模块不反向依赖 simulation/ai.py
（依赖链 §十三：combat 不反向依赖 simulation）。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from numba import njit

from Airis.lulilo.core.logging import get_logger

logger = get_logger(__name__)

# 提交池容量（docs §七.2：MAX_DAMAGE_EVENTS = 4096）
MAX_DAMAGE_EVENTS = 4096


class DamageKind(IntEnum):
    """伤害模型几何类型（docs §七.1）。

    FINE_CONE  细锥射流（动能长杆侵彻：APFSDS/AP/APCR/PELE——直径≈弹径、深度=穿深）
    CONE       锥形射流（化学能 EFP/HEAT/攻顶 ATGM）
    SPHERE     球形爆破（HE/HE_FRAG/SAL/巡飞弹——半径 + 线性衰减）
    FRAG_CONE  破片锥（HE_FRAG/HE——方向 + 角度 + 碎片数）
    """
    FINE_CONE = 0
    CONE = 1
    SPHERE = 2
    FRAG_CONE = 3


# Numba 用 int 常量（IntEnum 在 njit 中报错）
_DK_FINE_CONE = int(DamageKind.FINE_CONE)
_DK_CONE = int(DamageKind.CONE)
_DK_SPHERE = int(DamageKind.SPHERE)
_DK_FRAG_CONE = int(DamageKind.FRAG_CONE)


@njit(cache=True)
def resolve_area_damage(
    kind: np.ndarray,          # (E,) int8 伤害模型（SPHERE / FRAG_CONE）
    center: np.ndarray,        # (E, 3) float32 事件中心（世界坐标）
    radius: np.ndarray,        # (E,) float32 球半径 / 锥射程
    strength: np.ndarray,      # (E,) float32 事件强度（球=中心毁伤量 / 破片=单破片伤害）
    direction: np.ndarray,     # (E, 3) float32 锥方向（单位向量）
    cone_angle: np.ndarray,    # (E,) float32 锥半角（rad）
    frag_count: np.ndarray,    # (E,) int32 破片数（本阶段确定性结算，不逐破片随机）
    active: np.ndarray,        # (E,) bool_ 事件活跃掩码
    armor_pos: np.ndarray,     # (N, 3) float32 装甲位置
    armor_health: np.ndarray,  # (N,) float32 装甲血量（原地修改，下限 0）
    armor_active: np.ndarray,  # (N,) bool_ 装甲存活掩码
) -> None:
    """批量区域毁伤结算（O(E×N)，docs §七.2）。

    - SPHERE：d ≤ radius 的装甲受 base × (1 - d/radius) 线性衰减毁伤
    - FRAG_CONE：锥内（与 direction 夹角 ≤ cone_angle 且距离 ≤ radius）装甲受
      frag_dmg × (1 - d/radius) 毁伤（确定性，不逐破片采样）
    直接原地扣 armor.health（下限 0）；摧毁/部位毁伤由调用方 Python 层补
    （FiringControl._finalize_area_kills）。
    """
    e = active.shape[0]
    n = armor_pos.shape[0]
    for i in range(e):
        if not active[i]:
            continue
        k = kind[i]
        if k == _DK_SPHERE:
            cx = center[i, 0]
            cy = center[i, 1]
            cz = center[i, 2]
            r = radius[i]
            base = strength[i]
            r2 = r * r
            for j in range(n):
                if not armor_active[j]:
                    continue
                dx = armor_pos[j, 0] - cx
                dy = armor_pos[j, 1] - cy
                dz = armor_pos[j, 2] - cz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > r2:
                    continue
                d = np.sqrt(d2)
                factor = 1.0 - d / r
                hp = armor_health[j] - base * factor
                armor_health[j] = hp if hp > 0.0 else 0.0
        elif k == _DK_FRAG_CONE:
            cx = center[i, 0]
            cy = center[i, 1]
            cz = center[i, 2]
            dxu = direction[i, 0]
            dyu = direction[i, 1]
            dzu = direction[i, 2]
            r = radius[i]
            frag = strength[i]
            cos_half = np.cos(cone_angle[i])
            r2 = r * r
            for j in range(n):
                if not armor_active[j]:
                    continue
                vx = armor_pos[j, 0] - cx
                vy = armor_pos[j, 1] - cy
                vz = armor_pos[j, 2] - cz
                d2 = vx * vx + vy * vy + vz * vz
                if d2 > r2 or d2 <= 1e-12:
                    continue
                d = np.sqrt(d2)
                cos_a = (vx * dxu + vy * dyu + vz * dzu) / d
                if cos_a < cos_half:
                    continue
                factor = 1.0 - d / r
                hp = armor_health[j] - frag * factor
                armor_health[j] = hp if hp > 0.0 else 0.0


class DamageSubmissionPool:
    """伤害提交池：弹药把「伤害事件」提交进池，Tick 末统一批量结算（docs §七.2）。

    数组布局（SoA，MAX_DAMAGE_EVENTS=4096）：
        kind       (E,)   int8    0 细锥 1 锥形 2 球形 3 破片锥（DamageKind）
        center     (E, 3) float32 事件中心（世界坐标）
        radius     (E,)   float32 球半径 / 锥射程 (m)
        pen_mm     (E,)   float32 事件强度：锥=穿深(mm) / 球=中心毁伤量 / 破片=单破片伤害
        target_idx (E,)   int32   直接目标装甲索引（-1=纯范围）
        direction  (E, 3) float32 锥方向（单位向量）
        cone_angle (E,)   float32 锥半角（rad）
        frag_count (E,)   int32   破片数
        top_factor (E,)   float32 攻顶因子（锥形事件专用：顶部 = 正面 × top_factor；
                                  0 = 非攻顶。docs §五.3）
        active     (E,)   bool_   事件活跃掩码

    Python 扩展（超出 docs 核心字段清单，为攻顶语义所需）：
        top_factor —— 锥形事件做「攻顶」需要的顶部因子（SFM 已直接解析，供步骤④
                      攻顶 ATGM / 测试直接提交使用）
    """

    def __init__(self, capacity: int = MAX_DAMAGE_EVENTS) -> None:
        self.capacity = int(capacity)
        self.kind = np.zeros(capacity, dtype=np.int8)
        self.center = np.zeros((capacity, 3), dtype=np.float32)
        self.radius = np.zeros(capacity, dtype=np.float32)
        self.pen_mm = np.zeros(capacity, dtype=np.float32)
        self.target_idx = np.full(capacity, -1, dtype=np.int32)
        self.direction = np.zeros((capacity, 3), dtype=np.float32)
        self.cone_angle = np.zeros(capacity, dtype=np.float32)
        self.frag_count = np.zeros(capacity, dtype=np.int32)
        self.top_factor = np.zeros(capacity, dtype=np.float32)
        self.active = np.zeros(capacity, dtype=np.bool_)
        self._free_stack: list[int] = list(range(capacity))
        # 爆炸数据（渲染消费；与伤害结算解耦，submit 时记录，drain_explosions 取走）
        self._explosions: list[dict[str, Any]] = []

    def submit_damage(
        self,
        kind: int | DamageKind,
        center: tuple[float, float, float] | np.ndarray,
        *,
        radius: float = 0.0,
        pen_mm: float = 0.0,
        target_idx: int = -1,
        direction: tuple[float, float, float] | np.ndarray | None = None,
        cone_angle: float = 0.0,
        frag_count: int = 0,
        top_factor: float = 0.0,
    ) -> int:
        """提交一条伤害事件，返回事件索引。池满则扩容后继续。

        Args:
            kind: DamageKind（0 细锥 / 1 锥形 / 2 球形 / 3 破片锥）。
            center: 事件中心世界坐标 (x, y, z)。
            radius: 球半径 / 锥射程 (m)。
            pen_mm: 事件强度（锥=穿深 mm / 球=中心毁伤量 / 破片=单破片伤害）。
            target_idx: 直接目标装甲索引（-1=纯范围）。
            direction: 锥方向（单位向量；None → +Z）。
            cone_angle: 锥半角 (rad)。
            frag_count: 破片数。
            top_factor: 攻顶因子（锥形事件专用；0=非攻顶，用正面）。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.active[idx] = True
        self.kind[idx] = int(kind)
        self.center[idx] = center
        self.radius[idx] = float(radius)
        self.pen_mm[idx] = float(pen_mm)
        self.target_idx[idx] = int(target_idx)
        if direction is None:
            self.direction[idx] = (0.0, 0.0, 1.0)
        else:
            self.direction[idx] = direction
        self.cone_angle[idx] = float(cone_angle)
        self.frag_count[idx] = int(frag_count)
        self.top_factor[idx] = float(top_factor)
        # 爆炸记录（渲染数据源，随提交记录；结算不消费它）
        self._explosions.append(
            {
                "kind": int(kind),
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "radius": float(radius),
            }
        )
        return idx

    def clear(self) -> None:
        """清空所有事件（结算后调用，索引归还 free_stack）。"""
        for i in self.active_indices():
            self.active[int(i)] = False
            self._free_stack.append(int(i))

    @property
    def active_count(self) -> int:
        """当前待结算事件数。"""
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        """返回所有 active=True 的索引数组。"""
        return np.flatnonzero(self.active)

    def drain_explosions(self) -> list[dict[str, Any]]:
        """消费爆炸数据（渲染数据源）：返回 + 清空已记录爆炸。

        返回每条 {kind, x, y, z, radius}；与伤害结算解耦，渲染层可随时调用
        （结算不消费本列表）。
        """
        out = self._explosions
        self._explosions = []
        return out

    def _expand(self, factor: float = 2.0) -> None:
        """内部扩容（Tick 间隙调用，原子引用替换）。"""
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(
                slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim)
            )
            new_arr[slices] = old[slices]
            return new_arr

        self.kind = _resize(self.kind, (new_cap,))
        self.center = _resize(self.center, (new_cap, 3))
        self.radius = _resize(self.radius, (new_cap,))
        self.pen_mm = _resize(self.pen_mm, (new_cap,))
        self.target_idx = _resize(self.target_idx, (new_cap,))
        self.direction = _resize(self.direction, (new_cap, 3))
        self.cone_angle = _resize(self.cone_angle, (new_cap,))
        self.frag_count = _resize(self.frag_count, (new_cap,))
        self.top_factor = _resize(self.top_factor, (new_cap,))
        self.active = _resize(self.active, (new_cap,))

        old_cap = self.capacity
        # 非零默认值补填（_resize 用 zeros；同 ArmorPool._expand 修复，dev-progress 2026-08-14）
        self.target_idx[old_cap:] = -1
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))
