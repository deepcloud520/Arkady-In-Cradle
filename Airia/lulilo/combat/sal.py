"""激光制导炸弹（SAL）系统（Phase 6b 步骤 9，docs/phase6-guidance-pipeline.md §7.5）。

吊舱照射 + 漫反射光斑 designate + 光斑存活管理：

    - **照射器（吊舱）**：`EmitterPool`（band=LASER, kind=2），主动激光源，
      父实体=装甲/地面单位。激光束是**锥形**（发散角 `beam_divergence_deg`），
      漏光只向前（吊舱背面检测不到）——可被激光告警（Phase 7）。
    - **漫反射光斑（designate）**：吊舱通过 LOS 射线检测后在目标坐标放置
      「激光漫反射 emitter」（kind=4，无父属性，不走 sync），`normal` 存目标
      表面法线（**半球漫反射**：导引头在背面/被遮挡检测不到）。一个吊舱可
      designate 多个光斑，或多个吊舱各照射一个。
    - **光斑存活**：每 tick 检查「吊舱 active 且 LOS（吊舱 ↔ 光斑）通」，
      不满足 → deactivate（§6.2 光斑分支）。
    - **炸弹**（seeker=SEMI_ACTIVE_LASER）由 `FiringControl.fire(aim_at=光斑坐标)`
      发射；导引头检测光斑（`_seeker_probe` SEMI_ACTIVE_LASER 分支，既有）；
      **命中判定（弹丸 vs 地形 → remove_span 炸桥）** 在
      `engagement.py::_check_sal_impact` / `_detonate_sal`（新增）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from Airia.lulilo.detection.los import line_of_sight
from Airia.lulilo.entities.pools import (
    EMITTER_KIND_ILLUMINATOR,
    EMITTER_KIND_SPOT,
    EMITTER_PARENT_ARMOR,
    EmitterBand,
    EmitterPool,
)

# ---------------------------------------------------------------------------
# SAL 常量（docs §7.5：炸弹命中/弹坑）
# ---------------------------------------------------------------------------

# 炸弹到达 guid_target_pos 附近（距离 < 此值）即引爆（m）
SAL_DETONATE_RADIUS = 6.0
# 弹坑破坏范围（remove_span 柱体半径 / 挖深，m）
SAL_CRATER_RADIUS = 3.0
SAL_CRATER_DEPTH = 2.0
# 弹坑向上削掉的高度（把桥面板顶面也挖掉，m）
SAL_CRATER_ABOVE = 1.0
# 爆炸近距毁伤：半径 = 3×弹坑半径，线性衰减到 0（对范围内装甲，简化）
SAL_BLAST_DMG = 60.0


class SALSystem:
    """激光制导炸弹系统：吊舱激活 + 漫反射光斑 designate + 光斑存活管理。

    炸弹发射/导引头检测/命中毁伤由 FiringControl 负责（本类不接触弹丸）。
    """

    def __init__(self, emitters: EmitterPool, terrain: Any | None = None) -> None:
        self.emitters = emitters
        self.terrain = terrain  # TerrainMap（heightfield）或 None（无地形单测默认通视）

    # ------------------------------------------------------------------
    # 照射器（吊舱）
    # ------------------------------------------------------------------

    def activate_illuminator(
        self, armor_idx: int, power: float = 1000.0
    ) -> int:
        """装甲单位携带激光照射器（吊舱，band=LASER, kind=2）。"""
        return self.emitters.activate(
            EMITTER_PARENT_ARMOR,
            armor_idx,
            float(power),
            int(EmitterBand.LASER),
            EMITTER_KIND_ILLUMINATOR,
        )

    # ------------------------------------------------------------------
    # 漫反射光斑
    # ------------------------------------------------------------------

    def designate(
        self,
        illum_idx: int,
        target_pos: tuple[float, float, float] | np.ndarray,
        power: float,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> int:
        """吊舱 LOS 检查通过后在目标坐标放置漫反射光斑（kind=4）。

        LOS 不通 / 吊舱失效返回 -1。一个吊舱可 designate 多个光斑。
        """
        em = self.emitters
        if illum_idx < 0 or illum_idx >= em.capacity or not em.active[illum_idx]:
            return -1
        if em.kind[illum_idx] != EMITTER_KIND_ILLUMINATOR:
            return -1
        tx = float(target_pos[0])
        ty = float(target_pos[1])
        tz = float(target_pos[2])
        if self.terrain is not None:
            hf = self.terrain.heightfield
            if not line_of_sight(
                float(em.pos[illum_idx, 0]),
                float(em.pos[illum_idx, 1]),
                float(em.pos[illum_idx, 2]),
                tx, ty, tz,
                hf.spans, hf.count, hf.width, hf.cs, hf.ch, hf.z_min,
            ):
                return -1
        return em.designate(target_pos, power, normal, illum_idx)

    # ------------------------------------------------------------------
    # 光斑存活管理（每 tick 调用）
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """光斑存活 = 吊舱 active 且 LOS（吊舱 ↔ 光斑）通；不满足 → deactivate。

        在 `_step_sub` 的 emitter 同步之后、导引头检测之前调用，保证
        `_seeker_probe`（SEMI_ACTIVE_LASER 分支）读到最新光斑状态。
        """
        em = self.emitters
        for i in list(em.active_indices()):
            if em.kind[i] != EMITTER_KIND_SPOT:
                continue
            illum = int(em.parent_idx[i])
            if illum < 0 or illum >= em.capacity or not em.active[illum]:
                em.deactivate(i)
                continue
            if not self._los(illum, i):
                em.deactivate(i)

    def _los(self, a: int, b: int) -> bool:
        """两个 emitter 之间的视线（无 terrain 默认通视，供无地形单测）。"""
        if self.terrain is None:
            return True
        hf = self.terrain.heightfield
        return line_of_sight(
            float(self.emitters.pos[a, 0]),
            float(self.emitters.pos[a, 1]),
            float(self.emitters.pos[a, 2]),
            float(self.emitters.pos[b, 0]),
            float(self.emitters.pos[b, 1]),
            float(self.emitters.pos[b, 2]),
            hf.spans, hf.count, hf.width, hf.cs, hf.ch, hf.z_min,
        )
