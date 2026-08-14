"""末敏弹管线（Phase 6b 步骤 8，docs/phase6-guidance-pipeline.md §七）。

覆盖：
    - SCAN 阶段**双模扫描内核** `scan_armor_detect`：毫米波反射（ArmorPool.rcs）+
      红外辐射（EmitterPool 红外 emitter 的 cur_power，含残热）**双达标**才判定命中
      ——抗诱饵（§7.2/§2.3）。
    - `SfmScanner` 扫描服务：按 ammo 分组参数调用内核（不新增 SoA 数组，陷阱 §八.9）。
    - 阶段切换（BALLISTIC → SCAN）在 `engagement.py::_update_guidance_stages`
      （既有，2026-08-14）；EFP 攻顶毁伤在 `engagement.py::_resolve_sfm_hit`
      （复用 DamageResolver，§7.3）。本模块只做「扫描探测」。

SCAN → DETONATE（§7.1）：
    扫描命中 → 立即 EFP 毁伤 + 销毁弹丸 + 关 emitter（一次事件，engagement 层执行）。
SCAN → 自毁：`age > scan_duration_s` → `_recycle_stale` 回收（既有，复用超龄路径）。

复杂度 O(M×N)（M≤32 × N≤2048 ≈ 6.5 万对/帧），远低于碰撞 3355 万对（§7.2）。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import njit, prange

from Airis.lulilo.combat.guidance import GuidanceStage
from Airis.lulilo.entities.pools import (
    EMITTER_KIND_IR_SOURCE,
    EmitterBand,
    EmitterPool,
)

# ---------------------------------------------------------------------------
# 双模扫描内核（docs §7.2）
# ---------------------------------------------------------------------------


@njit(cache=True)
def _ir_emitter_hot(
    armor_idx: int,
    ir_cur_power: np.ndarray,   # (E,) 红外 emitter 当前热强度（含残热）
    ir_parent: np.ndarray,      # (E,) 红外 emitter 的父装甲索引
    ir_active: np.ndarray,      # (E,) 红外 emitter 是否有效
    ir_threshold: float,
) -> bool:
    """装甲 j 是否有「cur_power ≥ ir_threshold」的有效红外 emitter（双模之一）。"""
    for e in range(ir_cur_power.shape[0]):
        if ir_active[e] and ir_parent[e] == armor_idx and ir_cur_power[e] >= ir_threshold:
            return True
    return False


@njit(cache=True, parallel=True)
def scan_armor_detect(
    proj_x: np.ndarray,          # (M,) SCAN 阶段末敏弹位置
    proj_y: np.ndarray,
    proj_z: np.ndarray,
    scan_radius: float,          # float（按 ammo 分组）
    armor_x: np.ndarray,         # (N,) 装甲位置
    armor_y: np.ndarray,
    armor_rcs: np.ndarray,       # (N,) 毫米波反射（ArmorPool.rcs）
    armor_active: np.ndarray,    # (N,)
    ir_emitter_cur_power: np.ndarray,  # (E,) 红外当前热强度（band=IR，含残热）
    ir_emitter_parent: np.ndarray,     # (E,) 红外辐射源的父装甲索引
    ir_emitter_active: np.ndarray,     # (E,) 红外辐射源是否有效
    mmw_threshold: float,        # 毫米波反射阈值
    ir_threshold: float,         # 红外辐射阈值
    out_target: np.ndarray,      # (M,) int32 输出：命中装甲索引（-1 无）
) -> None:
    """双模扫描：毫米波 + 红外**双达标**才判定命中（抗诱饵，§7.2）。"""
    m = proj_x.shape[0]
    for i in prange(m):
        best = -1
        for j in range(armor_x.shape[0]):
            if not armor_active[j]:
                continue
            dx = armor_x[j] - proj_x[i]
            dy = armor_y[j] - proj_y[i]
            if dx * dx + dy * dy > scan_radius * scan_radius:
                continue
            if armor_rcs[j] < mmw_threshold:
                continue                      # 毫米波反射不达标
            if not _ir_emitter_hot(
                j, ir_emitter_cur_power, ir_emitter_parent, ir_emitter_active, ir_threshold
            ):
                continue                      # 红外辐射不达标（含残热）
            best = j
            break
        out_target[i] = best


# ---------------------------------------------------------------------------
# 扫描服务
# ---------------------------------------------------------------------------


class SfmScanner:
    """末敏弹扫描服务：对 SCAN 阶段弹丸做双模扫描，返回 {proj_idx: armor_idx}。

    扫描参数（scan_radius / mmw_threshold / ir_threshold）**按 ammo 分组**传入内核
    （同 §4.1 guid_law 分组模式，陷阱 §八.9），不新增 SoA 数组。
    """

    def __init__(self, armor: Any, emitters: EmitterPool) -> None:
        self.armor = armor
        self.emitters = emitters

    def scan(
        self, projectiles: Any, ammo_of: Callable[[int], Any]
    ) -> dict[int, int]:
        """扫描所有 SCAN 阶段弹丸，返回命中映射 {proj_idx: armor_idx}。

        Args:
            projectiles: ProjectilePool（active / guid_stage / pos / ammo）。
            ammo_of: proj_idx → AmmoConfig（engagement.FiringControl._ammo_of）。
        """
        p = projectiles
        scan_idx = np.flatnonzero(
            p.active & (p.guid_stage == int(GuidanceStage.SCAN))
        )
        if scan_idx.size == 0:
            return {}

        em = self.emitters
        a = self.armor
        # 红外 emitter 子集（band=IR & kind=3）；父死亡残热衰减期间 cur_power 仍有效
        ir_mask = (
            em.active
            & (em.band == int(EmitterBand.IR))
            & (em.kind == EMITTER_KIND_IR_SOURCE)
        )
        ir_idx = np.flatnonzero(ir_mask)

        # 按 ammo 注册表索引分组（同 ammo 同扫描参数）
        grouped: dict[int, list[int]] = {}
        for proj in scan_idx:
            ammo = ammo_of(int(proj))
            if ammo is None:
                continue
            grouped.setdefault(int(p.ammo[proj]), []).append(int(proj))
        if not grouped:
            return {}

        hits: dict[int, int] = {}
        for _key, projs in grouped.items():
            ammo = ammo_of(projs[0])
            if ammo is None:
                continue
            m = len(projs)
            px = np.empty(m, dtype=np.float32)
            py = np.empty(m, dtype=np.float32)
            pz = np.empty(m, dtype=np.float32)
            for k, idx in enumerate(projs):
                px[k] = p.pos[idx, 0]
                py[k] = p.pos[idx, 1]
                pz[k] = p.pos[idx, 2]
            out = np.full(m, -1, dtype=np.int32)
            scan_armor_detect(
                px, py, pz,
                float(ammo.scan_radius_m),
                a.pos[:, 0], a.pos[:, 1], a.rcs, a.active,
                em.cur_power[ir_idx], em.parent_idx[ir_idx], em.active[ir_idx],
                float(ammo.mmw_threshold), float(ammo.ir_threshold),
                out,
            )
            for k, idx in enumerate(projs):
                if out[k] >= 0:
                    hits[idx] = int(out[k])
        return hits
