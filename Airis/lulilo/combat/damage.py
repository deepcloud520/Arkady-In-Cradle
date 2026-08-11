"""穿透 / 毁伤解析（AGENTS.md Phase 3：combat/damage.py）。

对应 docs/phase3-combat-design.md §三~§五：
    - 斜距效应：armor_eff = armor_mm / cos(impact_angle) × ARMOR_EFFECTIVENESS
    - 跳弹判定：入射角 > 75° 必跳弹；>60° 且弹速低于阈值 → 高概率跳弹
    - 穿透判定：DeMarre 变体衰减（动能弹）/ 固定穿深（化学能）
    - 后效毁伤：崩落 / 起火 / 弹药殉爆 / 结构损伤（映射 PartType）

与 SoA 映射（design doc §5.3）：
    armor.health  ← 结构损伤累积
    box.active    ← 部位毁伤（TURRET 毁 → 失去开火；AMMO_RACK 毁 → 殉爆秒杀）

性能目标（design doc §六）：2048 发同时开火 → 散布+穿透两内核合计 < 1ms。
Numba 内核只做数学；随机数由 RNGManager 预生成传入，保持可测。
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numba import njit, prange

from Airis.lulilo.core.logging import get_logger

logger = get_logger(__name__)

# 跳弹阈值（度）
RICHOCHET_HARD_ANGLE = 75.0
RICHOCHET_SOFT_ANGLE = 60.0
# DeMarre 衰减下限
DECAY_FLOOR = 0.3
# Overmatch 上限
OVERMATCH_CAP = 3.0


# ---------------------------------------------------------------------------
# 装甲类型与有效性表
# ---------------------------------------------------------------------------


class ArmorType:
    """装甲类型（design doc §3.3，映射 CSV Protection_Level）。

    Note: 用普通 int 常量而非 IntEnum——ArmorType 直接作为 Numba 内核参数，
    而 Numba 对 IntEnum 的索引/int() 转换支持有限（2026-08-11 实测均报错）。
    """
    RHA = 0        # 均质钢
    COMPOSITE = 1  # 复合（对 HEAT ~2.5×）
    REACTIVE = 2   # 反应装甲（对 HEAT ~2.0×）
    SPACED = 3     # 间隙装甲


# ARMOR_EFFECTIVENESS[armor_type, is_heat] —— design doc §3.3 表格
# 列 0 = vs 动能弹 (KE)，列 1 = vs 化学能 (HEAT)
ARMOR_EFFECTIVENESS = np.asarray(
    [
        [1.0, 1.0],   # RHA
        [1.5, 2.5],   # COMPOSITE
        [1.0, 2.0],   # REACTIVE
        [0.9, 1.3],   # SPACED
    ],
    dtype=np.float32,
)

# 毁伤基础概率（design doc §5.2）
SPALL_BASE = 0.3
FIRE_BASE = 0.1
COOKOFF_BASE = 0.05


# ---------------------------------------------------------------------------
# 单发判定（Numba，可独立测试）
# ---------------------------------------------------------------------------


@njit(cache=True)
def compute_armor_eff(
    armor_mm: float,
    armor_type: int,
    impact_angle_deg: float,
    is_heat: bool,
) -> float:
    """斜距效应 + 装甲类型修正（design doc §3.1、§3.3）。

    Args:
        armor_mm: 标称装甲厚度 (mm)。
        armor_type: ArmorType 整数值。
        impact_angle_deg: 入射角（相对法线，度）。0°=正击。
        is_heat: 化学能弹。

    Returns:
        等效装甲厚度 (mm)。
    """
    cos_a = np.cos(np.radians(abs(impact_angle_deg)))
    # 余弦保护：入射角接近 90° 时 cos→0，等效厚度爆炸（配合跳弹判定使用）
    cos_a = max(cos_a, 0.05)
    eff = armor_mm / cos_a
    eff *= ARMOR_EFFECTIVENESS[armor_type, 1 if is_heat else 0]
    return eff


@njit(cache=True)
def resolve_penetration_single(
    pen_rha: float,
    armor_eff: float,
    impact_angle_deg: float,
    range_m: float,
    is_heat: bool,
    drag_coefficient: float,
    muzzle_velocity: float,
    out_margin: np.ndarray,     # (1,) float 穿透余量（>0 击穿）
    out_ricochet: np.ndarray,   # (1,) bool  是否跳弹
) -> None:
    """单发穿透判定（design doc §四）。

    动能弹：pen = pen_rha × max(0.3, 1 - drag×range/1e5) ** 1.5
    化学能：pen = pen_rha（不衰减）
    跳弹：  |angle|>75° 必跳弹；>60° 且速度低于阈值 → 跳弹
    """
    out_ricochet[0] = False
    if abs(impact_angle_deg) > RICHOCHET_HARD_ANGLE:
        out_ricochet[0] = True
        out_margin[0] = -armor_eff
        return
    if abs(impact_angle_deg) > RICHOCHET_SOFT_ANGLE:
        # 低速易跳弹（阈值简化为 500 m/s，Phase 3 迭代可调）
        if muzzle_velocity < 500.0:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return

    if is_heat:
        pen = pen_rha
    else:
        decay = max(DECAY_FLOOR, 1.0 - drag_coefficient * range_m / 100000.0)
        pen = pen_rha * decay ** 1.5

    out_margin[0] = pen - armor_eff


@njit(cache=True)
def resolve_behind_armor_single(
    margin: float,
    armor_eff: float,
    rng_u: float,              # [0,1) 均匀随机
    out_health_dmg: np.ndarray,  # (1,) 结构损伤
    out_fire: np.ndarray,        # (1,) bool 起火
    out_cookoff: np.ndarray,     # (1,) bool 殉爆
) -> None:
    """单发后效毁伤（design doc §五，映射 PartType）。

    overmatch = margin / max(1, armor_eff)，上限 3.0
        - 未击穿：仅累积结构损伤（margin ≤ 0）
        - 击穿：结构损伤随 overmatch 增大 + 起火/殉爆判定
    """
    om = min(OVERMATCH_CAP, margin / max(1.0, armor_eff))

    # 结构损伤：未击穿也累积
    if margin <= 0.0:
        # 未击穿：仅累积结构损伤（design doc §4.3「health 微降 + 士气影响」）。
        # 动能越大震动越大，但封顶——未击穿不会直接摧毁（多次命中可耗损）。
        dmg = min(15.0, 2.0 + 0.02 * abs(margin))
        out_health_dmg[0] = dmg
        out_fire[0] = False
        out_cookoff[0] = False
        return

    dmg = 15.0 + 35.0 * om          # 击穿：显著结构损伤
    out_health_dmg[0] = dmg
    out_fire[0] = rng_u < FIRE_BASE
    out_cookoff[0] = rng_u < FIRE_BASE + COOKOFF_BASE and rng_u >= FIRE_BASE


# ---------------------------------------------------------------------------
# 批量内核（并行，供整帧调用）
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def resolve_hit_dispersion(
    sigma: np.ndarray,         # (N,) 每发散布半径（已含全部修正因子）
    aim_pos: np.ndarray,       # (N, 3) 瞄准点
    rng: np.ndarray,           # (N, 2) 均匀随机 [0,1)
    out_hit_pos: np.ndarray,   # (N, 3) 实际落点（含高斯散布，2D）
) -> None:
    """高斯散布落点生成（design doc §六，Box-Muller）。

    落点 = 瞄准点 + 高斯散布（水平面内，z 不变）。
    """
    n = sigma.shape[0]
    for i in prange(n):
        theta = rng[i, 0] * 2.0 * np.pi
        r = sigma[i] * np.sqrt(-2.0 * np.log(max(rng[i, 1], 1e-12)))
        out_hit_pos[i, 0] = aim_pos[i, 0] + r * np.cos(theta)
        out_hit_pos[i, 1] = aim_pos[i, 1] + r * np.sin(theta)
        out_hit_pos[i, 2] = aim_pos[i, 2]


@njit(cache=True, parallel=True)
def resolve_penetration_batch(
    pen_rha: np.ndarray,         # (N,) 弹药标称穿深
    armor_eff: np.ndarray,       # (N,) 等效装甲
    impact_angle: np.ndarray,    # (N,) 入射角（度）
    range_m: np.ndarray,         # (N,) 射距
    is_heat: np.ndarray,         # (N,) bool 化学能
    drag_coefficient: np.ndarray,  # (N,)
    muzzle_velocity: np.ndarray,   # (N,)
    out_margin: np.ndarray,      # (N,) 穿透余量
    out_ricochet: np.ndarray,    # (N,) bool 跳弹
) -> None:
    """批量穿透判定（design doc §六）。"""
    n = pen_rha.shape[0]
    for i in prange(n):
        resolve_penetration_single(
            pen_rha[i], armor_eff[i], impact_angle[i], range_m[i],
            is_heat[i], drag_coefficient[i], muzzle_velocity[i],
            out_margin[i:i + 1], out_ricochet[i:i + 1],
        )


# ---------------------------------------------------------------------------
# 毁伤解析器（Python 封装）
# ---------------------------------------------------------------------------


class DamageResolver:
    """毁伤解析器：把一次命中（CollisionHit + 弹/甲参数）转化为 SoA 更新。

    骨架流程（design doc §七 开火链第 6~7 步）：
        resolve(hit, armor_idx) →
            compute_armor_eff → resolve_penetration_single → resolve_behind_armor_single
            → 更新 armor.health / box.active

    部位毁伤（box 级）在 Phase 3 正式迭代中实现；本骨架提供接口占位。
    """

    def __init__(self, armor_type_map: dict[int, int] | None = None) -> None:
        """初始化。

        Args:
            armor_type_map: {armor_idx: ArmorType} 静态映射（骨架期用 dict；
                正式版下沉为 SoA 数组 armor_armor_type (N,) uint8）。
        """
        self._armor_type_map: dict[int, int] = armor_type_map or {}

    def set_armor_type(self, armor_idx: int, armor_type: int) -> None:
        """为装甲设置装甲类型（ArmorType 整数值）。"""
        self._armor_type_map[armor_idx] = int(armor_type)

    def armor_type_of(self, armor_idx: int) -> int:
        return int(self._armor_type_map.get(armor_idx, int(ArmorType.RHA)))

    def resolve(
        self,
        *,
        armor_idx: int,
        armor_mm: float,
        impact_angle_deg: float,
        range_m: float,
        pen_mm_rha: float,
        is_heat: bool,
        drag_coefficient: float,
        muzzle_velocity: float,
        rng_u: float,
    ) -> dict[str, float | bool]:
        """解析一次命中，返回毁伤结果（不直接改池，由调用方提交）。

        Args:
            armor_idx: 被命中装甲索引（取装甲类型）。
            armor_mm: 命中部位标称装甲厚度 (mm)。
            impact_angle_deg: 入射角（度）。
            range_m: 射距 (m)。
            pen_mm_rha: 弹药 0m 穿深。
            is_heat: 化学能。
            drag_coefficient / muzzle_velocity: 弹药参数。
            rng_u: [0,1) 均匀随机数（后效判定）。

        Returns:
            {
                "armor_eff": 等效装甲,
                "margin": 穿透余量,
                "ricochet": 是否跳弹,
                "health_dmg": 结构损伤,
                "fire": 是否起火,
                "cookoff": 是否殉爆,
            }
        """
        armor_type = self.armor_type_of(armor_idx)
        armor_eff = compute_armor_eff(armor_mm, armor_type, impact_angle_deg, is_heat)

        margin = np.zeros(1, dtype=np.float32)
        ricochet = np.zeros(1, dtype=np.bool_)
        resolve_penetration_single(
            pen_mm_rha, armor_eff, impact_angle_deg, range_m,
            is_heat, drag_coefficient, muzzle_velocity,
            margin, ricochet,
        )

        hdmg = np.zeros(1, dtype=np.float32)
        fire = np.zeros(1, dtype=np.bool_)
        cookoff = np.zeros(1, dtype=np.bool_)
        resolve_behind_armor_single(float(margin[0]), armor_eff, rng_u, hdmg, fire, cookoff)

        return {
            "armor_eff": float(armor_eff),
            "margin": float(margin[0]),
            "ricochet": bool(ricochet[0]),
            "health_dmg": float(hdmg[0]),
            "fire": bool(fire[0]),
            "cookoff": bool(cookoff[0]),
        }

    @classmethod
    def with_builtin(cls) -> Self:
        """内置示例：M1A2 用 COMPOSITE，T-90 用 REACTIVE（design doc §八）。"""
        return cls()
