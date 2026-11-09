"""穿透 / 毁伤解析（AGENTS.md Phase 3：combat/damage.py）。

对应 docs/phase3-combat-design.md §三~§五 + docs/penetration-algorithms-design.md：
    - 斜距效应：armor_eff = armor_mm / cos(impact_angle) × ARMOR_EFFECTIVENESS
    - 跳弹判定：入射角 > 75° 必跳弹；>60° 且弹速低于阈值 → 高概率跳弹
    - 穿透判定（分派管线）：按弹种选算法——APFSDS 走 A-T 两段式长杆
      （penetration_at_erosion，综述 Eq.13~17）、其余动能弹走 DeMarre 速度标度
      （penetration_demarre）、化学能固定穿深；输入含「随距离衰减的速度」
      （impact_velocity）。旧 resolve_penetration_single 保留作兼容路径。
    - 后效毁伤：崩落 / 起火 / 弹药殉爆 / 结构损伤（映射 PartType）

与 SoA 映射（design doc §5.3）：
    armor.health  ← 结构损伤累积
    box.active    ← 部位毁伤（TURRET 毁 → 失去开火；AMMO_RACK 毁 → 殉爆秒杀）

性能目标（design doc §六）：2048 发同时开火 → 散布+穿透两内核合计 < 1ms。
Numba 内核只做数学；随机数由 RNGManager 预生成传入，保持可测。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

import numpy as np
from numba import njit, prange

from Airia.lulilo.combat.weapons import AmmoType
from Airia.lulilo.core.logging import get_logger

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


# ---------------------------------------------------------------------------
# 穿深算法分派（docs/penetration-algorithms-design.md）
# ---------------------------------------------------------------------------


class PenetrationModel(IntEnum):
    """穿深算法模型（按弹种分派；Numba 内核传 int 值，勿直接传枚举）。"""
    FIXED = 0        # 化学能/固定穿深（不衰减）
    DEMARRE = 1      # 德马尔速度标度（全口径 AP / APCR / APDS）
    AT_EROSION = 2   # A-T 两段式长杆（APFSDS，侵蚀 + 刚性残杆）
    LO_APPROX = 3    # 简化 Lanz-Odermatt（可选快速近似/标定参考）


# 靶侧长杆侵彻参数（Tate 1986 标定；docs/penetration-algorithms-design.md §7.2）。
# 键 = ArmorType 整数值；值 = (靶密度 ρt kg/m³, 靶体阻力 Rt Pa)。
# RHA: BHN≈350 → σyt=1.47GPa → Rt=σyt(2/3+ln(0.57Et/σyt))≈7.4GPa。
# COMPOSITE/REACTIVE/SPACED 为等效待标定值。
TARGET_ARMOR_MODEL: dict[int, tuple[float, float]] = {
    ArmorType.RHA:       (7850.0, 7.4e9),
    ArmorType.COMPOSITE: (4500.0, 8.5e9),
    ArmorType.REACTIVE:  (7800.0, 5.5e9),
    ArmorType.SPACED:    (7850.0, 4.0e9),
}

# 化学能/爆炸系弹种（穿深固定，不随距离/速度衰减）
_CHEMICAL_AMMO_TYPES = frozenset(
    {
        int(AmmoType.HEAT), int(AmmoType.HE), int(AmmoType.ATGM),
        int(AmmoType.HESH), int(AmmoType.HE_FRAG), int(AmmoType.PELE),
        int(AmmoType.SAPHEI), int(AmmoType.SENSOR_FUZED),
    }
)

# Numba 内核内禁止 int(IntEnum)（nopython 不支持），用纯整型常量分派
_PEN_FIXED = int(PenetrationModel.FIXED)
_PEN_DEMARRE = int(PenetrationModel.DEMARRE)
_PEN_AT_EROSION = int(PenetrationModel.AT_EROSION)
_PEN_LO_APPROX = int(PenetrationModel.LO_APPROX)


def penetration_model_of(ammo_type: int, has_rod_params: bool) -> int:
    """弹种 → 穿深模型（docs/penetration-algorithms-design.md §三）。

    - 化学能/爆炸系 → FIXED（固定穿深）
    - APFSDS 且长杆参数齐（L>0 且 Yp>0）→ AT_EROSION（A-T 两段式）
    - 其余动能弹 / 缺参数的 APFSDS → DEMARRE（德马尔速度标度）
    """
    if ammo_type in _CHEMICAL_AMMO_TYPES:
        return int(PenetrationModel.FIXED)
    if ammo_type == int(AmmoType.APFSDS) and has_rod_params:
        return int(PenetrationModel.AT_EROSION)
    return int(PenetrationModel.DEMARRE)

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


# ---------------------------------------------------------------------------
# 穿深算法内核（docs/penetration-algorithms-design.md §四~§六）
# ---------------------------------------------------------------------------


@njit(cache=True)
def impact_velocity(v0: float, alpha: float, range_m: float) -> float:
    """闭合式速度衰减：v_imp = v0·exp(−α·range)。alpha<=0 或 v0<=0 时原速返回。"""
    if alpha <= 0.0 or v0 <= 0.0:
        return v0
    return v0 * np.exp(-alpha * range_m)


@njit(cache=True)
def penetration_demarre(pen0: float, v0: float, v_imp: float) -> float:
    """德马尔速度标度：P = pen0·(v_imp/v0)^(1/0.7)。"""
    if v0 <= 0.0:
        return pen0
    return pen0 * (max(v_imp, 0.0) / v0) ** 1.428571


@njit(cache=True)
def interface_velocity(v: float, mu2: float, Vc2: float) -> float:
    """A-T 界面侵彻速度（综述式(17)）：u = (v − √(μ²v² + (1−μ²)Vc²)) / (1−μ²)。

    单一真源：跳飞判据（§5.7）与穿深积分（penetration_at_erosion）共用。
    mu2 = ρt/ρp（μ²），Vc2 = Vc²。μ=1 时退化 u = (v²−Vc²)/(2v)。
    """
    disc = max(mu2 * v * v + (1.0 - mu2) * Vc2, 0.0)
    if abs(1.0 - mu2) > 1e-9:
        return max((v - np.sqrt(disc)) / (1.0 - mu2), 0.0)
    return max((v * v - Vc2) / (2.0 * v), 0.0)   # μ=1 退化


@njit(cache=True)
def penetration_at_erosion(
    L0: float,
    rho_p: float,
    rho_t: float,
    Yp: float,
    Rt: float,
    v_imp: float,
) -> float:
    """A-T 两段式长杆穿深（综述 Eq.13~17 + 终止条件，SI 内部，返回 mm）。

    段① 流体动力学侵蚀（v > v_end）：修正伯努利解界面速度 u，
        弹杆侵蚀 dl'/dt=-(v-u)、弹体减速 dv/dt=-Yp/(ρp·l')、穿深 dp/dt=u。
    段② 刚性残杆次级侵彻（仅 Rt < Yp）：剩余杆以 dv/dt=-Rt/(ρp·l') 减速侵彻。

    终止：Rt>Yp → v=Vc 处 u→0 侵彻停止；Rt<Yp → v=Vc/μ 处 u=v 转刚性段。
    物理上界：Yp=Rt=0 纯伯努利时 P → L·√(ρp/ρt)（流体动力学极限）。
    """
    if L0 <= 0.0 or rho_p <= 0.0 or v_imp <= 0.0:
        return 0.0
    mu2 = rho_t / rho_p                       # μ²
    mu = np.sqrt(mu2)                         # μ = √(ρt/ρp)
    Vc2 = max(0.0, 2.0 * abs(Rt - Yp) / rho_p)
    Vc = np.sqrt(Vc2)
    rigid = Rt < Yp                           # 仅杆强于靶才有刚性残杆次级
    v_end = Vc / mu if rigid else Vc

    L = L0 * 1e-3                             # m
    v = v_imp                                 # m/s
    P = 0.0                                   # m
    # 段① 流体动力学侵蚀
    while L > 1e-6 and v > v_end + 1e-3:
        u = interface_velocity(v, mu2, Vc2)            # Eq.(17)
        dL = -(v - u)                                  # 弹杆侵蚀
        dv = -Yp / (rho_p * L)                         # 弹体减速
        dP = u                                         # 穿深推进
        dt = 0.02 * L / max(v - u, 1e-3)               # 自适应：每步消耗 ~2% 杆长
        L += dL * dt
        v += dv * dt
        P += dP * dt
    # 段② 刚性残杆次级侵彻（仅 Rt < Yp）
    if rigid and v > 1e-3:
        while L > 1e-6 and v > 1e-3:
            dv = -Rt / (rho_p * L)                     # 刚性减速
            dP = v
            dt = 0.02 * L / max(v, 1e-3)
            v += dv * dt
            P += dP * dt
    return P * 1000.0                                 # → mm


@njit(cache=True)
def penetration_lanz_odermatt(
    L: float,
    rho_p: float,
    sigma_t: float,
    beta: float,
    k: float,
    v_imp: float,
) -> float:
    """简化 Lanz-Odermatt 长杆穿深（ammunition-expansion-design.md §5.1）。

    V* = v·√(ρp/σt)，P = L·V*^β/(k+V*^β)。σt 取靶侧 Rt。V*→∞ 时 P→L（速度饱和）。
    """
    if L <= 0.0 or rho_p <= 0.0 or sigma_t <= 0.0:
        return 0.0
    vstar = v_imp * np.sqrt(rho_p / sigma_t)
    vbeta = vstar ** beta
    return L * vbeta / (k + vbeta)


@njit(cache=True)
def compute_penetration(
    model: int,
    pen0: float,
    v0: float,
    v_imp: float,
    L0: float,
    rho_p: float,
    Yp: float,
    rho_t: float,
    Rt: float,
    lo_beta: float,
    lo_k: float,
) -> float:
    """按模型分派穿深（PenetrationModel 整数值）。"""
    if model == _PEN_DEMARRE:
        return penetration_demarre(pen0, v0, v_imp)
    if model == _PEN_AT_EROSION:
        return penetration_at_erosion(L0, rho_p, rho_t, Yp, Rt, v_imp)
    if model == _PEN_LO_APPROX:
        return penetration_lanz_odermatt(L0, rho_p, Rt, lo_beta, lo_k, v_imp)
    return pen0                                   # FIXED


@njit(cache=True)
def resolve_penetration_v2(
    model: int,
    pen_rha: float,
    armor_eff: float,
    impact_angle_deg: float,
    range_m: float,
    v0: float,
    alpha: float,
    L0: float,
    rho_p: float,
    Yp: float,
    rho_t: float,
    Rt: float,
    lo_beta: float,
    lo_k: float,
    out_margin: np.ndarray,     # (1,) float 穿透余量（>0 击穿）
    out_ricochet: np.ndarray,   # (1,) bool  是否跳弹
) -> None:
    """分派式单发穿透判定（docs/penetration-algorithms-design.md §六）。

    跳弹：
    - 长杆（AT_EROSION）半流体段（v_imp > Vc）→ Rosenberg 式(60) 速度相关判据
      tan²β > (ρp·v²/Rt)·(v+u)/(v−u)（§5.7）；v_imp ≤ Vc 刚性段回退固定阈值。
    - DeMarre/FIXED/LO → 固定阈值（>75° 必跳；60~75° 且 v_imp<500 → 跳）。
    """
    out_ricochet[0] = False
    v_imp = impact_velocity(v0, alpha, range_m)

    # 长杆（AT_EROSION）半流体段：Rosenberg 速度相关跳飞判据（§5.7）
    if model == _PEN_AT_EROSION and rho_p > 0.0 and Rt > 0.0:
        Vc2 = max(0.0, 2.0 * abs(Rt - Yp) / rho_p)
        Vc = np.sqrt(Vc2)
        if v_imp > Vc:
            u = interface_velocity(v_imp, rho_t / rho_p, Vc2)
            T = (rho_p * v_imp * v_imp / Rt) * ((v_imp + u) / max(v_imp - u, 1e-3))
            tan2 = np.tan(np.radians(abs(impact_angle_deg))) ** 2
            if tan2 > T:
                out_ricochet[0] = True
                out_margin[0] = -armor_eff
                return

    # 固定阈值（DeMarre/FIXED/LO；及长杆刚性段/退化回退）
    if abs(impact_angle_deg) > RICHOCHET_HARD_ANGLE:
        out_ricochet[0] = True
        out_margin[0] = -armor_eff
        return
    if abs(impact_angle_deg) > RICHOCHET_SOFT_ANGLE:
        if v_imp < 500.0:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return

    pen = compute_penetration(
        model, pen_rha, v0, v_imp, L0, rho_p, Yp, rho_t, Rt, lo_beta, lo_k
    )
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
        # ---- 穿深算法分派（docs/penetration-algorithms-design.md §八）----
        # model=None → 旧行为（DeMarre 距离衰减）；否则走分派管线。
        model: int | None = None,
        velocity_alpha: float = 0.0,
        penetrator_length_mm: float = 0.0,
        penetrator_density: float = 17500.0,
        penetrator_strength_pa: float = 0.0,
        target_density: float = 7850.0,
        target_strength_pa: float = 7.4e9,
        lo_beta: float = 0.67,
        lo_k: float = 3.0,
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
            model: PenetrationModel 整数值；None = 旧 DeMarre 距离衰减路径。
            velocity_alpha: 距离衰减系数 (1/m)。
            penetrator_*: 弹杆固有参数（L mm / ρp kg/m³ / Yp Pa）。
            target_density / target_strength_pa: 靶侧参数（ρt / Rt，ArmorType 查表）。

        Returns:
            {
                "armor_eff": 等效装甲,
                "margin": 穿透余量,
                "pen": 实际穿深（分派管线首层有效，供多层装甲递减）,
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
        if model is None:
            resolve_penetration_single(
                pen_mm_rha, armor_eff, impact_angle_deg, range_m,
                is_heat, drag_coefficient, muzzle_velocity,
                margin, ricochet,
            )
            pen = pen_mm_rha
        else:
            resolve_penetration_v2(
                int(model), pen_mm_rha, armor_eff, impact_angle_deg, range_m,
                muzzle_velocity, velocity_alpha,
                penetrator_length_mm, penetrator_density, penetrator_strength_pa,
                target_density, target_strength_pa, lo_beta, lo_k,
                margin, ricochet,
            )
            pen = float(margin[0]) + armor_eff  # 反推实际穿深（首层用）

        hdmg = np.zeros(1, dtype=np.float32)
        fire = np.zeros(1, dtype=np.bool_)
        cookoff = np.zeros(1, dtype=np.bool_)
        resolve_behind_armor_single(float(margin[0]), armor_eff, rng_u, hdmg, fire, cookoff)

        return {
            "armor_eff": float(armor_eff),
            "margin": float(margin[0]),
            "pen": float(pen),
            "ricochet": bool(ricochet[0]),
            "health_dmg": float(hdmg[0]),
            "fire": bool(fire[0]),
            "cookoff": bool(cookoff[0]),
        }

    @classmethod
    def with_builtin(cls) -> Self:
        """内置示例：M1A2 用 COMPOSITE，T-90 用 REACTIVE（design doc §八）。"""
        return cls()
