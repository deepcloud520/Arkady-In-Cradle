"""传感器模型：雷达 / 红外 / ESM（AGENTS.md Phase 2）。

对应 AGENTS.md Phase 2「检测/传感器」：
- 雷达模型：RCS → 检测距离，受地形遮挡
- 红外模型：热信号 → 检测距离
- ESM：被动射频探测（雷达辐射源定位）

所有计算使用 NumPy/Numba，支持批量评估。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# 物理常量
# ---------------------------------------------------------------------------

# 雷达参考 RCS（m²）
RCS_TANK = 30.0       # 主战坦克
RCS_APC = 15.0        # 装甲运兵车
RCS_TRUCK = 10.0      # 卡车
RCS_HELICOPTER = 5.0  # 直升机
RCS_PERSON = 1.0      # 人员
# Phase 7 飞机 RCS（docs/phase7-air-pipeline.md §四.2）
RCS_FIGHTER = 5.0     # 战斗机
RCS_BOMBER = 10.0     # 轰炸机/攻击机（大翼展）

# 红外特征（相对值，无量纲）
IR_TANK_IDLE = 0.3       # 坦克怠速
IR_TANK_MOVING = 0.7     # 坦克行进中
IR_TANK_FIRING = 1.0     # 坦克开火后（炮管热）
IR_APC = 0.4
IR_HELICOPTER = 0.8
IR_PERSON = 0.1


# ---------------------------------------------------------------------------
# 雷达模型
# ---------------------------------------------------------------------------


@njit(cache=True)
def radar_detection_range(
    rcs: float,
    radar_power_db: float = 60.0,
    radar_gain_db: float = 30.0,
    noise_floor_db: float = -100.0,
    freq_ghz: float = 10.0,
) -> float:
    """雷达方程（简化）：计算对给定 RCS 目标的检测距离。

    $$R = \\sqrt[4]{\\frac{P_t G^2 \\lambda^2 \\sigma}{(4\\pi)^3 kTBF \\cdot SNR_{min}}}$$

    简化实现：使用参考距离和 RCS 四次方根关系。

    Args:
        rcs: 目标雷达截面积 (m²)。
        radar_power_db: 雷达发射功率 (dBm)。
        radar_gain_db: 天线增益 (dB)。
        noise_floor_db: 噪声基底 (dBm)。
        freq_ghz: 雷达频率 (GHz)。

    Returns:
        最大检测距离 (m)。
    """
    # 波长
    wavelength = 0.3 / freq_ghz  # c / f，单位 m

    # 简化雷达方程：R_max ∝ (P * G² * λ² * σ)^(1/4)
    # 参考：R_ref = 10km @ RCS=1m², P=60dBm, G=30dB, f=10GHz
    ref_range = 10000.0
    ref_rcs = 1.0

    # 四次方根关系：R ∝ σ^(1/4)
    factor = (rcs / ref_rcs) ** 0.25

    # 功率/增益修正（每 3dB 约 1.19× 距离）
    power_factor = 10.0 ** ((radar_power_db - 60.0) / 40.0)
    gain_factor = 10.0 ** ((radar_gain_db - 30.0) / 20.0)

    return ref_range * factor * power_factor * gain_factor


@njit(cache=True)
def radar_detection_probability(
    distance: float,
    max_range: float,
) -> float:
    """给定距离下的检测概率（S 曲线）。

    Args:
        distance: 当前距离 (m)。
        max_range: 最大检测距离 (m)。

    Returns:
        检测概率 [0, 1]。
    """
    if distance <= 0.0:
        return 1.0
    if distance >= max_range * 1.5:
        return 0.0

    # Logistic S 曲线：在 max_range 处概率为 0.5
    k = 10.0 / max_range  # 陡峭度
    x0 = max_range
    return 1.0 / (1.0 + np.exp(k * (distance - x0)))


# ---------------------------------------------------------------------------
# 红外模型
# ---------------------------------------------------------------------------


@njit(cache=True)
def ir_detection_range(
    ir_signature: float,
    atmospheric_transmission: float = 0.7,
    sensor_sensitivity: float = 1.0,
) -> float:
    """红外检测距离（简化模型）。

    大气衰减 + 平方反比定律。

    Args:
        ir_signature: 目标红外特征 (0~1，无量纲)。
        atmospheric_transmission: 大气透过率 (0~1)。
        sensor_sensitivity: 传感器灵敏度因子 (>0)。

    Returns:
        最大检测距离 (m)。
    """
    # 参考：IR=1.0、大气=0.7、灵敏度=1.0 → 5000m
    ref_range = 5000.0
    return ref_range * np.sqrt(ir_signature * atmospheric_transmission * sensor_sensitivity)


@njit(cache=True)
def ir_detection_probability(
    distance: float,
    max_range: float,
) -> float:
    """红外检测概率（随距离衰减）。"""
    if distance <= 0.0:
        return 1.0
    if distance >= max_range * 1.3:
        return 0.0
    # 比雷达更陡的衰减（红外受大气影响更大）
    k = 15.0 / max_range
    x0 = max_range * 0.7
    return 1.0 / (1.0 + np.exp(k * (distance - x0)))


# ---------------------------------------------------------------------------
# ESM（电子支援措施）
# ---------------------------------------------------------------------------


@njit(cache=True)
def esm_detection_range(
    emitter_power_db: float = 60.0,
    esm_sensitivity_db: float = -80.0,
    freq_ghz: float = 10.0,
) -> float:
    """ESM 检测距离（被动接收，单程传播）。

    ESM 比雷达有优势：只需要接收回波的单程路径（雷达需要往返）。

    Args:
        emitter_power_db: 辐射源功率 (dBm)。
        esm_sensitivity_db: ESM 接收机灵敏度 (dBm)。
        freq_ghz: 辐射频率 (GHz)。

    Returns:
        最大检测距离 (m)。
    """
    # 简化：单程 Friis 公式
    # R ∝ √(P_t * G_t * G_r * λ² / (SNR_min))
    # 参考：60dBm 发射源、-80dBm 灵敏度 → 50km
    ref_range = 50000.0

    # 功率余量每 6dB 翻倍距离
    margin_db = emitter_power_db - esm_sensitivity_db
    factor = 10.0 ** ((margin_db - 140.0) / 20.0)  # 140 = 60 - (-80)

    return ref_range * max(factor, 0.01)


@njit(cache=True)
def esm_detection_probability(
    distance: float,
    max_range: float,
) -> float:
    """ESM 检测概率。"""
    if distance <= 0.0:
        return 1.0
    if distance >= max_range * 1.2:
        return 0.0
    # ESM 比雷达更可靠（单程传播），衰减更缓
    k = 5.0 / max_range
    x0 = max_range
    return 1.0 / (1.0 + np.exp(k * (distance - x0)))


# ---------------------------------------------------------------------------
# 综合传感器评估（批量）
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def evaluate_sensor_detection(
    sensor_x: np.ndarray,        # (N,) float32  传感器 x
    sensor_y: np.ndarray,        # (N,) float32  传感器 y
    sensor_z: np.ndarray,        # (N,) float32  传感器 z
    target_x: np.ndarray,        # (M,) float32  目标 x
    target_y: np.ndarray,        # (M,) float32  目标 y
    target_z: np.ndarray,        # (M,) float32  目标 z
    sensor_type: np.ndarray,     # (N,) int32    0=雷达 1=红外 2=ESM
    target_rcs: np.ndarray,      # (M,) float32  目标 RCS
    target_ir: np.ndarray,       # (M,) float32  目标红外特征
    sensor_active: np.ndarray,   # (N,) bool_
    target_active: np.ndarray,   # (M,) bool_
    max_range_cache: np.ndarray,  # (N, M) float32  预计算的最大检测距离（输出）
) -> None:
    """批量评估所有传感器-目标对的检测距离。

    结果写入 max_range_cache (N, M)，供后续 LOS 过滤使用。

    Args:
        max_range_cache: 输出数组，[s, t] = 传感器 s 对目标 t 的最大检测距离。
    """
    n = sensor_x.shape[0]
    m = target_x.shape[0]

    for s in prange(n):
        if not sensor_active[s]:
            continue
        stype = int(sensor_type[s])
        for t in range(m):
            if not target_active[t]:
                max_range_cache[s, t] = 0.0
                continue

            # 欧几里得距离
            dx = target_x[t] - sensor_x[s]
            dy = target_y[t] - sensor_y[s]
            dz = target_z[t] - sensor_z[s]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            if stype == 0:  # 雷达
                max_r = radar_detection_range(target_rcs[t])
            elif stype == 1:  # 红外
                max_r = ir_detection_range(target_ir[t])
            else:  # ESM
                max_r = esm_detection_range()

            # 检测概率衰减施加于 max_range（距离若超过 1.5× max_r，概率≈0）
            max_range_cache[s, t] = max_r if dist <= max_r * 1.5 else 0.0


# ---------------------------------------------------------------------------
# Phase 6b 导引头检测内核（docs/phase6-guidance-pipeline.md §6.3/§6.5）
# ---------------------------------------------------------------------------
# 现有 evaluate_sensor_detection 服务「装甲平台的雷达/红外/ESM」链路，**不动它**。
# 制导检测新增三类独立内核（避免破坏既有装甲检测链路）：
#   - evaluate_seeker_detection       射频（主动雷达）导引头对装甲：RCS 四次方根 × (1-ECM)
#   - evaluate_esm_emitter_detection  ESM 对辐射源（band=RF 特例，docs §6.3）
#   - evaluate_passive_emitter_detection  被动接收（IR/RF/LASER 按 band 分派单程公式）
# 「被动接收电磁波」是同一个物理模型——红外/ESM/激光共用，仅波段公式不同。
# 检测概率（S 曲线）复用既有 radar/ir/esm_detection_probability。


@njit(cache=True, parallel=True)
def evaluate_seeker_detection(
    seeker_x: np.ndarray,        # (M,) 导引头位置 x（=弹丸 pos）
    seeker_y: np.ndarray,        # (M,)
    seeker_z: np.ndarray,        # (M,)
    seeker_type: np.ndarray,     # (M,) int32（SeekerType；本内核服务 ACTIVE_RADAR=1）
    seeker_active: np.ndarray,   # (M,) bool_
    target_x: np.ndarray,        # (N,) 目标位置 x（装甲）
    target_y: np.ndarray,        # (N,)
    target_z: np.ndarray,        # (N,)
    target_rcs: np.ndarray,      # (N,) RCS（反射特性）
    target_ecm: np.ndarray,      # (N,) ECM 干扰强度 0~1（压制检测距离）
    target_active: np.ndarray,   # (N,) bool_
    max_range_cache: np.ndarray,  # (M, N) float32 输出：最大检测距离（含 ECM 衰减后）
) -> None:
    """射频（主动雷达）导引头对装甲：RCS 四次方根 × (1 - ECM)。

    干扰闭环：目标 ecm 上升 → 检测距离下降 → 制导丢失（docs §6.5 陷阱 §八.6）。
    """
    for s in prange(seeker_x.shape[0]):
        if not seeker_active[s]:
            continue
        for t in range(target_x.shape[0]):
            if not target_active[t]:
                max_range_cache[s, t] = 0.0
                continue
            dx = target_x[t] - seeker_x[s]
            dy = target_y[t] - seeker_y[s]
            dz = target_z[t] - seeker_z[s]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            ecm = max(0.0, min(1.0, target_ecm[t]))
            max_r = radar_detection_range(target_rcs[t]) * (1.0 - ecm)
            max_range_cache[s, t] = max_r if dist <= max_r * 1.5 else 0.0


@njit(cache=True, parallel=True)
def evaluate_esm_emitter_detection(
    sensor_x: np.ndarray,        # (N,) ESM 传感器位置 x（= 装甲平台 pos）
    sensor_y: np.ndarray,        # (N,)
    sensor_z: np.ndarray,        # (N,)
    sensor_active: np.ndarray,   # (N,) bool_
    emitter_x: np.ndarray,       # (M,) 辐射源位置 x
    emitter_y: np.ndarray,       # (M,)
    emitter_z: np.ndarray,       # (M,)
    emitter_power_db: np.ndarray,  # (M,) 发射功率 (dBm)
    emitter_active: np.ndarray,  # (M,) bool_
    max_range_cache: np.ndarray,  # (N, M) float32 输出：sensor 对 emitter 的最大检测距离
) -> None:
    """ESM 被动检测辐射源（射频单程，docs §6.3）。

    - 主动导引头（kind=1）/ 平台雷达（kind=0）开机即被 ESM 截获 → 可被 ARM 反辐射。
    - 辐射源关机（deactivate → active=False）→ 立即从 ESM 视野消失（触发记忆模式）。
    """
    for s in prange(sensor_x.shape[0]):
        if not sensor_active[s]:
            continue
        for t in range(emitter_x.shape[0]):
            if not emitter_active[t]:
                max_range_cache[s, t] = 0.0
                continue
            dx = emitter_x[t] - sensor_x[s]
            dy = emitter_y[t] - sensor_y[s]
            dz = emitter_z[t] - sensor_z[s]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            max_r = esm_detection_range(emitter_power_db[t])
            max_range_cache[s, t] = max_r if dist <= max_r * 1.5 else 0.0


@njit(cache=True, parallel=True)
def evaluate_passive_emitter_detection(
    sensor_x: np.ndarray,        # (N,) 接收器位置 x（红外导引头/ESM/激光告警）
    sensor_y: np.ndarray,        # (N,)
    sensor_z: np.ndarray,        # (N,)
    sensor_band: np.ndarray,     # (N,) int32 接收波段（0=RF 1=IR 2=LASER）
    sensor_active: np.ndarray,   # (N,) bool_
    emitter_x: np.ndarray,       # (M,) 辐射源位置 x
    emitter_y: np.ndarray,       # (M,)
    emitter_z: np.ndarray,       # (M,)
    emitter_cur_power: np.ndarray,  # (M,) 当前强度（红外=当前热强度含残热；RF=功率）
    emitter_band: np.ndarray,    # (M,) int32 波段
    emitter_active: np.ndarray,  # (M,) bool_
    max_range_cache: np.ndarray,  # (N, M) float32 输出：最大检测距离
) -> None:
    """被动接收电磁波（IR/RF/LASER 统一：按波段分派单程公式，docs §6.5）。

    - 红外（band=IR）：平方反比，读 `cur_power`（当前热强度，含残热/殉爆脉冲）。
    - 射频（band=RF）：ESM 单程（= evaluate_esm_emitter_detection 的 band 特例）。
    - 激光（band=LASER）：直线可及（激光告警；半球限制由调用方按 normal 过滤）。
    - 波段不匹配 → 检测不到（红外诱饵需显式假 IR emitter 才能欺骗）。
    """
    for s in prange(sensor_x.shape[0]):
        if not sensor_active[s]:
            continue
        sband = int(sensor_band[s])
        for t in range(emitter_x.shape[0]):
            if not emitter_active[t]:
                max_range_cache[s, t] = 0.0
                continue
            if int(emitter_band[t]) != sband:
                max_range_cache[s, t] = 0.0
                continue
            dx = emitter_x[t] - sensor_x[s]
            dy = emitter_y[t] - sensor_y[s]
            dz = emitter_z[t] - sensor_z[s]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if sband == 1:  # IR：红外平方反比（当前热强度）
                max_r = ir_detection_range(emitter_cur_power[t])
            elif sband == 0:  # RF：ESM 单程
                max_r = esm_detection_range(emitter_cur_power[t])
            else:  # LASER：激光直线可及（简化，半球/FOV 由上层过滤）
                max_r = 20000.0
            max_range_cache[s, t] = max_r if dist <= max_r * 1.5 else 0.0
