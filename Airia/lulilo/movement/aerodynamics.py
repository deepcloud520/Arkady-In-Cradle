"""气动模型与点质量飞行积分（Phase 7，docs/phase7-air-pipeline.md §三）。

模型选型（design §二.1，兵棋行业标准切分）：
    - **点质量 3-DOF**：不做完整 6DoF 姿态动力学（舵面/惯性矩/角动量）。
      状态 = 位置 + 速度向量；姿态角（heading/pitch/bank）仅用于渲染与飞控反馈。
    - **「解 α」升阻闭环**（满足「不同迎风面的升力阻力」）：迎角 α 不是自由状态，
      而是**每帧由所需升力反解**——先按飞控指令算平衡升力 L_req，反推 α，再套
      升力/阻力曲线。大攻角 → 升力增大 + 诱导阻力激增的物理闭环自然成立。
    - **大气模型**：密度指数衰减 ρ(z) = ρ0·exp(−z/H)，动压 q = ½ρV²。

气动系数（per-机型 `AircraftConfig`，经 `AircraftAero.apply` 广播进 SoA 数组）：
    CL(α) = CL_α·α（CL_α≈5.7/rad 线性段），截断到 ±CL_max·dmg（结构损伤退化升力）
    CD(α) = CD0 + k_ind·CL²（抛物线阻力极曲线：零升阻力 + 诱导阻力）

点质量 EOM（比力形式，单位质量）：
    V̇  = a_T − a_D − g·sinγ
    γ̇  = (a_L·cosφ − g·cosγ) / V
    ψ̇  = a_L·sinφ / (V·cosγ)      # bank-to-turn；限幅 ±turn_rate_max
    ẋ = V·cosγ·cosψ,  ẏ = V·cosγ·sinψ,  ż = V·sinγ

约束：
    - 依赖方向 movement → 不 import combat（`AircraftAero.apply` 用鸭子类型读
      `cfg.wing_area_m2` 等字段，广播代码在 combat/aircraft.py 侧）。
    - Numba 对 IntEnum 报错 → 状态比较用模块级 int 字面量（对齐 guidance 层约定）。
    - 热路径向量化：本模块内核 `@njit(parallel=True)`；飞控/决策是 Python 冷路径。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 大气模型（docs/phase7-air-pipeline.md §3.1）
_RHO0 = 1.225        # 海平面密度 kg/m³
_ALTITUDE_SCALE = 8500.0  # 密度标高 m

# 气动
_CL_ALPHA = 5.7      # 升力线斜率 /rad（≈ 0.1/deg 线性段）
_DEG2RAD = np.pi / 180.0

# 坠毁状态（AircraftState.CRASHING=5；entities 层 enum，Numba 用 int 字面量）
_AIRCRAFT_STATE_TAKEOFF = 0
_AIRCRAFT_STATE_CRASHING = 5
_AIRCRAFT_STATE_PARKED = 6


@njit(inline="always", cache=True)
def _clamp(v: float, lo: float, hi: float) -> float:
    """Numba 内联限幅（避免在 prange 里写 min/max 长表达式）。"""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True, parallel=True)
def integrate_aircraft_motion(
    pos: np.ndarray,                # (N,3) float32 世界坐标（in/out）
    vel: np.ndarray,                # (N,3) float32 速度向量（in/out）
    heading: np.ndarray,            # (N,) float32 航向 ψ（in/out）
    pitch: np.ndarray,              # (N,) float32 体轴俯仰 θ=γ+α（in/out）
    bank: np.ndarray,               # (N,) float32 滚转 φ（in/out）
    throttle: np.ndarray,           # (N,) float32 油门 0~1（飞控写）
    phi_cmd: np.ndarray,            # (N,) float32 期望滚转 rad（飞控写）
    gamma_cmd: np.ndarray,          # (N,) float32 期望航迹角速率 γ̇ rad/s（飞控写）
    state: np.ndarray,              # (N,) uint8 AircraftState
    dmg_factor: np.ndarray,         # (N,) float32 结构损伤 0~1（升力/推力退化）
    active: np.ndarray,             # (N,) bool_
    wing_area: np.ndarray,          # (N,) float32 机翼面积 m²
    cl_max: np.ndarray,             # (N,) float32 最大升力系数
    cd0: np.ndarray,                # (N,) float32 零升阻力系数
    k_ind: np.ndarray,              # (N,) float32 诱导阻力因子
    thrust_max: np.ndarray,         # (N,) float32 最大推力 N
    mass: np.ndarray,               # (N,) float32 质量 kg
    max_g: np.ndarray,              # (N,) float32 过载限制 g
    v_min: np.ndarray,              # (N,) float32 失速速度 m/s
    v_max: np.ndarray,              # (N,) float32 最大平飞速度 m/s
    turn_rate_max: np.ndarray,      # (N,) float32 最大转弯率 rad/s
    climb_rate_max: np.ndarray,     # (N,) float32 最大爬升率 m/s
    fuel: np.ndarray,               # (N,) float32 归一化燃油 0~1（in/out）
    fuel_burn: np.ndarray,          # (N,) float32 燃油消耗比例 /s（per-机型）
    rho0: float,                    # 海平面密度（默认 1.225）
    altitude_scale: float,          # 密度标高（默认 8500）
    gravity: float,                 # 重力加速度（默认 9.81）
    dt: float,                      # 本帧时长 s
    stall_out: np.ndarray,          # (N,) bool_ out：失速标志（Python 层转 CRASHING）
) -> None:
    """把活跃飞机推进 dt 秒（点质量 EOM + 解 α 升阻，docs §3.2）。

    - CRASHING：升力≈0（失控）、推力=0，仅重力 + 水平惯性滑行（弹道坠落）。
    - 其余状态：升力反解 → α → CL/CD → 比力 EOM → 半隐式积分。
    - 失速：V < v_min·(1+0.3·(1−dmg)) → stall_out=True（Python 层转 CRASHING）。
    - 燃油：按 fuel_burn（每机型巡航消耗率）线性扣减。
    """
    n = pos.shape[0]
    for i in prange(n):  # noqa: F821
        if not active[i]:
            continue
        st = int(state[i])

        # ---- 坠毁：弹道坠落（升力≈0，推力=0，仅重力 + 水平惯性）----
        if st == _AIRCRAFT_STATE_CRASHING:
            vel[i, 2] -= gravity * dt
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            pos[i, 2] += vel[i, 2] * dt
            continue

        # ---- 驻停：静止（不积分，PARKED 终态）----
        if st == _AIRCRAFT_STATE_PARKED:
            continue

        # ---- 起飞滑跑（TAKOFF 且 v < 失速）：轮式支撑，纯水平加速 ----
        stall_v = v_min[i] * (1.0 + 0.3 * (1.0 - dmg_factor[i]))
        if st == _AIRCRAFT_STATE_TAKEOFF:
            vx0 = vel[i, 0]
            vy0 = vel[i, 1]
            hv = np.sqrt(vx0 * vx0 + vy0 * vy0)
            v_cur = hv
            if v_cur < stall_v:
                a_t = throttle[i] * thrust_max[i] * dmg_factor[i] / mass[i]
                v2 = v_cur + a_t * dt
                if v2 > v_max[i]:
                    v2 = v_max[i]
                if hv < 1e-3:
                    dx = np.cos(heading[i])
                    dy = np.sin(heading[i])
                else:
                    dx = vx0 / hv
                    dy = vy0 / hv
                vel[i, 0] = v2 * dx
                vel[i, 1] = v2 * dy
                vel[i, 2] = 0.0
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                pos[i, 2] = max(pos[i, 2], 1.0)   # 滑跑贴地
                heading[i] = np.arctan2(vel[i, 1], vel[i, 0]) if v2 > 1e-3 else heading[i]
                if fuel[i] > 0.0:
                    fuel[i] -= fuel_burn[i] * dt
                    if fuel[i] < 0.0:
                        fuel[i] = 0.0
                continue

        # ---- 当前飞行状态 ----
        vx = vel[i, 0]
        vy = vel[i, 1]
        vz = vel[i, 2]
        v = np.sqrt(vx * vx + vy * vy + vz * vz)
        if v < 1e-3:
            # 静止（起飞滑跑前）：水平加速，γ=0
            v = 1e-3
            gamma = 0.0
            psi = float(heading[i])
        else:
            gamma = np.arcsin(_clamp(vz / v, -1.0, 1.0))
            psi = np.arctan2(vy, vx)

        # ---- 大气 + 动压 ----
        z = pos[i, 2]
        if z < 0.0:
            z = 0.0
        rho = rho0 * np.exp(-z / altitude_scale)
        q = 0.5 * rho * v * v

        # ---- 升力反解（§3.1 核心：由 γ̇_cmd 反推 α）----
        # γ̇ = (L·cosφ)/(mV) − (g·cosγ)/V → L_req = m·(V·γ̇ + g·cosγ)/cosφ
        cos_phi = max(np.cos(phi_cmd[i]), 0.2)   # 大滚转时升力分量不足 → 保守放大 L_req
        l_req = mass[i] * (v * gamma_cmd[i] + gravity * np.cos(gamma)) / cos_phi
        clmax = cl_max[i] * dmg_factor[i]
        cl_req = l_req / max(q * wing_area[i], 1e-6)
        alpha = _clamp(cl_req / _CL_ALPHA, -clmax, clmax)
        cl = _CL_ALPHA * alpha
        cd = cd0[i] + k_ind[i] * cl * cl

        # ---- 比力 ----
        a_l = q * wing_area[i] * cl / mass[i]
        a_d = q * wing_area[i] * cd / mass[i]
        a_t = throttle[i] * thrust_max[i] * dmg_factor[i] / mass[i]

        # 过载限制（|a_L| ≤ max_g·g，限制盘旋/拉起）
        lim = max_g[i] * gravity
        if a_l > lim:
            a_l = lim
        elif a_l < -lim:
            a_l = -lim

        # ---- EOM ----
        vdot = a_t - a_d - gravity * np.sin(gamma)
        gdot = (a_l * np.cos(bank[i]) - gravity * np.cos(gamma)) / max(v, 1e-3)
        psidot = a_l * np.sin(bank[i]) / max(v * np.cos(gamma), 1e-3)
        if psidot > turn_rate_max[i]:
            psidot = turn_rate_max[i]
        elif psidot < -turn_rate_max[i]:
            psidot = -turn_rate_max[i]

        v2 = v + vdot * dt
        # 失速判定（TAKOFF 滑跑豁免：轮式支撑，v<v_min 不失速）
        stall = v_min[i] * (1.0 + 0.3 * (1.0 - dmg_factor[i]))
        if st != _AIRCRAFT_STATE_TAKEOFF and v2 < stall:
            stall_out[i] = True
            v2 = stall
        if v2 > v_max[i]:
            v2 = v_max[i]

        # 航迹角限幅：由爬升率上限反推（同爬升率下高速爬升角更小）
        climb_lim = np.arcsin(_clamp(climb_rate_max[i] / max(v2, 1e-3), -1.0, 1.0))
        gamma2 = _clamp(gamma + gdot * dt, -climb_lim, climb_lim)
        psi2 = psi + psidot * dt

        # ---- 燃油消耗（按机型巡航率线性扣减）----
        if fuel[i] > 0.0:
            fuel[i] -= fuel_burn[i] * dt
            if fuel[i] < 0.0:
                fuel[i] = 0.0

        # ---- 提交 ----
        vel[i, 0] = v2 * np.cos(gamma2) * np.cos(psi2)
        vel[i, 1] = v2 * np.cos(gamma2) * np.sin(psi2)
        vel[i, 2] = v2 * np.sin(gamma2)
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt
        heading[i] = psi2
        pitch[i] = gamma2 + alpha       # 体轴俯仰 = 航迹角 + 迎角
        bank[i] = phi_cmd[i]            # bank-to-turn：滚转直接跟命令（简化）


# ---------------------------------------------------------------------------
# per-index 气动数组容器（内核参数源；机型参数广播见 combat/aircraft.py）
# ---------------------------------------------------------------------------


class AircraftAero:
    """每架飞机的气动/包线参数数组（`AircraftPool` 的旁挂容器）。

    为什么独立于 AircraftPool：机型参数来自 `combat/aircraft.py::AircraftConfig`
    （combat 层），而 entities 层不能反向 import combat（AGENTS.md §4.1 依赖方向），
    故由本容器持有数值数组、由 combat 侧 `apply_aircraft_config()` 广播填充。
    扩容时与 `AircraftPool._expand` 同步调用 `resize`。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.wing_area = np.full(capacity, 30.0, dtype=np.float32)
        self.cl_max = np.full(capacity, 1.6, dtype=np.float32)
        self.cd0 = np.full(capacity, 0.02, dtype=np.float32)
        self.k_ind = np.full(capacity, 0.05, dtype=np.float32)
        self.thrust_max = np.full(capacity, 100000.0, dtype=np.float32)
        self.mass = np.full(capacity, 15000.0, dtype=np.float32)
        self.max_g = np.full(capacity, 9.0, dtype=np.float32)
        self.v_min = np.full(capacity, 80.0, dtype=np.float32)
        self.v_max = np.full(capacity, 750.0, dtype=np.float32)
        self.turn_rate_max = np.full(capacity, 18.0 * _DEG2RAD, dtype=np.float32)
        self.climb_rate_max = np.full(capacity, 150.0, dtype=np.float32)
        self.fuel_burn = np.full(capacity, 1.0 / 5400.0, dtype=np.float32)  # 90min 满油

    def apply(self, idx: int, cfg: Any) -> None:
        """把机型配置广播到一架飞机（鸭子类型读 `cfg` 属性，不 import combat）。

        `cfg` 须提供字段（与 `AircraftConfig` 对齐）：
            wing_area_m2 / cl_max / cd0 / k_ind / thrust_max / mass_kg /
            max_g / v_min_m_s / v_max_m_s / turn_rate_max_deg_s /
            climb_rate_m_s / fuel_capacity_s
        """
        self.wing_area[idx] = float(cfg.wing_area_m2)
        self.cl_max[idx] = float(cfg.cl_max)
        self.cd0[idx] = float(cfg.cd0)
        self.k_ind[idx] = float(cfg.k_ind)
        self.thrust_max[idx] = float(cfg.thrust_max)
        self.mass[idx] = float(cfg.mass_kg)
        self.max_g[idx] = float(cfg.max_g)
        self.v_min[idx] = float(cfg.v_min_m_s)
        self.v_max[idx] = float(cfg.v_max_m_s)
        self.turn_rate_max[idx] = float(cfg.turn_rate_max_deg_s) * _DEG2RAD
        self.climb_rate_max[idx] = float(cfg.climb_rate_m_s)
        self.fuel_burn[idx] = 1.0 / max(float(cfg.fuel_capacity_s), 1.0)

    def resize(self, capacity: int) -> None:
        """与 AircraftPool 扩容同步（新段回填默认值）。"""
        if capacity <= self.capacity:
            return
        old = self.capacity

        def _resize(old_arr: np.ndarray, fill: float) -> np.ndarray:
            new_arr = np.full(capacity, fill, dtype=old_arr.dtype)
            new_arr[:old] = old_arr
            return new_arr

        self.wing_area = _resize(self.wing_area, 30.0)
        self.cl_max = _resize(self.cl_max, 1.6)
        self.cd0 = _resize(self.cd0, 0.02)
        self.k_ind = _resize(self.k_ind, 0.05)
        self.thrust_max = _resize(self.thrust_max, 100000.0)
        self.mass = _resize(self.mass, 15000.0)
        self.max_g = _resize(self.max_g, 9.0)
        self.v_min = _resize(self.v_min, 80.0)
        self.v_max = _resize(self.v_max, 750.0)
        self.turn_rate_max = _resize(self.turn_rate_max, 18.0 * _DEG2RAD)
        self.climb_rate_max = _resize(self.climb_rate_max, 150.0)
        self.fuel_burn = _resize(self.fuel_burn, 1.0 / 5400.0)
        self.capacity = capacity
