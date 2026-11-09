"""飞控层：把单位 AI 意图转成气动内核输入（Phase 7，docs/phase7-air-pipeline.md §3.3）。

职责：
    - **定高定航（CRUISE 核心）**：三个独立通道，带限幅的 PID 简化——
      ① 高度保持：垂直加速度指令 → 航迹角速率指令（小角近似）
      ② 速度保持：油门 PI（满油不再加）
      ③ 航路点转向：水平距离 < WP_RADIUS → wp_idx+1；φ_cmd 限幅 ±max_bank
    - **机动原语**（单位 AI 调用）：
      占位 lead pursuit（指向拦截点）/ 规避 kinematic defeat（急转弯+降高度）/
      俯冲-拉起（对地投弹程序化序列）。
    - **movement hook**：飞控写输入 → 气动积分 → 失速转 CRASHING → 机身碰撞箱
      与辐射源同步（docs §3.4）。

依赖方向：movement 层，不 import combat/simulation；`AircraftAero` 提供包线数组。
热路径纪律：本层是 Python 决策层（≤128 架 × 0.1s，可忽略），积分在 aerodynamics 内核。
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from Airia.lulilo.movement.aerodynamics import AircraftAero, integrate_aircraft_motion

# 状态常量（entities 层 IntEnum，Python 层可直接比较）
_AIRCRAFT_STATE_TAKEOFF = 0
_AIRCRAFT_STATE_CRUISE = 1
_AIRCRAFT_STATE_ENGAGE = 2
_AIRCRAFT_STATE_RTB = 3
_AIRCRAFT_STATE_LANDING = 4
_AIRCRAFT_STATE_CRASHING = 5
_AIRCRAFT_STATE_PARKED = 6

_DEG2RAD = math.pi / 180.0


def _wrap_pi(a: float) -> float:
    """角度差归一化到 (-π, π]。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class FlightControl:
    """逐架飞控：把状态/意图写成气动内核输入（throttle/phi_cmd/gamma_cmd）。"""

    # 三通道增益（docs §3.3；巡航 10min 高度/速度误差 < 1% 的标定值）
    KP_H = 0.02          # 高度比例 (1/s²)
    KD_H = 0.05          # 高度微分（垂直速度阻尼）
    KP_V = 0.004         # 速度比例 (1/s)
    KI_V = 0.0001        # 速度积分 (1/s²)
    KP_PSI = 2.0         # 航向比例
    MAX_BANK = 45.0 * _DEG2RAD    # 最大滚转 ±45°
    WP_RADIUS = 200.0    # 航路点到达半径 (m)

    # 机动原语参数
    DEFEAT_BANK = 60.0 * _DEG2RAD
    DEFEAT_DIVE_GAMMA = -0.06      # 规避下降航迹角速率 (rad/s)
    DIVE_GAMMA = -0.12             # 俯冲投弹
    PULLUP_GAMMA = 0.15            # 投后拉起

    def __init__(self, capacity: int, aero: AircraftAero) -> None:
        self.aero = aero
        self.capacity = int(capacity)
        self._v_int = np.zeros(capacity, dtype=np.float32)   # 油门积分项

    def resize(self, capacity: int) -> None:
        if capacity > self.capacity:
            new = np.zeros(capacity, dtype=np.float32)
            new[: self.capacity] = self._v_int
            self._v_int = new
            self.capacity = capacity

    # ------------------------------------------------------------------
    # 主入口：逐架分派
    # ------------------------------------------------------------------

    def tick(self, pool: Any, dt: float) -> None:
        """写全部活跃飞机的飞控输入（按 AircraftState 分派）。"""
        for i in pool.active_indices():
            i = int(i)
            st = int(pool.state[i])
            if st == _AIRCRAFT_STATE_CRASHING or st == _AIRCRAFT_STATE_PARKED:
                continue
            if st == _AIRCRAFT_STATE_TAKEOFF:
                self.takeoff_control(pool, i, dt)
                continue
            if st == _AIRCRAFT_STATE_LANDING:
                self.landing_control(pool, i, dt)
                continue
            # CRUISE / ENGAGE / RTB：定高定航 + 航路点
            self.cruise_control(pool, i, dt)

    # ------------------------------------------------------------------
    # 定高定航（三通道）
    # ------------------------------------------------------------------

    def cruise_control(self, pool: Any, i: int, dt: float) -> None:
        """CRUISE 核心：高度保持 + 速度保持 + 航路点转向（docs §3.3）。"""
        vx, vy, vz = float(pool.vel[i, 0]), float(pool.vel[i, 1]), float(pool.vel[i, 2])
        v = math.sqrt(vx * vx + vy * vy + vz * vz)
        z = float(pool.pos[i, 2])

        # ① 高度保持：a_z_cmd = Kp·(h_cruise−z) − Kd·vz → γ̇ = a_z / V
        a_z = self.KP_H * (float(pool.cruise_alt[i]) - z) - self.KD_H * vz
        climb_max = float(self.aero.climb_rate_max[i])
        a_z = _clamp(a_z, -climb_max, climb_max)
        pool.gamma_cmd[i] = a_z / max(v, 1e-3)

        # ② 速度保持：油门 PI（编队追 wp 时允许超速收敛，docs §5.4）
        target_spd = float(pool.cruise_spd[i])
        wc = int(pool.wp_count[i])
        wi = int(pool.wp_idx[i])
        if 0 < wc and wi < wc:
            dw = math.hypot(
                float(pool.wp[i, wi, 0]) - float(pool.pos[i, 0]),
                float(pool.wp[i, wi, 1]) - float(pool.pos[i, 1]),
            )
            if dw > 500.0:
                target_spd += 60.0   # 远离航路点（含僚机槽位）→ 超速收敛
        err = target_spd - v
        self._v_int[i] += err * float(dt)
        self._v_int[i] = _clamp(self._v_int[i], -2000.0, 2000.0)  # 防积分饱和
        pool.throttle[i] = _clamp(
            self.KP_V * err + self.KI_V * float(self._v_int[i]), 0.0, 1.0
        )

        # ③ 航路点转向
        self._waypoint_turn(pool, i, v)

    def _waypoint_turn(self, pool: Any, i: int, v: float) -> None:
        """朝当前航路点转向（到达 → wp_idx+1；耗尽 → 松滚转平飞）。"""
        wc = int(pool.wp_count[i])
        if wc <= 0:
            pool.phi_cmd[i] = 0.0
            return
        wi = int(pool.wp_idx[i])
        if wi >= wc:
            pool.phi_cmd[i] = 0.0
            return
        dx = float(pool.wp[i, wi, 0]) - float(pool.pos[i, 0])
        dy = float(pool.wp[i, wi, 1]) - float(pool.pos[i, 1])
        if math.hypot(dx, dy) < self.WP_RADIUS:
            wi += 1
            pool.wp_idx[i] = wi
            if wi >= wc:
                pool.phi_cmd[i] = 0.0
                return
            dx = float(pool.wp[i, wi, 0]) - float(pool.pos[i, 0])
            dy = float(pool.wp[i, wi, 1]) - float(pool.pos[i, 1])
        psi_des = math.atan2(dy, dx)
        dpsi = _wrap_pi(psi_des - float(pool.heading[i]))
        pool.phi_cmd[i] = _clamp(self.KP_PSI * dpsi, -self.MAX_BANK, self.MAX_BANK)

    # ------------------------------------------------------------------
    # 机动原语（单位 AI 调用，docs §3.3）
    # ------------------------------------------------------------------

    def lead_pursuit(self, pool: Any, i: int, tpos: np.ndarray, tvel: np.ndarray) -> None:
        """占位 lead pursuit：指向拦截点（目标 pos + vel·t_go）。"""
        x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
        tx, ty = float(tpos[0]), float(tpos[1])
        rx, ry = tx - x, ty - y
        r = math.hypot(rx, ry)
        if r < 1e-3:
            pool.phi_cmd[i] = 0.0
            return
        # 接近速度（视线方向相对速度投影）
        vx, vy = float(pool.vel[i, 0]), float(pool.vel[i, 1])
        rel_vx = vx - float(tvel[0])
        rel_vy = vy - float(tvel[1])
        closing = (rx * rel_vx + ry * rel_vy) / r
        t_go = r / max(-closing, 20.0)     # 接近中 t_go>0；远离时下限保护
        ipx = tx + float(tvel[0]) * t_go
        ipy = ty + float(tvel[1]) * t_go
        psi_des = math.atan2(ipy - y, ipx - x)
        dpsi = _wrap_pi(psi_des - float(pool.heading[i]))
        pool.phi_cmd[i] = _clamp(self.KP_PSI * dpsi, -self.MAX_BANK, self.MAX_BANK)
        # 占位同时保持能量：油门中高
        pool.throttle[i] = max(pool.throttle[i], 0.5)

    def kinematic_defeat(self, pool: Any, i: int, dir_sign: float = 1.0) -> None:
        """规避 kinematic defeat（RWR 锁定）：急转弯 + 降高度入杂波。"""
        pool.phi_cmd[i] = self.DEFEAT_BANK * (1.0 if dir_sign >= 0 else -1.0)
        pool.gamma_cmd[i] = self.DEFEAT_DIVE_GAMMA
        pool.throttle[i] = 1.0

    def dive(self, pool: Any, i: int, gamma: float | None = None) -> None:
        """俯冲投弹：下俯 + 松滚转。"""
        pool.gamma_cmd[i] = gamma if gamma is not None else self.DIVE_GAMMA
        pool.phi_cmd[i] = 0.0

    def pullup(self, pool: Any, i: int, gamma: float | None = None) -> None:
        """投后拉起脱离。"""
        pool.gamma_cmd[i] = gamma if gamma is not None else self.PULLUP_GAMMA
        pool.phi_cmd[i] = 0.0

    # ------------------------------------------------------------------
    # 起降（闭环简化，docs §九；步骤⑨补全程序化进近）
    # ------------------------------------------------------------------

    def takeoff_control(self, pool: Any, i: int, dt: float) -> None:
        """起飞滑跑 → 爬升到巡航高度（对齐跑道航向由外部设定 wp）。

        滑跑阶段贴地（z<1 钳到地面，点质量模型无轮式支撑），
        离地速度达标后给爬升指令（takeoff 完成判定在 AirfieldOps）。
        """
        vx, vy, vz = float(pool.vel[i, 0]), float(pool.vel[i, 1]), float(pool.vel[i, 2])
        v = math.sqrt(vx * vx + vy * vy + vz * vz)
        # 滑跑贴地：升力不足时重力会拉穿地面（点质量无轮式支撑）
        if float(pool.pos[i, 2]) < 1.0:
            pool.pos[i, 2] = 1.0
            pool.vel[i, 2] = 0.0
        # 滑跑加速（满油门），到离地速度 → 爬升指令
        pool.throttle[i] = 1.0
        if v > float(self.aero.v_min[i]) * 1.1 and v > 1e-3:
            a_z = self.KP_H * (float(pool.cruise_alt[i]) - float(pool.pos[i, 2]))
            a_z = _clamp(a_z, -float(self.aero.climb_rate_max[i]),
                         float(self.aero.climb_rate_max[i]))
            pool.gamma_cmd[i] = a_z / v
        else:
            pool.gamma_cmd[i] = 0.0
        self._waypoint_turn(pool, i, v)

    def landing_control(self, pool: Any, i: int, dt: float) -> None:
        """程序化进近：保持进近高度（cruise_alt）+ 减速 + 对准跑道 wp。

        拉平/触地由 AirfieldOps.tick 接管（跑道附近下压）。
        """
        vx, vy, vz = float(pool.vel[i, 0]), float(pool.vel[i, 1]), float(pool.vel[i, 2])
        v = math.sqrt(vx * vx + vy * vy + vz * vz)
        # 贴地保护（触地瞬间的穿地防御，PARKED 转移由 AirfieldOps 接管）
        if float(pool.pos[i, 2]) < 1.0:
            pool.pos[i, 2] = 1.0
            pool.vel[i, 2] = 0.0
        # 高度保持（进近高度 = cruise_alt，由 request_landing 设为跑道上方 300m）
        a_z = self.KP_H * (float(pool.cruise_alt[i]) - float(pool.pos[i, 2])) - self.KD_H * vz
        a_z = _clamp(a_z, -float(self.aero.climb_rate_max[i]),
                     float(self.aero.climb_rate_max[i]))
        pool.gamma_cmd[i] = a_z / max(v, 1e-3)
        # 减速（目标进近速度 120 m/s，慢车上限）
        err = 120.0 - v
        pool.throttle[i] = _clamp(self.KP_V * err, 0.0, 0.4)
        self._waypoint_turn(pool, i, v)


# ---------------------------------------------------------------------------
# movement hook（docs §3.4）
# ---------------------------------------------------------------------------


def build_air_movement_hook(
    pool: Any,
    aero: AircraftAero,
    boxes: Any,
    emitters: Any,
    armor: Any,
    projectiles: Any,
    gravity: float = 9.81,
) -> Callable:
    """构建 air_movement 子系统 hook：飞控 → 气动积分 → 失速转坠毁 → 箱/辐射源同步。

    供 ``SimulationLoop.attach("air_movement", hook)`` 使用（docs §1.2）。
    FlightControl 实例在闭包内持久（油门积分项跨帧保留）。

    Args:
        pool: AircraftPool。
        aero: AircraftAero（气动/包线数组）。
        boxes: BoxPool（机身碰撞箱）。
        emitters: EmitterPool（机载雷达/尾焰）。
        armor: ArmorPool（透传 sync）。
        projectiles: ProjectilePool（透传 sync）。
        gravity: 重力加速度。
    """
    flight = FlightControl(pool.capacity, aero)
    stall_out = np.zeros(pool.capacity, dtype=np.bool_)

    def hook(_loop: Any, dt: float) -> None:
        nonlocal stall_out
        if pool.capacity > flight.capacity:
            flight.resize(pool.capacity)
        if stall_out.shape[0] < pool.capacity:
            stall_out = np.zeros(pool.capacity, dtype=np.bool_)
        stall_out[:] = False

        # ① 飞控写输入（Python 决策层）
        flight.tick(pool, float(dt))

        # ② 气动积分（Numba 并行内核）
        integrate_aircraft_motion(
            pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
            pool.throttle, pool.phi_cmd, pool.gamma_cmd,
            pool.state, pool.dmg_factor, pool.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            pool.fuel, aero.fuel_burn,
            1.225, 8500.0, gravity, float(dt), stall_out,
        )

        # ③ 失速 → 坠毁（Python 层副作用：状态迁移）
        idxs = np.flatnonzero(stall_out & pool.active)
        if idxs.size:
            pool.state[idxs] = _AIRCRAFT_STATE_CRASHING

        # ④ 机身碰撞箱 + 辐射源同步（父实体位置跟随）
        boxes.sync_all_from_parent(armor, aircraft=pool)
        emitters.sync_from_parent(armor, projectiles, float(dt), aircraft=pool)

    return hook
