"""导弹控制律注册表与内置控制回调（Phase 6b 制导管线，docs/phase6-guidance-pipeline.md §三）。

架构（2026-08-14 用户澄清）：
    - `guid_law` 是指向「函数注册表」（MissileControlRegistry）的索引，
      不是「有无制导」的二元开关。**末敏弹也有 guid_law 索引**（sfm_ballistic），
      但 BALLISTIC 阶段该函数不制导（输出 0 → 直线 + 重力，与 guid_law=-1 等价）。
    - `guid_stage` / `guid_override` 是**传给该函数的状态机输入**：
      控制回调签名含 `stage`（GuidanceStage）与 `override`（ManualOverride），
      内部按状态机分派行为（BALLISTIC 不制导 / TRACK/MEMORY/TERMINAL 制导+推进；
      override=HOLD 滑翔 / REIGNITE 二次点火）。
    - 制导律（PN/PP）降级为**公共转向求解器**：纯数学、不注册、不进回调参数，
      被控制回调直接 import 调用（Numba 编译期内联/直调）——避免「控制律 × 制导律」
      缓存版本笛卡尔积（陷阱 §八.3）。

    - 执行器饱和（|a| ≤ max_accel）/ 导引头 FOV 门限 / 重力由积分内核统一施加
      （movement/integrate.py::integrate_guided_projectile_movement），回调只算总加速度。

Numba 约束（AGENTS.md + memory 2026-08-11）：
    - Numba 对 IntEnum 的索引/int() 均报错 → 回调与内核内一律用模块级 int 常量比较。
    - 回调是 @njit 纯函数：只读输入、只写 out，不得读写全局/外部状态（prange 并行安全）。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# 枚举（追加值不重排既有编号，旧存档/旧测试兼容，陷阱 §八.11）
# ---------------------------------------------------------------------------


class GuidanceStage(IntEnum):
    """导弹阶段状态机（传给控制回调的 stage 输入）。"""
    TRACK = 0       # 跟踪（导引头锁定目标坐标，guid_target_pos 有效）
    MEMORY = 1      # 记忆模式（辐射源关机 → 惯性制导飞向最后已知位置 guid_memory_pos）
    TERMINAL = 2    # 末端引导（双模导引头锁定目标本体坐标）
    BALLISTIC = 3   # 弹道飞行（末敏弹/曲射：该阶段**控制回调不制导**，直线+重力）
    SCAN = 4        # 开伞扫描（末敏弹特有：内核悬停 + Python 层双模扫描）


class ManualOverride(IntEnum):
    """人在回路推进指令（经 guid_comm 门控后写入 SoA；非数据链协议，仅推进位）。

    改目标 / 自毁不占位——由 Python 层直接改 guid_target_pos / destroy。
    """
    AUTO = 0          # 自主飞行（回调按 age + 距离查时序）
    HOLD = 1          # 停止推进（滑翔）
    REIGNITE = 2      # 二次点火（操作员下令，无视时序/距离门限）


# Numba 用 int 常量（IntEnum 在 njit 中会报错）
_STAGE_TRACK = int(GuidanceStage.TRACK)
_STAGE_MEMORY = int(GuidanceStage.MEMORY)
_STAGE_TERMINAL = int(GuidanceStage.TERMINAL)
_STAGE_BALLISTIC = int(GuidanceStage.BALLISTIC)
_STAGE_SCAN = int(GuidanceStage.SCAN)

_OVERRIDE_AUTO = int(ManualOverride.AUTO)
_OVERRIDE_HOLD = int(ManualOverride.HOLD)
_OVERRIDE_REIGNITE = int(ManualOverride.REIGNITE)


# ---------------------------------------------------------------------------
# 公共转向求解器（制导律）：纯数学，无副作用，被控制回调直接调用
# ---------------------------------------------------------------------------
# 统一签名（design §三，out-param 风格，逐弹丸标量化）：
#   guidance_fn(proj_pos, proj_vel, target_pos, target_vel, nav_gain, dt, out) -> None
# 只写 out（理想转向加速度），不做执行器饱和/FOV/重力（内核统一施加）。


@njit(cache=True)
def guidance_proportional_nav(
    proj_pos: np.ndarray,    # (3,) float32
    proj_vel: np.ndarray,    # (3,) float32
    target_pos: np.ndarray,  # (3,) float32
    target_vel: np.ndarray,  # (3,) float32（地面目标恒 0）
    nav_gain: float,
    dt: float,
    out: np.ndarray,         # (3,) float32 输出理想转向加速度 (m/s²)
) -> None:
    """比例导引（PN）：a = N·max(Vc,0)·λ̇（纯数学，无副作用）。

    Vc = -(r·v_rel)/|r| 接近速度；λ̇ = (r×v_rel)/|r|² 视线角速率向量。
    数值稳定性：r2<ε → 零加速度（防除零）；Vc≤0 → 停导（目标远离/平行）。
    """
    rx = target_pos[0] - proj_pos[0]
    ry = target_pos[1] - proj_pos[1]
    rz = target_pos[2] - proj_pos[2]
    r2 = rx * rx + ry * ry + rz * rz
    if r2 < 1e-6:
        out[0] = out[1] = out[2] = 0.0
        return
    vrx = target_vel[0] - proj_vel[0]
    vry = target_vel[1] - proj_vel[1]
    vrz = target_vel[2] - proj_vel[2]
    vc = -(rx * vrx + ry * vry + rz * vrz) / np.sqrt(r2)
    if vc <= 0.0:
        out[0] = out[1] = out[2] = 0.0
        return
    lx = (ry * vrz - rz * vry) / r2
    ly = (rz * vrx - rx * vrz) / r2
    lz = (rx * vry - ry * vrx) / r2
    n = nav_gain * vc
    out[0] = n * lx
    out[1] = n * ly
    out[2] = n * lz


@njit(cache=True)
def guidance_pure_pursuit(
    proj_pos: np.ndarray,    # (3,) float32
    proj_vel: np.ndarray,    # (3,) float32
    target_pos: np.ndarray,  # (3,) float32
    target_vel: np.ndarray,  # (3,) float32（地面目标恒 0）
    nav_gain: float,
    dt: float,
    out: np.ndarray,         # (3,) float32 输出理想转向加速度 (m/s²)
) -> None:
    """纯追踪（PP）：期望速度 = 当前速率 × 视线单位向量（保持速率转向）。"""
    rx = target_pos[0] - proj_pos[0]
    ry = target_pos[1] - proj_pos[1]
    rz = target_pos[2] - proj_pos[2]
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    if r < 1e-6:
        out[0] = out[1] = out[2] = 0.0
        return
    speed = np.sqrt(proj_vel[0] ** 2 + proj_vel[1] ** 2 + proj_vel[2] ** 2)
    out[0] = nav_gain * (speed * rx / r - proj_vel[0])
    out[1] = nav_gain * (speed * ry / r - proj_vel[1])
    out[2] = nav_gain * (speed * rz / r - proj_vel[2])


# ---------------------------------------------------------------------------
# 内置推进剖面（写死在控制函数内；per-ammo SoA 参数化留后续）
# ---------------------------------------------------------------------------

# 单次开机 + 滑翔（中远程典型）：30g 轴向 5s
_BOOST_DUR = 5.0
_BOOST_ACCEL = 294.0
# 双脉冲：首段 3s 30g + 末端 15g 二次点火（距目标 < 800m 或操作员下令）
_BOOST1_DUR = 3.0
_BOOST1_ACCEL = 294.0
_BOOST2_RANGE = 800.0
_BOOST2_ACCEL = 147.0

# ---------------------------------------------------------------------------
# 导弹控制回调：推进剖面 × 制导律合一（docs §三.5）
# ---------------------------------------------------------------------------
# 统一签名（out 输出**总加速度 = 转向 + 轴向推力**）：
#   control_fn(proj_pos, proj_vel, target_pos, target_vel, age,
#              stage, override, nav_gain, dt, out) -> None
#   - age:     飞行时间（推进剖面查时序：开机/滑翔/二次点火）
#   - stage:   阶段状态机（GuidanceStage；BALLISTIC → 本函数不制导）
#   - override: 人在回路推进指令（ManualOverride；已通过 guid_comm 门控）
# 回调不感知通信质量，只消费「操作员已下令且确实送达」的结果（门控在 Python 层）。


@njit(cache=True)
def control_boost_coast_pn(
    pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out,
) -> None:
    """单次开机 + 滑翔 + PN（中远程典型）。

    BALLISTIC（弹道阶段）→ 不制导、不推进（直线 + 重力由内核施加）。
    TRACK/MEMORY/TERMINAL → PN 转向 + 开机窗口内轴向推力（override=HOLD 滑翔）。
    """
    if stage == _STAGE_BALLISTIC or stage == _STAGE_SCAN:
        out[0] = out[1] = out[2] = 0.0
        return
    guidance_proportional_nav(pos, vel, tpos, tvel, nav_gain, dt, out)
    if age < _BOOST_DUR and override != _OVERRIDE_HOLD:
        sp = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        if sp > 1e-6:
            out[0] += _BOOST_ACCEL * vel[0] / sp
            out[1] += _BOOST_ACCEL * vel[1] / sp
            out[2] += _BOOST_ACCEL * vel[2] / sp


@njit(cache=True)
def control_dual_pulse_pn(
    pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out,
) -> None:
    """双脉冲（二次点火）+ PN。

    首段开机 3s 30g → 滑翔 → 距目标 < 800m 或操作员 REIGNITE → 末端二次点火 15g。
    """
    if stage == _STAGE_BALLISTIC or stage == _STAGE_SCAN:
        out[0] = out[1] = out[2] = 0.0
        return
    guidance_proportional_nav(pos, vel, tpos, tvel, nav_gain, dt, out)
    sp = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
    if sp < 1e-6:
        return
    ux = vel[0] / sp
    uy = vel[1] / sp
    uz = vel[2] / sp
    rx = tpos[0] - pos[0]
    ry = tpos[1] - pos[1]
    rz = tpos[2] - pos[2]
    dist = np.sqrt(rx * rx + ry * ry + rz * rz)
    if age < _BOOST1_DUR:
        thrust = _BOOST1_ACCEL
    elif override == _OVERRIDE_REIGNITE or dist < _BOOST2_RANGE:
        thrust = _BOOST2_ACCEL
    else:
        thrust = 0.0
    out[0] += thrust * ux
    out[1] += thrust * uy
    out[2] += thrust * uz


@njit(cache=True)
def control_boost_coast_pp(
    pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out,
) -> None:
    """单次开机 + 滑翔 + 纯追踪（低性能弹：无 PN 前置量，直接追视线）。"""
    if stage == _STAGE_BALLISTIC or stage == _STAGE_SCAN:
        out[0] = out[1] = out[2] = 0.0
        return
    guidance_pure_pursuit(pos, vel, tpos, tvel, nav_gain, dt, out)
    if age < _BOOST_DUR and override != _OVERRIDE_HOLD:
        sp = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        if sp > 1e-6:
            out[0] += _BOOST_ACCEL * vel[0] / sp
            out[1] += _BOOST_ACCEL * vel[1] / sp
            out[2] += _BOOST_ACCEL * vel[2] / sp


@njit(cache=True)
def control_sfm_ballistic(
    pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out,
) -> None:
    """末敏弹控制回调（有 guid_law 索引，但**弹道阶段不制导**）。

    BALLISTIC → 恒 0 加速度：直线 + 重力由内核统一施加（与 guid_law=-1 直线积分等价）。
    SCAN → 内核悬停（本回调不被调用）；扫描/引爆由 Python 层（combat/sfm.py）驱动。
    TRACK/MEMORY/TERMINAL → 末敏弹理论不进入，恒 0（防御）。
    """
    out[0] = out[1] = out[2] = 0.0


@njit(cache=True)
def control_glide_bomb(
    pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out,
) -> None:
    """滑翔炸弹（Phase 7，docs/phase7-air-pipeline.md §8.1）。

    无推力（只读重力转向）：BALLISTIC → 不制导（自由落体）；TRACK/MEMORY/TERMINAL
    → 比例导引转向（无轴向推力项——与 control_boost_coast_pn 的唯一区别）。
    投放后先自由落体（BALLISTIC），激光指示/末制导进入后转 PN 滑翔修正。
    """
    if stage == _STAGE_BALLISTIC or stage == _STAGE_SCAN:
        out[0] = out[1] = out[2] = 0.0
        return
    guidance_proportional_nav(pos, vel, tpos, tvel, nav_gain, dt, out)
    # 无推力项（重力由内核统一施加）


# ---------------------------------------------------------------------------
# 控制律注册表（law_id → @njit 控制回调；插入序即索引序，读档确定性）
# ---------------------------------------------------------------------------


class MissileControlRegistry:
    """导弹控制律注册表：law_id → @njit 控制回调（推进 + 制导合一）。

    - `guid_law` SoA 数组存的就是这里的**索引**（插入序）。
    - 读档后必须保证构建顺序一致（with_builtin 固定注册顺序），否则索引错位
      （docs §2.5）；存档只存数据数组，不存函数对象。
    - 制导律求解器（PN/PP）**不注册**——它们是公共求解器，被控制回调直接调用。
    """

    def __init__(self) -> None:
        self._laws: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 注册 / 查询
    # ------------------------------------------------------------------

    def register(self, law_id: str, fn: Any) -> None:
        """注册控制回调（覆盖同 id）。"""
        self._laws[law_id] = fn

    def get(self, law_id: str) -> Any:
        """按 id 取控制回调（KeyError 语义同 WeaponRegistry）。"""
        return self._laws[law_id]

    def index_of(self, law_id: str) -> int:
        """law_id → 注册表索引（未注册返回 -1）。"""
        try:
            return list(self._laws.keys()).index(law_id)
        except ValueError:
            return -1

    def law_at(self, index: int) -> Any | None:
        """注册表索引 → 控制回调（越界返回 None）。"""
        if index < 0 or index >= len(self._laws):
            return None
        return list(self._laws.values())[index]

    @property
    def size(self) -> int:
        return len(self._laws)

    def ids(self) -> list[str]:
        return list(self._laws.keys())

    @classmethod
    def with_builtin(cls) -> "MissileControlRegistry":
        """内置注册表（**固定注册顺序**——读档确定性依赖，勿重排）。"""
        reg = cls()
        reg.register("boost_coast_pn", control_boost_coast_pn)
        reg.register("dual_pulse_pn", control_dual_pulse_pn)
        reg.register("boost_coast_pp", control_boost_coast_pp)
        reg.register("sfm_ballistic", control_sfm_ballistic)
        reg.register("glide_bomb", control_glide_bomb)  # Phase 7（末尾追加，读档索引稳定）
        return reg
