"""Phase 6b 制导管线测试（docs/phase6-guidance-pipeline.md）。

覆盖：
    - MissileControlRegistry：law_id ↔ 索引往返 + 内置固定顺序（读档确定性）
    - PN / PP 公共求解器：静止/运动目标收敛命中、停导分支
    - 控制回调状态机（guid_stage / guid_override 是传给函数的状态机输入）：
        BALLISTIC 不制导（输出 0） / boost 窗口轴向推力 / 滑翔 / 二次点火
    - 制导积分内核：BALLISTIC 直线+重力 / SCAN 悬停 / TRACK 制导 / FOV 门限
    - EmitterPool 生命周期：activate/deactivate/reactivate / 级联失效 / 红外充能-残热 / pulse
    - 序列化 round-trip：制导字段 + emitter + rcs/ecm + 旧存档兼容
    - 端到端：fire 写制导字段（含末敏弹有 guid_law 索引）→ 制导导弹命中

运行：uv run python -m pytest tests/test_guidance_pipeline.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from Airia.lulilo.combat.damage import ArmorType
from Airia.lulilo.combat.engagement import FiringControl
from Airia.lulilo.combat.guidance import (
    GuidanceStage,
    ManualOverride,
    MissileControlRegistry,
    control_boost_coast_pn,
    control_dual_pulse_pn,
    control_sfm_ballistic,
    guidance_proportional_nav,
    guidance_pure_pursuit,
)
from Airia.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    GuidanceType,
    SeekerType,
    WeaponRegistry,
    WeaponSystem,
)
from Airia.lulilo.entities.pools import (
    EmitterBand,
    EmitterPool,
    PartType,
    ProjectilePool,
    ProjectileType,
)
from Airia.lulilo.movement.integrate import (
    integrate_guided_projectile_movement,
    integrate_projectile_movement,
)
from Airia.lulilo.simulation.loop import EventBus
from Airia.lulilo.simulation.serialization import (
    deserialize_battlefield,
    serialize_battlefield,
)
from Airia.lulilo.simulation.state import SimulationState
from tests.test_phase3_integration import (
    DEFAULT_PARTS,
    make_battlefield,
    spawn_tank,
)

# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


class _TestRNG:
    """可控 RNG：散布恒 0（落点=瞄准点）+ uniform 固定值 + 有限 normal。

    u=0.0 → 恒检测成功；u=0.99 → 恒检测失败（ECM 干扰测试用）。
    """

    def __init__(self, u: float = 0.0, seed: int = 0) -> None:
        self.u = float(u)
        self._np = np.random.Generator(np.random.PCG64(seed))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self.u

    def dispersion_offset(self, sigma: float) -> tuple[float, float]:
        return (0.0, 0.0)

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._np.normal(loc, scale))

    def chance_batch(self, prob, size: int) -> np.ndarray:
        """试验桩：禁止归一化暴击（恒 False）→ 回到纯血量确定性判定。"""
        return np.zeros(size, dtype=bool)


# 自定义控制回调：纯 PN、无推力（低速稳定命中测试，design §十 test_custom_control_law_injected）
_GUIDANCE_STAGE_BALLISTIC = int(GuidanceStage.BALLISTIC)
_GUIDANCE_STAGE_SCAN = int(GuidanceStage.SCAN)
_GUIDANCE_STAGE_TRACK = int(GuidanceStage.TRACK)


@njit(cache=True)
def control_pn_only(pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out) -> None:
    if stage == _GUIDANCE_STAGE_BALLISTIC or stage == _GUIDANCE_STAGE_SCAN:
        out[0] = out[1] = out[2] = 0.0
        return
    guidance_proportional_nav(pos, vel, tpos, tvel, nav_gain, dt, out)


def make_guided_registry() -> WeaponRegistry:
    """内置注册表 + 制导弹药（ATGM / 末敏弹）。"""
    reg = WeaponRegistry.with_builtin()
    reg.register_ammo(
        AmmoConfig(
            ammo_id="atgm_pn",
            name="ATGM PN（测试）",
            ammo_type=AmmoType.ATGM,
            pen_mm_rha=700.0,
            muzzle_velocity=250.0,
            projectile_type=ProjectileType.MISSILE,
            guidance=GuidanceType.FIRE_AND_FORGET,
            control_law_id="boost_coast_pn",
            nav_gain=3.0,
            seeker=SeekerType.ACTIVE_RADAR,
            emitter_power_db=60.0,
            search_fov_deg=30.0,
            track_fov_deg=5.0,
            filter_quality=0.9,
            is_heat=True,
            drag_coefficient=0.0,
        )
    )
    reg.register_ammo(
        AmmoConfig(
            ammo_id="sfm_155",
            name="155mm 末敏弹（测试）",
            ammo_type=AmmoType.SENSOR_FUZED,
            pen_mm_rha=700.0,
            muzzle_velocity=300.0,
            projectile_type=ProjectileType.MISSILE,
            guidance=GuidanceType.FIRE_AND_FORGET,
            control_law_id="sfm_ballistic",
            seeker=SeekerType.SENSOR_FUZED,
            scan_altitude_m=100.0,
            scan_duration_s=5.0,
            efp_pen_mm=200.0,
            mmw_emitter_power_db=40.0,
            is_heat=True,
            drag_coefficient=0.0,
        )
    )
    return reg


def make_guided_fc(u: float = 0.0, reg: WeaponRegistry | None = None) -> FiringControl:
    """制导 FiringControl（_TestRNG 可控检测 + 制导弹药注册表）。"""
    armor, boxes, projectiles, ammo, fuel, _ = make_battlefield(rng=_TestRNG(u=u))
    registry = reg or make_guided_registry()
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        weapon_system=WeaponSystem(registry),
        rng=_TestRNG(u=u),
        events=EventBus(),
    )
    return fc


def spawn_at(fc: FiringControl, x: float, y: float, *, yaw: float = 0.0, **kw) -> int:
    """生成带全套碰撞箱的坦克（复用 Phase 3 辅助）。"""
    return spawn_tank(
        fc.armor, fc.boxes, fc.ammo, fc.fuel, fc, x, y, yaw=yaw, **kw
    )


def make_fc_pool_only() -> tuple[FiringControl, ProjectilePool]:
    """仅对象池的 FiringControl（手动 spawn 制导弹丸，绕开 fire 的散布/冷却）。"""
    armor, boxes, projectiles, ammo, fuel, _ = make_battlefield(rng=_TestRNG())
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        rng=_TestRNG(u=0.0),
        events=EventBus(),
    )
    return fc, projectiles


# ===========================================================================
# 注册表
# ===========================================================================


class TestMissileControlRegistry:
    def test_index_roundtrip(self) -> None:
        reg = MissileControlRegistry()
        reg.register("a", control_boost_coast_pn)
        reg.register("b", control_dual_pulse_pn)
        assert reg.index_of("a") == 0
        assert reg.index_of("b") == 1
        assert reg.index_of("missing") == -1
        assert reg.law_at(0) is control_boost_coast_pn
        assert reg.law_at(1) is control_dual_pulse_pn
        assert reg.law_at(2) is None
        assert reg.law_at(-1) is None

    def test_builtin_fixed_order(self) -> None:
        """读档确定性：with_builtin 固定注册顺序（docs §2.5）。"""
        reg = MissileControlRegistry.with_builtin()
        # Phase 7：glide_bomb 追加在末尾（读档索引稳定，前 4 索引不变）
        assert reg.ids() == [
            "boost_coast_pn", "dual_pulse_pn", "boost_coast_pp",
            "sfm_ballistic", "glide_bomb",
        ]
        assert reg.index_of("sfm_ballistic") == 3  # 末敏弹有函数索引
        assert reg.index_of("glide_bomb") == 4     # Phase 7 滑翔弹


# ===========================================================================
# PN / PP 公共求解器
# ===========================================================================


class TestGuidanceLaws:
    def test_pn_converges_stationary(self) -> None:
        """PN：静止目标——视线角速率衰减、最终命中（docs §3.2）。"""
        pos = np.array([0.0, 0.0, 0.0], np.float32)
        vel = np.array([100.0, 0.0, 0.0], np.float32)
        target = np.array([500.0, 0.0, 0.0], np.float32)
        zero = np.zeros(3, np.float32)
        acc = np.zeros(3, np.float32)
        dt = 0.05
        hit = False
        for _ in range(2000):
            guidance_proportional_nav(pos, vel, target, zero, 4.0, dt, acc)
            vel += acc * dt
            pos += vel * dt
            if np.linalg.norm(pos[:2] - target[:2]) < 5.0:
                hit = True
                break
        assert hit, f"PN 未命中静止目标，末位置 {pos[:2]}"

    def test_pp_hits_moving_target(self) -> None:
        """PP：匀速运动目标——追踪命中（docs §3.3）。"""
        pos = np.array([0.0, 0.0, 0.0], np.float32)
        vel = np.array([120.0, 0.0, 0.0], np.float32)
        target = np.array([500.0, 0.0, 0.0], np.float32)
        target_vel = np.array([5.0, 3.0, 0.0], np.float32)
        acc = np.zeros(3, np.float32)
        dt = 0.02
        hit = False
        for _ in range(3000):
            guidance_pure_pursuit(pos, vel, target, target_vel, 2.0, dt, acc)
            vel += acc * dt
            pos += vel * dt
            target += target_vel * dt
            if np.linalg.norm(pos[:2] - target[:2]) < 5.0:
                hit = True
                break
        assert hit, f"PP 未命中运动目标，末位置 {pos[:2]}"

    def test_pn_closing_speed_zero_stops(self) -> None:
        """PN：目标远离/平行（Vc≤0）→ 零加速度（停导，docs §3.2）。"""
        pos = np.array([0.0, 0.0, 0.0], np.float32)
        vel = np.array([-100.0, 0.0, 0.0], np.float32)  # 远离目标
        target = np.array([500.0, 0.0, 0.0], np.float32)
        zero = np.zeros(3, np.float32)
        acc = np.ones(3, np.float32) * 999.0
        guidance_proportional_nav(pos, vel, target, zero, 4.0, 0.01, acc)
        assert acc[0] == acc[1] == acc[2] == 0.0


# ===========================================================================
# 控制回调状态机（guid_stage / guid_override 是传给函数的状态机输入）
# ===========================================================================


def _probe(
    fn, *, age: float, stage: int, override: int, tpos, dist: float
) -> np.ndarray:
    """调控制回调，返回输出加速度。"""
    pos = np.array([0.0, 0.0, 0.0], np.float32)
    vel = np.array([100.0, 0.0, 0.0], np.float32)
    tvel = np.zeros(3, np.float32)
    out = np.zeros(3, np.float32)
    fn(pos, vel, tpos, tvel, age, stage, override, 3.0, 0.01, out)
    return out


class TestControlCallbacks:
    def test_ballistic_no_guidance(self) -> None:
        """BALLISTIC（弹道阶段）：函数不制导 → 输出 0（直线+重力由内核施加）。"""
        tpos = np.array([500.0, 0.0, 0.0], np.float32)
        for fn in (control_boost_coast_pn, control_dual_pulse_pn, control_sfm_ballistic):
            out = _probe(fn, age=0.5, stage=int(GuidanceStage.BALLISTIC),
                         override=0, tpos=tpos, dist=0.0)
            assert out[0] == out[1] == out[2] == 0.0

    def test_boost_coast_boost_window(self) -> None:
        """开机窗口（age<5s）：轴向推力 + PN 转向（总加速度非零且沿速度方向分量>0）。"""
        tpos = np.array([500.0, 0.0, 0.0], np.float32)
        out = _probe(control_boost_coast_pn, age=1.0, stage=int(GuidanceStage.TRACK),
                     override=0, tpos=tpos, dist=0.0)
        assert out[0] > 0.0  # 轴向推力 + 转向，向前加速

    def test_boost_coast_coast_no_thrust(self) -> None:
        """滑翔段（age>5s）：无轴向推力（仅转向，无前向加速）。"""
        tpos = np.array([500.0, 0.0, 0.0], np.float32)
        out = _probe(control_boost_coast_pn, age=8.0, stage=int(GuidanceStage.TRACK),
                     override=0, tpos=tpos, dist=0.0)
        # 目标在正前方 → PN 转向 ≈ 0，且无推力 → 总加速度 ≈ 0
        assert abs(out[0]) < 1.0 and abs(out[1]) < 1.0

    def test_hold_override_slides(self) -> None:
        """override=HOLD（滑翔）：开机窗口内也无推力。"""
        tpos = np.array([500.0, 0.0, 0.0], np.float32)
        out = _probe(control_boost_coast_pn, age=1.0, stage=int(GuidanceStage.TRACK),
                     override=int(ManualOverride.HOLD), tpos=tpos, dist=0.0)
        assert abs(out[0]) < 1.0  # 无前向推力

    def test_dual_pulse_reignite(self) -> None:
        """override=REIGNITE：二次点火（无视时序/距离门限）。"""
        tpos = np.array([5000.0, 0.0, 0.0], np.float32)  # 远距，正常不会触发
        out = _probe(control_dual_pulse_pn, age=6.0, stage=int(GuidanceStage.TRACK),
                     override=int(ManualOverride.REIGNITE), tpos=tpos, dist=0.0)
        assert out[0] > 100.0  # 二次点火推力

    def test_dual_pulse_range_trigger(self) -> None:
        """距目标 < 800m：自动二次点火。"""
        tpos = np.array([300.0, 0.0, 0.0], np.float32)  # 300m < 800m
        out = _probe(control_dual_pulse_pn, age=6.0, stage=int(GuidanceStage.TRACK),
                     override=0, tpos=tpos, dist=0.0)
        assert out[0] > 100.0  # 末端二次点火

    def test_sfm_always_zero(self) -> None:
        """末敏弹控制回调：任何阶段都输出 0（弹道段不制导；SCAN 由内核悬停）。"""
        tpos = np.array([500.0, 0.0, 0.0], np.float32)
        for stage in (
            GuidanceStage.BALLISTIC,
            GuidanceStage.SCAN,
            GuidanceStage.TRACK,
        ):
            out = _probe(control_sfm_ballistic, age=1.0, stage=int(stage),
                         override=0, tpos=tpos, dist=0.0)
            assert out[0] == out[1] == out[2] == 0.0


# ===========================================================================
# 制导积分内核
# ===========================================================================


def _make_guided_buffers(n: int = 1):
    """构造制导积分内核的 SoA 缓冲（n 弹丸）。"""
    pos = np.zeros((n, 3), np.float32)
    vel = np.zeros((n, 3), np.float32)
    active = np.ones(n, np.bool_)
    age = np.zeros(n, np.float32)
    gtp = np.zeros((n, 3), np.float32)
    gtv = np.zeros((n, 3), np.float32)     # Phase 7：目标速度
    gtk = np.zeros(n, np.int8)             # Phase 7：目标种类（0=地面）
    gmp = np.zeros((n, 3), np.float32)
    stage = np.zeros(n, np.int32)
    gain = np.full(n, 3.0, np.float32)
    ovr = np.zeros(n, np.int32)
    sfov = np.ones(n, np.float32)
    tfov = np.ones(n, np.float32)
    return pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov


class TestGuidedKernel:
    def test_ballistic_like_straight(self) -> None:
        """BALLISTIC：直线 + 重力，与 guid_law=-1 直线积分一致（回归）。"""
        pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov = _make_guided_buffers()
        pos[0] = (0.0, 0.0, 100.0)
        vel[0] = (100.0, 0.0, 0.0)
        stage[0] = int(GuidanceStage.BALLISTIC)
        integrate_guided_projectile_movement(
            pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
            392.4, 0.1, 9.81, control_sfm_ballistic,
        )
        # 直线 + 重力（垂直速度被重力拉低）
        assert pos[0, 0] == pytest.approx(10.0)
        assert vel[0, 2] == pytest.approx(-0.981)

    def test_scan_hovers(self) -> None:
        """SCAN：悬停（位置/速度不变）。"""
        pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov = _make_guided_buffers()
        pos[0] = (100.0, 50.0, 80.0)
        vel[0] = (20.0, -5.0, 3.0)
        stage[0] = int(GuidanceStage.SCAN)
        integrate_guided_projectile_movement(
            pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
            392.4, 0.1, 9.81, control_sfm_ballistic,
        )
        assert pos[0, 0] == pytest.approx(100.0)
        assert vel[0, 0] == pytest.approx(20.0)

    def test_track_guides_to_target(self) -> None:
        """TRACK：控制回调制导——弹丸偏转向目标。"""
        pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov = _make_guided_buffers()
        pos[0] = (0.0, 0.0, 0.0)
        vel[0] = (100.0, 0.0, 0.0)
        gtp[0] = (500.0, 0.0, 0.0)  # 目标正前方
        stage[0] = int(GuidanceStage.TRACK)
        integrate_guided_projectile_movement(
            pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
            392.4, 0.1, 9.81, control_pn_only,
        )
        # 位置推进（x 增），y 不变（目标在正前方，PN 零横向）
        assert pos[0, 0] == pytest.approx(10.0)
        assert abs(pos[0, 1]) < 1e-3

        # 目标在斜向（3D）→ 弹丸应被拉向目标方向（PN 转向，3D 经 z 分量实现）
        pos[0] = (0.0, 0.0, 0.0)
        vel[0] = (100.0, 0.0, 0.0)
        gtp[0] = (500.0, 300.0, 0.0)
        d0 = np.linalg.norm(pos[0] - gtp[0])
        for _ in range(50):
            integrate_guided_projectile_movement(
                pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
                392.4, 0.1, 9.81, control_pn_only,
            )
        d1 = np.linalg.norm(pos[0] - gtp[0])
        assert d1 < d0 * 0.5  # 显著接近目标（转向生效）

    def test_fov_gate_straight(self) -> None:
        """FOV 门限：目标超出接收视场 → 停导直射（docs §4.2）。"""
        pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov = _make_guided_buffers()
        pos[0] = (0.0, 0.0, 0.0)
        vel[0] = (100.0, 0.0, 0.0)
        gtp[0] = (0.0, 1000.0, 0.0)  # 侧面远目标（与速度方向夹角 90°）
        stage[0] = int(GuidanceStage.TRACK)
        tfov[0] = float(np.cos(np.radians(10.0)))  # 窄跟踪视场 10°
        integrate_guided_projectile_movement(
            pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
            392.4, 0.1, 9.81, control_pn_only,
        )
        # 停导直射：只前进 x，不转向 y
        assert pos[0, 0] == pytest.approx(10.0)
        assert abs(pos[0, 1]) < 1e-3

    def test_max_accel_saturation(self) -> None:
        """执行器饱和：总加速度被 max_accel 截断。"""
        pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov = _make_guided_buffers()
        pos[0] = (0.0, 0.0, 0.0)
        vel[0] = (100.0, 0.0, 0.0)
        gtp[0] = (500.0, 0.0, 0.0)
        stage[0] = int(GuidanceStage.TRACK)
        low_accel = 10.0  # 远小于 boost 推力 294 → 必须被截断
        integrate_guided_projectile_movement(
            pos, vel, active, age, gtp, gtv, gtk, gmp, stage, gain, ovr, sfov, tfov,
            low_accel, 0.01, 9.81, control_boost_coast_pn,
        )
        # 速度增量 = 加速度*dt ≤ low_accel*dt
        assert vel[0, 0] - 100.0 <= 10.0 * 0.01 + 1e-6


# ===========================================================================
# EmitterPool 生命周期
# ===========================================================================


class TestEmitterPool:
    def test_activate_deactivate_reactivate(self) -> None:
        em = EmitterPool(capacity=16)
        idx = em.activate(1, 3, 60.0, EmitterBand.RF, 1)  # 弹丸 3 的主动导引头
        assert em.active[idx]
        assert em.parent_kind[idx] == 1
        assert em.band[idx] == int(EmitterBand.RF)
        em.deactivate(idx)
        assert not em.active[idx]
        assert idx not in em._free_stack  # 战术关机保留索引
        em.reactivate(idx)
        assert em.active[idx]
        # 重新激活不会复用（deactivate 未归还）
        idx2 = em.activate(1, 5, 60.0, EmitterBand.RF, 1)
        assert idx2 != idx

    def test_cascade_failure_projectile(self) -> None:
        """弹丸销毁 → 主动导引头 emitter 级联失效 + 索引归还。"""
        fc, projectiles = make_fc_pool_only()
        em = fc.emitters
        proj = projectiles.spawn(0, 0, 0, 1, 0, 0)
        e = em.activate_from_projectile(proj, 60.0, EmitterBand.RF)
        assert em.active[e]
        # _destroy_projectile 内部先显式关闭 emitter，再 destroy（双保险之一）
        fc._destroy_projectile(proj)
        assert not em.active[e]
        assert e in em._free_stack
        assert not projectiles.active[proj]

    def test_cascade_failure_armor_non_ir(self) -> None:
        """装甲销毁 → 非红外主动 emitter 立即失效；红外保留（残热）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        e_rf = fc.emitters.activate(0, a, 60.0, EmitterBand.RF, 0)  # 平台雷达
        e_ir = fc.emitters.activate_ir(a, 0.7, decay_per_s=0.2)
        # 充能
        fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        assert fc.emitters.cur_power[e_ir] == pytest.approx(0.7)
        # 摧毁装甲
        fc._destroy_armor(a, cause="test")
        fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        assert not fc.emitters.active[e_rf]
        assert e_rf in fc.emitters._free_stack
        # 红外：残热衰减后仍 active（cur_power 下降）
        assert fc.emitters.active[e_ir]
        assert fc.emitters.cur_power[e_ir] < 0.7
        # 持续衰减到阈值以下（0.7·exp(-0.2·dt·n) < 0.05 → n > 132）→ 失活归还
        for _ in range(250):
            fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        assert not fc.emitters.active[e_ir]
        assert e_ir in fc.emitters._free_stack

    def test_ir_charge_decay_model(self) -> None:
        """红外充能-残热模型：父存活充能到额定；死亡指数衰减（陷阱 §八.16）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        e = fc.emitters.activate_ir(a, 0.5, decay_per_s=0.5)
        # 从 0 充能到额定
        for _ in range(20):
            fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        assert fc.emitters.cur_power[e] == pytest.approx(0.5)
        # 父死亡 → 指数衰减
        fc._destroy_armor(a, cause="test")
        fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        after = fc.emitters.cur_power[e]
        expected = 0.5 * np.exp(-0.5 * 0.1)
        assert after == pytest.approx(expected, rel=1e-3)

    def test_pulse_peak(self) -> None:
        """殉爆充能：cur_power 瞬时拉高（接入点 = cookoff 事件）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        e = fc.emitters.activate_ir(a, 0.3, decay_per_s=0.5)
        fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        fc.emitters.pulse(e, 5.0)
        assert fc.emitters.cur_power[e] == pytest.approx(5.0)

    def test_designate_spot(self) -> None:
        """激光漫反射光斑：kind=4 无父，法线记录（docs §7.5）。"""
        em = EmitterPool(capacity=16)
        s = em.designate((100.0, 50.0, 8.0), power=10.0, normal=(0.0, 0.0, 1.0), illum_idx=2)
        assert em.active[s]
        assert em.kind[s] == 4
        assert em.parent_kind[s] == 2  # 无父，不走 sync
        assert em.normal[s, 2] == pytest.approx(1.0)


# ===========================================================================
# 序列化
# ===========================================================================


class TestGuidanceSerialization:
    def test_guidance_fields_roundtrip(self) -> None:
        """制导字段 round-trip 无丢失（docs §2.5）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        proj = projectiles.spawn(0, 0, 0, 1, 0, 0)
        projectiles.guid_law[proj] = 1
        projectiles.guid_target_pos[proj] = (100.0, 50.0, 10.0)
        projectiles.guid_stage[proj] = int(GuidanceStage.TERMINAL)
        projectiles.guid_nav_gain[proj] = 4.5
        projectiles.guid_comm[proj] = 0.7
        projectiles.guid_override[proj] = int(ManualOverride.REIGNITE)
        projectiles.guid_search_fov_cos[proj] = 0.8
        projectiles.guid_track_fov_cos[proj] = 0.9

        state = SimulationState(sim_time=1.0, tick_count=5)
        state.armor_pool, state.box_pool, state.projectile_pool = armor, boxes, projectiles
        blob = serialize_battlefield(state, armor, boxes, projectiles)
        save = deserialize_battlefield(blob)
        p2 = save.projectiles
        assert p2.guid_law[proj] == 1
        assert p2.guid_target_pos[proj, 0] == pytest.approx(100.0)
        assert p2.guid_stage[proj] == int(GuidanceStage.TERMINAL)
        assert p2.guid_nav_gain[proj] == pytest.approx(4.5)
        assert p2.guid_comm[proj] == pytest.approx(0.7)
        assert p2.guid_override[proj] == int(ManualOverride.REIGNITE)
        assert p2.guid_track_fov_cos[proj] == pytest.approx(0.9)

    def test_emitters_roundtrip(self) -> None:
        """EmitterPool round-trip（含 band/kind/cur_power/normal）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        a = spawn_tank(armor, boxes, ammo, fuel, fc, 10.0, 20.0)
        e_ir = fc.emitters.activate_ir(a, 0.6, decay_per_s=0.3)
        fc.emitters.sync_from_parent(armor, projectiles, 0.1)
        e_rf = fc.emitters.activate(0, a, 60.0, EmitterBand.RF, 0)
        s = fc.emitters.designate((100.0, 100.0, 5.0), 10.0)

        state = SimulationState(sim_time=1.0, tick_count=5)
        state.armor_pool, state.box_pool, state.projectile_pool = armor, boxes, projectiles
        state.emitter_pool = fc.emitters
        blob = serialize_battlefield(state, armor, boxes, projectiles, emitters=fc.emitters)
        save = deserialize_battlefield(blob)
        em2 = save.emitters
        assert em2 is not None
        assert em2.active[e_ir]
        assert em2.band[e_ir] == int(EmitterBand.IR)
        assert em2.cur_power[e_ir] == pytest.approx(0.6)
        assert em2.active[e_rf]
        assert em2.active[s]
        assert em2.kind[s] == 4
        assert em2.normal[s, 2] == pytest.approx(1.0)

    def test_rcs_ecm_roundtrip(self) -> None:
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        a = spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        armor.rcs[a] = 120.0
        armor.ecm[a] = 0.8
        state = SimulationState()
        state.armor_pool = armor
        blob = serialize_battlefield(state, armor=armor)
        save = deserialize_battlefield(blob)
        assert save.armor.rcs[a] == pytest.approx(120.0)
        assert save.armor.ecm[a] == pytest.approx(0.8)

    def test_old_save_compat(self) -> None:
        """旧存档（无制导/emitter/rcs 字段）读入不崩溃（兼容性）。"""
        armor, boxes, projectiles, ammo, fuel, fc = make_battlefield(rng=_TestRNG())
        state = SimulationState(sim_time=2.0, tick_count=3)
        # 手工构造"旧版" payload（只有旧字段）
        from Airia.lulilo.simulation.serialization import _pack_pool_arrays

        payload = {
            "version": 1,
            "sim_time": 2.0,
            "tick_count": 3,
            "armor": {
                "capacity": armor.capacity,
                **_pack_pool_arrays(
                    {
                        "pos": armor.pos, "half": armor.half, "yaw": armor.yaw,
                        "hull_idx": armor.hull_idx, "active": armor.active,
                        "wp": armor.wp, "wp_count": armor.wp_count, "wp_idx": armor.wp_idx,
                    }
                ),
            },
            "projectiles": {
                "capacity": projectiles.capacity,
                **_pack_pool_arrays(
                    {
                        "pos": projectiles.pos, "vel": projectiles.vel,
                        "radius": projectiles.radius, "dmg": projectiles.dmg,
                        "kind": projectiles.kind, "active": projectiles.active,
                        "owner": projectiles.owner, "weapon": projectiles.weapon,
                        "ammo": projectiles.ammo, "age": projectiles.age,
                    }
                ),
            },
        }
        import msgpack

        save = deserialize_battlefield(msgpack.packb(payload, use_bin_type=True))
        assert save.armor is not None
        assert save.projectiles is not None
        assert save.emitters is None
        # 新字段用构造默认（guid_law=-1）
        assert save.projectiles.guid_law[0] == -1


# ===========================================================================
# 端到端
# ===========================================================================


class TestPipelineE2E:
    def test_fire_writes_guidance_fields(self) -> None:
        """fire() 写制导字段：guid_law 索引 + 初始瞄准坐标 + 视场 + 主动雷达 emitter。"""
        fc = make_guided_fc()
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 400.0, 0.0, yaw=np.pi)
        rec = fc.fire(m1, t90, ammo_id="atgm_pn")
        assert rec is not None
        p = fc.projectiles
        proj = rec.proj_idx
        assert p.guid_law[proj] >= 0  # 指向 MissileControlRegistry 函数索引
        assert p.guid_stage[proj] == int(GuidanceStage.TRACK)
        assert p.guid_target_pos[proj, 0] == pytest.approx(400.0)  # 初始瞄准坐标
        assert p.guid_nav_gain[proj] == pytest.approx(3.0)
        assert p.guid_track_fov_cos[proj] == pytest.approx(
            np.cos(np.radians(5.0)), rel=1e-4
        )
        # 主动雷达导引头 → 激活 RF emitter（父=弹丸）
        mask = (
            fc.emitters.active
            & (fc.emitters.parent_kind == 1)
            & (fc.emitters.parent_idx == proj)
        )
        assert np.any(mask)

    def test_fire_guided_missile_reaches_target(self) -> None:
        """端到端：制导导弹（boost_coast_pn + 主动雷达导引头）飞向并命中目标。"""
        fc = make_guided_fc(u=0.0)  # 恒检测成功
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 400.0, 0.0, yaw=np.pi)
        rec = fc.fire(m1, t90, ammo_id="atgm_pn")
        assert rec is not None
        proj = rec.proj_idx
        # 推进直到命中或超时（地面 tick 1s，内部 0.01s 子步）
        hits = 0
        for _ in range(300):  # 最多 300s
            fc.tick(1.0)
            if not fc.projectiles.active[proj]:
                hits += 1
                break
            dist = np.linalg.norm(
                fc.projectiles.pos[proj, :2] - fc.armor.pos[t90, :2]
            )
            if dist < 5.0:
                hits += 1
                break
        assert hits > 0, "制导导弹未命中目标"

    def test_sfm_has_law_index_ballistic_no_guidance(self) -> None:
        """末敏弹：有 guid_law 索引（sfm_ballistic），但 BALLISTIC 阶段函数不制导。"""
        fc = make_guided_fc()
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 500.0, 500.0, yaw=np.pi)
        rec = fc.fire(m1, t90, ammo_id="sfm_155")
        assert rec is not None
        p = fc.projectiles
        proj = rec.proj_idx
        # 有索引：guid_law = sfm_ballistic（非 -1）
        assert p.guid_law[proj] == fc.control.index_of("sfm_ballistic")
        # 初始阶段 = BALLISTIC（弹道飞行，不制导）
        assert p.guid_stage[proj] == int(GuidanceStage.BALLISTIC)
        # 弹道段：直线 + 重力（水平速度不转向目标——对比制导导弹的转向）
        v0 = p.vel[proj].copy()
        fc.tick(0.5)
        assert p.guid_law[proj] == fc.control.index_of("sfm_ballistic")  # 仍走回调路径
        # 水平速度方向保持（未转向目标）
        assert np.sign(p.vel[proj, 0]) == np.sign(v0[0])

    def test_ecm_breaks_lock_to_memory(self) -> None:
        """ECM 干扰：射频导引头检测失败 → TRACK 切 MEMORY（记忆点=最后坐标）。"""
        fc = make_guided_fc(u=0.99)  # 恒检测失败（模拟强干扰）
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 200.0, 0.0, yaw=np.pi)
        armor = fc.armor
        armor.ecm[t90] = 0.99  # 目标强 ECM
        rec = fc.fire(m1, t90, ammo_id="atgm_pn")
        assert rec is not None
        proj = rec.proj_idx
        assert fc.projectiles.guid_stage[proj] == int(GuidanceStage.TRACK)
        last = fc.projectiles.guid_target_pos[proj].copy()
        # 一个子步后：检测失败 → 记忆模式（memory_pos = 最后坐标）
        fc._step_sub(0.01)
        assert fc.projectiles.guid_stage[proj] == int(GuidanceStage.MEMORY)
        assert fc.projectiles.guid_memory_pos[proj, 0] == pytest.approx(last[0])

    def test_comm_gated_override(self) -> None:
        """guid_comm 门控：通信质量 < comm_threshold → 人在回路指令不生效。"""
        fc = make_guided_fc()
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 400.0, 0.0, yaw=np.pi)
        rec = fc.fire(m1, t90, ammo_id="atgm_pn")
        assert rec is not None
        proj = rec.proj_idx
        ammo = fc._ammo_of(proj)
        threshold = ammo.comm_threshold
        # 门控失败：comm 低于阈值 → override 不写
        fc.projectiles.guid_comm[proj] = threshold - 0.1
        fc._set_manual_override(proj, ManualOverride.REIGNITE)
        assert fc.projectiles.guid_override[proj] == int(ManualOverride.AUTO)
        # 门控通过：comm 达标 → override 写入
        fc.projectiles.guid_comm[proj] = threshold + 0.1
        fc._set_manual_override(proj, ManualOverride.REIGNITE)
        assert fc.projectiles.guid_override[proj] == int(ManualOverride.REIGNITE)

    def test_target_lost_stops_guidance(self) -> None:
        """目标销毁后：检测失败 + 无记忆模式 → 停导直射（脱靶回收）。"""
        fc = make_guided_fc(u=0.99)
        reg = fc.weapon_system.registry
        # 用 memory_mode=False 的弹药
        reg.register_ammo(
            AmmoConfig(
                ammo_id="atgm_nomem",
                ammo_type=AmmoType.ATGM,
                pen_mm_rha=700.0,
                muzzle_velocity=250.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="boost_coast_pn",
                seeker=SeekerType.ACTIVE_RADAR,
                memory_mode=False,
                is_heat=True,
            )
        )
        m1 = spawn_at(fc, 0.0, 0.0)
        t90 = spawn_at(fc, 200.0, 0.0, yaw=np.pi)
        fc.armor.ecm[t90] = 0.99  # 强 ECM → 检测概率 < 0.99 → 恒检测失败
        rec = fc.fire(m1, t90, ammo_id="atgm_nomem")
        assert rec is not None
        proj = rec.proj_idx
        fc._step_sub(0.01)
        # 检测失败 + memory_mode=False → guid_law=-1（停导，走直线积分）
        assert fc.projectiles.guid_law[proj] == -1

