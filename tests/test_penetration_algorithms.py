"""穿深算法分派管线测试（docs/penetration-algorithms-design.md §十）。

覆盖：
    - 分派正确性（弹种 → 模型）
    - 速度衰减（impact_velocity）
    - DeMarre 速度标度（含 0m 锚定）
    - A-T 两段式（物理上界 / 临界速度 / 两段式终止 / 单调性）
    - 标定公式量级（WHA/RHA → Yp/Rt/Vc）
    - 简化 Lanz-Odermatt（速度饱和上界）
    - 分派式单发判定（软跳弹速度改用 v_imp）
    - 端到端（FiringControl 命中链：A-T 击穿 / 复合装甲未击穿）

运行：.venv\\Scripts\\python.exe -m pytest tests/test_penetration_algorithms.py -q
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from Airis.lulilo.combat.damage import (
    ArmorType,
    DamageResolver,
    PenetrationModel,
    TARGET_ARMOR_MODEL,
    compute_penetration,
    impact_velocity,
    penetration_at_erosion,
    penetration_demarre,
    penetration_lanz_odermatt,
    penetration_model_of,
    resolve_penetration_v2,
)
from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airis.lulilo.combat.weapons import AmmoType, WeaponRegistry, WeaponSystem
from Airis.lulilo.entities.pools import ArmorPool, BoxPool, PartType, ProjectilePool
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.simulation.loop import EventBus

# M829A3 标定参数（Tate 1986：WHA BHN≈420 → Yp=1.7·σyp≈3.0GPa；RHA BHN≈350 → Rt≈7.4GPa）
L_M829 = 780.0
RHO_P_WHA = 17500.0
Yp_WHA = 3.0e9
RHO_T_RHA = 7850.0
Rt_RHA = 7.4e9
V0_M829 = 1555.0
ALPHA_M829 = 3.5e-5


# ===========================================================================
# 分派正确性
# ===========================================================================


class TestDispatch:
    def test_apfsds_with_rod_params(self) -> None:
        assert penetration_model_of(int(AmmoType.APFSDS), True) == int(
            PenetrationModel.AT_EROSION
        )

    def test_apfsds_legacy_fallback(self) -> None:
        """旧 APFSDS（无长杆参数）→ 回退 DeMarre。"""
        assert penetration_model_of(int(AmmoType.APFSDS), False) == int(
            PenetrationModel.DEMARRE
        )

    def test_ap_apcr_demarre(self) -> None:
        assert penetration_model_of(int(AmmoType.AP), False) == int(
            PenetrationModel.DEMARRE
        )
        assert penetration_model_of(int(AmmoType.APCR), False) == int(
            PenetrationModel.DEMARRE
        )

    def test_chemical_fixed(self) -> None:
        for t in (AmmoType.HEAT, AmmoType.HE, AmmoType.ATGM, AmmoType.HESH,
                  AmmoType.HE_FRAG, AmmoType.PELE, AmmoType.SAPHEI):
            assert penetration_model_of(int(t), False) == int(PenetrationModel.FIXED)

    def test_builtin_apfsds_has_rod_params(self) -> None:
        """内置 M829A3/3BM42 已补长杆参数 → 生产中 APFSDS 真实走 A-T。"""
        reg = WeaponRegistry.with_builtin()
        m829 = reg.get_ammo("m829a3_apfsds")
        assert m829.penetrator_length_mm > 0.0
        assert m829.penetrator_strength_pa > 0.0
        assert penetration_model_of(int(m829.ammo_type), True) == int(
            PenetrationModel.AT_EROSION
        )


# ===========================================================================
# 速度衰减
# ===========================================================================


class TestImpactVelocity:
    def test_zero_alpha_no_decay(self) -> None:
        assert impact_velocity(1555.0, 0.0, 3000.0) == pytest.approx(1555.0)

    def test_exp_decay(self) -> None:
        v = impact_velocity(V0_M829, ALPHA_M829, 2000.0)
        assert v == pytest.approx(1555.0 * np.exp(-ALPHA_M829 * 2000.0))

    def test_monotone_decrease(self) -> None:
        vs = [impact_velocity(V0_M829, ALPHA_M829, r) for r in (0.0, 1000.0, 3000.0, 5000.0)]
        assert all(vs[i] > vs[i + 1] for i in range(len(vs) - 1))


# ===========================================================================
# DeMarre 速度标度
# ===========================================================================


class TestDeMarre:
    def test_zero_m_anchor(self) -> None:
        """0m 锚定：v_imp == v0 → P == pen0。"""
        assert penetration_demarre(700.0, 1555.0, 1555.0) == pytest.approx(700.0)

    def test_velocity_exponent(self) -> None:
        """P ∝ v^1.4286：v 减半 → P × 2^(−1.4286)。"""
        p_half = penetration_demarre(700.0, 1555.0, 777.5)
        assert p_half == pytest.approx(700.0 * (0.5) ** 1.428571)

    def test_zero_velocity(self) -> None:
        assert penetration_demarre(700.0, 1555.0, 0.0) == pytest.approx(0.0)


# ===========================================================================
# A-T 两段式长杆
# ===========================================================================


class TestAtErosion:
    def test_hydrodynamic_upper_bound(self) -> None:
        """纯伯努利（Yp=Rt=0）→ P = L·√(ρp/ρt)（流体动力学极限）。"""
        p = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, 0.0, 0.0, V0_M829)
        assert p == pytest.approx(L_M829 * (RHO_P_WHA / RHO_T_RHA) ** 0.5, rel=1e-3)

    def test_critical_velocity_boundary(self) -> None:
        """Rt>Yp：v_imp ≤ Vc → u=0 无穿深；略高于 Vc → 开始侵蚀。"""
        Vc = np.sqrt(2.0 * abs(Rt_RHA - Yp_WHA) / RHO_P_WHA)
        assert Vc == pytest.approx(709.1, rel=0.01)
        assert penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, Vc * 0.99) == pytest.approx(0.0)
        assert penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, Vc * 1.01) > 0.0

    def test_two_regime_rt_gt_yp_no_secondary(self) -> None:
        """Rt>Yp（靶强于杆）：低于临界速度后无刚性次级（P=0）。"""
        assert penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 400.0) == pytest.approx(0.0)

    def test_two_regime_rt_lt_yp_rigid_secondary(self) -> None:
        """Rt<Yp（杆强于靶）：v_imp < Vc/μ 时整杆刚性侵彻（P>0）。"""
        # 软靶 Rt=1e9 vs 杆 Yp=3e9
        Vc = np.sqrt(2.0 * abs(1.0e9 - Yp_WHA) / RHO_P_WHA)  # ≈478
        p_rigid = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, 1.0e9, Vc * 0.5)
        assert p_rigid > 0.0
        # 且高于 Vc 时侵蚀 + 刚性两段叠加，穿深更大
        p_high = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, 1.0e9, Vc * 2.0)
        assert p_high > p_rigid

    def test_monotone_in_velocity_length_density(self) -> None:
        base = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 1500.0)
        assert penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 1600.0) > base
        assert penetration_at_erosion(L_M829 * 1.2, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 1500.0) > base
        assert penetration_at_erosion(L_M829, RHO_P_WHA * 1.1, RHO_T_RHA, Yp_WHA, Rt_RHA, 1500.0) > base

    def test_monotone_in_target_resistance(self) -> None:
        """Rt 越大穿深越小。"""
        weak = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, 5.0e9, 1500.0)
        strong = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, 8.0e9, 1500.0)
        assert weak > strong

    def test_distance_decay_via_alpha(self) -> None:
        """α>0 → 距离越远 v_imp 越小 → A-T 穿深下降。"""
        near = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA,
                                      impact_velocity(V0_M829, ALPHA_M829, 500.0))
        far = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA,
                                     impact_velocity(V0_M829, ALPHA_M829, 4000.0))
        assert near > far

    def test_m829a3_magnitude(self) -> None:
        """量级自检：M829A3 0m ≈ 520mm（物理模型；标定锚为可选跟进）。"""
        p0 = penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, V0_M829)
        assert 450.0 < p0 < 600.0


# ===========================================================================
# 简化 Lanz-Odermatt
# ===========================================================================


class TestLanzOdermatt:
    def test_saturation_upper_bound(self) -> None:
        """V*→∞ 时 P → L（速度饱和，渐近）。"""
        p = penetration_lanz_odermatt(780.0, 17500.0, 7.4e9, 0.67, 3.0, 1e9)
        assert p == pytest.approx(780.0, rel=1e-3)

    def test_monotone_in_velocity(self) -> None:
        p1 = penetration_lanz_odermatt(780.0, 17500.0, 7.4e9, 0.67, 3.0, 1000.0)
        p2 = penetration_lanz_odermatt(780.0, 17500.0, 7.4e9, 0.67, 3.0, 2000.0)
        assert p2 > p1


# ===========================================================================
# 统一分派内核 + 单发判定
# ===========================================================================


class TestComputePenetration:
    def test_fixed(self) -> None:
        assert compute_penetration(int(PenetrationModel.FIXED), 700.0, 1555.0, 1200.0,
                                   L_M829, RHO_P_WHA, Yp_WHA, RHO_T_RHA, Rt_RHA, 0.67, 3.0) == pytest.approx(700.0)

    def test_demarre(self) -> None:
        p = compute_penetration(int(PenetrationModel.DEMARRE), 700.0, 1555.0, 1200.0,
                                L_M829, RHO_P_WHA, Yp_WHA, RHO_T_RHA, Rt_RHA, 0.67, 3.0)
        assert p == pytest.approx(penetration_demarre(700.0, 1555.0, 1200.0))

    def test_at_erosion(self) -> None:
        p = compute_penetration(int(PenetrationModel.AT_EROSION), 700.0, 1555.0, 1500.0,
                                L_M829, RHO_P_WHA, Yp_WHA, RHO_T_RHA, Rt_RHA, 0.67, 3.0)
        assert p == pytest.approx(
            penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 1500.0)
        )


class TestResolveV2:
    def _call(self, model, *, angle=0.0, v0=1555.0, alpha=0.0, range_m=0.0):
        margin = np.zeros(1, dtype=np.float32)
        ric = np.zeros(1, dtype=np.bool_)
        resolve_penetration_v2(
            model, 700.0, 500.0, angle, range_m, v0, alpha,
            L_M829, RHO_P_WHA, Yp_WHA, RHO_T_RHA, Rt_RHA, 0.67, 3.0,
            margin, ric,
        )
        return float(margin[0]), bool(ric[0])

    def test_soft_ricochet_uses_v_imp(self) -> None:
        """65° + 低速（v_imp<500）→ 跳弹；65° + 高速 → 不跳弹。"""
        m1, r1 = self._call(int(PenetrationModel.AT_EROSION), angle=65.0, v0=400.0)
        assert r1 and m1 < 0.0
        m2, r2 = self._call(int(PenetrationModel.AT_EROSION), angle=65.0, v0=1555.0)
        assert not r2

    def test_hard_angle_always_ricochet(self) -> None:
        _, r = self._call(int(PenetrationModel.AT_EROSION), angle=80.0)
        assert r

    def test_at_erosion_penetrates(self) -> None:
        m, r = self._call(int(PenetrationModel.AT_EROSION), v0=V0_M829)
        assert not r and m > 0.0

    # ---- 长杆跳飞判据（Rosenberg 式(60)，§5.7）----
    # βc 参考：v=1555→75.6°，v=1000→63.7°，v≈Vc(709)→47.5°

    def test_rosenberg_low_speed_ricochets_at_65(self) -> None:
        """1000 m/s（βc≈63.7°）：65° → 跳弹（低速更易跳，公式自带）。"""
        m, r = self._call(int(PenetrationModel.AT_EROSION), angle=65.0, v0=1000.0)
        assert r and m < 0.0

    def test_rosenberg_high_speed_no_ricochet_at_65(self) -> None:
        """1555 m/s（βc≈75.6°）：65° → 不跳弹。"""
        m, r = self._call(int(PenetrationModel.AT_EROSION), angle=65.0, v0=1555.0)
        assert not r

    def test_rosenberg_below_critical_angle_no_ricochet(self) -> None:
        """1000 m/s 下 60° < βc(63.7°) → 不跳弹。"""
        m, r = self._call(int(PenetrationModel.AT_EROSION), angle=60.0, v0=1000.0)
        assert not r

    def test_rosenberg_subcritical_falls_back_to_fixed(self) -> None:
        """v_imp ≤ Vc（刚性段）→ 判据不适用，回退固定阈值。"""
        # 700 < Vc≈709：80° 必跳（固定硬阈值）
        _, r1 = self._call(int(PenetrationModel.AT_EROSION), angle=80.0, v0=700.0)
        assert r1
        # 50° 低角：不跳弹；A-T 无穿深（v_imp<Vc → pen=0）
        m2, r2 = self._call(int(PenetrationModel.AT_EROSION), angle=50.0, v0=700.0)
        assert not r2 and m2 < 0.0

    def test_rosenberg_demarre_unaffected(self) -> None:
        """同速度同角度：DeMarre 维持固定阈值（65°/1000 m/s 不跳），长杆跳。"""
        _, r_dem = self._call(int(PenetrationModel.DEMARRE), angle=65.0, v0=1000.0)
        _, r_at = self._call(int(PenetrationModel.AT_EROSION), angle=65.0, v0=1000.0)
        assert not r_dem and r_at


# ===========================================================================
# 靶侧参数表
# ===========================================================================


class TestTargetModel:
    def test_rha_tate_calibration(self) -> None:
        """Tate 1986 标定量级：RHA → Rt≈7.4GPa，ρt≈7850。"""
        rho_t, Rt = TARGET_ARMOR_MODEL[ArmorType.RHA]
        assert rho_t == pytest.approx(7850.0)
        assert Rt == pytest.approx(7.4e9, rel=0.05)

    def test_all_armor_types_present(self) -> None:
        for t in (ArmorType.RHA, ArmorType.COMPOSITE, ArmorType.REACTIVE, ArmorType.SPACED):
            assert t in TARGET_ARMOR_MODEL


# ===========================================================================
# 端到端（FiringControl 命中链）
# ===========================================================================


class TestEndToEnd:
    def _make(self):
        armor = ArmorPool(capacity=64)
        boxes = BoxPool(capacity=512)
        projectiles = ProjectilePool(capacity=1024)
        ammo = AmmoManager(capacity=64)
        fuel = FuelManager(capacity=64)
        fc = FiringControl(armor, boxes, projectiles, ammo=ammo, fuel=fuel,
                           events=EventBus())
        return armor, boxes, projectiles, ammo, fuel, fc

    def _spawn(self, armor, boxes, ammo, fuel, fc, x, *, armor_mm=300.0,
               armor_type=int(ArmorType.RHA)):
        idx = armor.spawn(x, 0.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
        # HULL（整车血）+ TURRET（can_fire 要求活跃炮塔）
        bidx = boxes.spawn(idx, int(PartType.HULL), 1.5, 3.0, 1.0, 0.0, 0.0, 0.0)
        boxes.hp[bidx] = 100.0
        boxes.hp_max[bidx] = 100.0
        armor.bind_hull(idx, bidx)
        boxes.spawn(idx, int(PartType.TURRET), 1.0, 1.2, 0.6, 0.0, 0.0, 0.6)
        boxes.sync_all_from_parent(armor)
        fc.set_unit_config(
            idx, UnitCombatConfig(weapon_id="m256_120mm", armor_mm=armor_mm,
                                  armor_type=armor_type, crew_skill=0.9),
        )
        ammo.state.set_unit(idx, 40.0, 40.0)
        fuel.state.set_unit(idx, 500.0, 500.0)
        return idx

    def test_apfsds_at_penetrates_rha(self) -> None:
        """M829A3（A-T）正击 300mm RHA → 击穿 + 扣血（端到端分派管线）。"""
        armor, boxes, projectiles, ammo, fuel, fc = self._make()
        m1 = self._spawn(armor, boxes, ammo, fuel, fc, 0.0)
        t = self._spawn(armor, boxes, ammo, fuel, fc, 500.0, armor_mm=300.0)
        fc.fire(m1, t)
        fc.tick(1.0)
        hj = armor.hull_idx[t]
        assert boxes.hp[hj] < 100.0          # 击穿扣血（A-T 穿深 > 300mm 等效）
        kinds = [e.kind for e in fc.events.drain()]
        assert "hit" in kinds

    def test_apfsds_at_blocked_by_composite(self) -> None:
        """M829A3（A-T）vs 复合 900mm（等效 1350）→ 未击穿，仅震荡微降。"""
        armor, boxes, projectiles, ammo, fuel, fc = self._make()
        m1 = self._spawn(armor, boxes, ammo, fuel, fc, 0.0)
        t = self._spawn(armor, boxes, ammo, fuel, fc, 500.0,
                        armor_mm=900.0, armor_type=int(ArmorType.COMPOSITE))
        fc.fire(m1, t)
        fc.tick(1.0)
        hj = armor.hull_idx[t]
        assert 100.0 - boxes.hp[hj] <= 15.0  # 未击穿震荡（± 后效 × 震荡比例）

    def test_legacy_resolve_path_unchanged(self) -> None:
        """model=None（旧路径）行为保持：近距离 DeMarre 衰减 ≈ pen0。"""
        resolver = DamageResolver()
        resolver.set_armor_type(0, int(ArmorType.RHA))
        r = resolver.resolve(
            armor_idx=0, armor_mm=500.0, impact_angle_deg=0.0, range_m=500.0,
            pen_mm_rha=700.0, is_heat=False, drag_coefficient=0.15,
            muzzle_velocity=1555.0, rng_u=0.9,
        )
        assert r["margin"] > 0.0
        assert not r["ricochet"]

    def test_new_path_via_resolver(self) -> None:
        """model 传入 → 分派管线（A-T）产出实际穿深。"""
        resolver = DamageResolver()
        resolver.set_armor_type(0, int(ArmorType.RHA))
        r = resolver.resolve(
            armor_idx=0, armor_mm=300.0, impact_angle_deg=0.0, range_m=500.0,
            pen_mm_rha=700.0, is_heat=False, drag_coefficient=0.15,
            muzzle_velocity=V0_M829, rng_u=0.9,
            model=int(PenetrationModel.AT_EROSION),
            velocity_alpha=ALPHA_M829,
            penetrator_length_mm=L_M829, penetrator_density=RHO_P_WHA,
            penetrator_strength_pa=Yp_WHA,
            target_density=RHO_T_RHA, target_strength_pa=Rt_RHA,
        )
        assert r["margin"] > 0.0
        assert 400.0 < r["pen"] < 700.0


# ===========================================================================
# 性能冒烟（宽松上界，防 CI 抖动）
# ===========================================================================


class TestPerf:
    def test_at_erosion_1000_calls(self) -> None:
        t0 = time.perf_counter()
        for _ in range(1000):
            penetration_at_erosion(L_M829, RHO_P_WHA, RHO_T_RHA, Yp_WHA, Rt_RHA, 1500.0)
        dt = time.perf_counter() - t0
        assert dt < 1.0  # 1000 发 < 1s（单发亚毫秒量级）
