"""Phase 3 骨架冒烟测试（combat / logistics / simulation）。

覆盖每个新建模块的最小可用路径，验证骨架可导入、可运行：
    - weapons:    注册表 / 散布解算
    - collision:  胶囊体 vs AABB（半径膨胀 + Slabs）+ 空间哈希检测
    - damage:     斜距 / 穿透 / 跳弹 / 后效毁伤
    - logistics:  ammo / fuel 消耗与补给
    - simulation: CommandQueue / State / Loop 主循环

运行：uv run python -m pytest tests/test_phase3_skeleton.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.combat.collision import (
    CollisionDetector,
    segment_aabb_single,
)
from Airia.lulilo.combat.damage import (
    ArmorType,
    DamageResolver,
    compute_armor_eff,
    resolve_behind_armor_single,
    resolve_hit_dispersion,
    resolve_penetration_single,
)
from Airia.lulilo.combat.weapons import WeaponRegistry, WeaponSystem
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.logistics.fuel import FuelManager
from Airia.lulilo.simulation.commands import Command, CommandQueue, CommandType
from Airia.lulilo.simulation.loop import SimulationLoop
from Airia.lulilo.simulation.state import SimulationState


# ===========================================================================
# weapons
# ===========================================================================


class TestWeapons:
    def test_builtin_registry(self) -> None:
        reg = WeaponRegistry.with_builtin()
        ids = [w.weapon_id for w in reg.all_weapons()]
        assert "m256_120mm" in ids and "2a46m_125mm" in ids
        m256 = reg.get("m256_120mm")
        assert m256.caliber_mm == 120.0
        assert m256.base_accuracy_mrad == 0.2

    def test_get_definition_default_ammo(self) -> None:
        reg = WeaponRegistry.with_builtin()
        d = reg.get_definition("m256_120mm")
        assert d.weapon.weapon_id == "m256_120mm"
        assert d.ammo.ammo_id == "m829a3_apfsds"  # 默认取第一枚兼容弹药

    def test_get_definition_explicit_ammo(self) -> None:
        reg = WeaponRegistry.with_builtin()
        d = reg.get_definition("2a46m_125mm", "3bk18_heat")
        assert d.ammo.ammo_id == "3bk18_heat"
        assert d.is_heat

    def test_from_dict(self) -> None:
        reg = WeaponRegistry.from_dict(
            {
                "weapons": [
                    {"weapon_id": "w1", "caliber_mm": 90.0, "compatible_ammo": ["a1"]},
                ],
                "ammo": [
                    {"ammo_id": "a1", "pen_mm_rha": 300.0},
                ],
            }
        )
        assert reg.get("w1").caliber_mm == 90.0
        assert reg.get_ammo("a1").pen_mm_rha == 300.0

    def test_sigma_range_relation(self) -> None:
        """距离越远 sigma 越大。"""
        sys = WeaponSystem()
        near = sys.compute_sigma_m(500.0, 0.2)
        far = sys.compute_sigma_m(3000.0, 0.2)
        assert far > near

    def test_sigma_crew_skill_relation(self) -> None:
        """技能越高 sigma 越小。"""
        sys = WeaponSystem()
        green = sys.compute_sigma_m(2000.0, 0.2, crew_skill=0.5)
        elite = sys.compute_sigma_m(2000.0, 0.2, crew_skill=0.95)
        assert elite < green

    def test_solve(self) -> None:
        sys = WeaponSystem()
        sol = sys.solve("m256_120mm", None, 1000.0)
        assert sol.weapon_id == "m256_120mm"
        assert sol.sigma_m > 0.0
        assert sol.muzzle_velocity == pytest.approx(1555.0)


# ===========================================================================
# collision（胶囊体 vs AABB）
# ===========================================================================


class TestCollision:
    def test_segment_hit_center(self) -> None:
        """线段穿过 AABB 中心 → 命中。"""
        hit, t = segment_aabb_single(
            -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,   # p0 → p1
            0.0, 0.0, 0.0,                      # box 中心
            1.0, 3.0, 1.0,                      # half
        )
        assert hit
        assert 0.0 <= t <= 1.0

    def test_segment_miss_off_axis(self) -> None:
        """线段侧向偏移超出半轴 → 未命中（radius=0）。"""
        hit, _ = segment_aabb_single(
            -10.0, 3.5, 0.0, 10.0, 3.5, 0.0,
            0.0, 0.0, 0.0, 1.0, 3.0, 1.0,
        )
        assert not hit

    def test_radius_expansion_hits(self) -> None:
        """同一线段，radius 膨胀后命中（大弹头半径覆盖）。"""
        # y=3.5 超出 half_y=3.0 达 0.5；radius=0.5 → ey=3.5 → 命中
        hit, _ = segment_aabb_single(
            -10.0, 3.5, 0.0, 10.0, 3.5, 0.0,
            0.0, 0.0, 0.0, 1.0, 3.0, 1.0,
            0.5,  # radius
        )
        assert hit

    def test_fast_projectile_no_tunnel(self) -> None:
        """高速弹丸（单帧位移 > 目标厚度）不穿透（线段防穿透）。"""
        detector = CollisionDetector(grid_size=50.0)
        armor_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        armor_half = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        armor_active = np.array([True])
        detector.build_index(armor_pos, armor_half, armor_active)

        # 单帧位移 20m（dt=0.1, v=200），从 y=-15 穿到 y=+5，必穿过装甲
        proj_pos = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        proj_vel = np.array([[0.0, 200.0, 0.0]], dtype=np.float32)
        proj_radius = np.array([0.1], dtype=np.float32)
        proj_active = np.array([True])

        hits = detector.detect(
            proj_pos, proj_vel, proj_radius, proj_active,
            armor_pos, armor_half, armor_active, dt=0.1,
        )
        assert len(hits) == 1
        assert hits[0].armor_idx == 0

    def test_detect_miss(self) -> None:
        """弹丸远离装甲 → 无命中。"""
        detector = CollisionDetector(grid_size=50.0)
        armor_pos = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 0.0]], dtype=np.float32)
        armor_half = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        armor_active = np.array([True, True])

        proj_pos = np.array([[50.0, 50.0, 0.0]], dtype=np.float32)
        proj_vel = np.zeros((1, 3), dtype=np.float32)
        proj_radius = np.array([0.5], dtype=np.float32)
        proj_active = np.array([True])

        hits = detector.detect(
            proj_pos, proj_vel, proj_radius, proj_active,
            armor_pos, armor_half, armor_active, dt=0.1,
        )
        assert hits == []


# ===========================================================================
# damage
# ===========================================================================


class TestDamage:
    def test_armor_eff_normal(self) -> None:
        assert compute_armor_eff(500.0, ArmorType.RHA, 0.0, False) == pytest.approx(500.0)

    def test_armor_eff_oblique(self) -> None:
        # 60° 斜击 → 厚度翻倍（cos60=0.5）
        eff = compute_armor_eff(500.0, ArmorType.RHA, 60.0, False)
        assert eff == pytest.approx(1000.0)

    def test_armor_eff_composite_vs_heat(self) -> None:
        # 复合装甲 vs HEAT → ×2.5
        eff = compute_armor_eff(500.0, ArmorType.COMPOSITE, 0.0, True)
        assert eff == pytest.approx(1250.0)

    def test_penetration_close_ke(self) -> None:
        margin = np.zeros(1, dtype=np.float32)
        ric = np.zeros(1, dtype=np.bool_)
        resolve_penetration_single(
            700.0, 500.0, 0.0, 500.0, False, 0.15, 1555.0, margin, ric,
        )
        assert margin[0] > 0.0 and not ric[0]

    def test_penetration_far_decay(self) -> None:
        """远距衰减 + 高等效装甲 → 未击穿。"""
        margin = np.zeros(1, dtype=np.float32)
        ric = np.zeros(1, dtype=np.bool_)
        resolve_penetration_single(
            600.0, 650.0, 0.0, 4000.0, False, 1.0, 1500.0, margin, ric,
        )
        assert margin[0] <= 0.0

    def test_ricochet_hard_angle(self) -> None:
        margin = np.zeros(1, dtype=np.float32)
        ric = np.zeros(1, dtype=np.bool_)
        resolve_penetration_single(
            700.0, 500.0, 80.0, 500.0, False, 0.15, 1555.0, margin, ric,
        )
        assert ric[0]

    def test_behind_armor_no_pen(self) -> None:
        hdmg = np.zeros(1, dtype=np.float32)
        fire = np.zeros(1, dtype=np.bool_)
        cook = np.zeros(1, dtype=np.bool_)
        resolve_behind_armor_single(-50.0, 600.0, 0.9, hdmg, fire, cook)
        assert hdmg[0] > 0.0 and not fire[0] and not cook[0]

    def test_behind_armor_fire(self) -> None:
        hdmg = np.zeros(1, dtype=np.float32)
        fire = np.zeros(1, dtype=np.bool_)
        cook = np.zeros(1, dtype=np.bool_)
        resolve_behind_armor_single(100.0, 500.0, 0.05, hdmg, fire, cook)
        assert fire[0] and not cook[0]

    def test_behind_armor_cookoff(self) -> None:
        hdmg = np.zeros(1, dtype=np.float32)
        fire = np.zeros(1, dtype=np.bool_)
        cook = np.zeros(1, dtype=np.bool_)
        resolve_behind_armor_single(100.0, 500.0, 0.12, hdmg, fire, cook)
        assert cook[0] and not fire[0]

    def test_dispersion_zero_sigma(self) -> None:
        """sigma=0 → 落点 = 瞄准点。"""
        aim = np.array([[10.0, 20.0, 5.0]], dtype=np.float32)
        sigma = np.zeros(1, dtype=np.float32)
        rng = np.random.default_rng(0).random((1, 2)).astype(np.float32)
        out = np.zeros_like(aim)
        resolve_hit_dispersion(sigma, aim, rng, out)
        np.testing.assert_allclose(out, aim, atol=1e-5)

    def test_resolver_end_to_end(self) -> None:
        resolver = DamageResolver()
        resolver.set_armor_type(0, int(ArmorType.RHA))
        r = resolver.resolve(
            armor_idx=0, armor_mm=500.0, impact_angle_deg=0.0, range_m=500.0,
            pen_mm_rha=700.0, is_heat=False, drag_coefficient=0.15,
            muzzle_velocity=1555.0, rng_u=0.9,
        )
        assert r["margin"] > 0.0
        assert not r["ricochet"]
        assert r["health_dmg"] > 0.0


# ===========================================================================
# logistics
# ===========================================================================


class TestLogistics:
    def test_ammo_consume_partial(self) -> None:
        mgr = AmmoManager()
        mgr.state.set_unit(0, cap=40.0)
        used = mgr.consume(0, 5.0)
        assert used == pytest.approx(5.0)
        assert mgr.state.reserve[0] == pytest.approx(35.0)

    def test_ammo_consume_over(self) -> None:
        mgr = AmmoManager()
        mgr.state.set_unit(0, cap=10.0)
        used = mgr.consume(0, 50.0)  # 超出 → 耗尽剩余
        assert used == pytest.approx(10.0)
        assert mgr.state.reserve[0] == pytest.approx(0.0)

    def test_ammo_consume_batch(self) -> None:
        mgr = AmmoManager()
        mgr.state.set_unit(0, cap=10.0)
        mgr.state.set_unit(1, cap=10.0)
        consumed = mgr.consume_batch(np.array([3.0, 999.0], dtype=np.float32))
        assert consumed[0] == pytest.approx(3.0)
        assert consumed[1] == pytest.approx(10.0)
        assert mgr.state.reserve[1] == pytest.approx(0.0)

    def test_ammo_resupply_cap(self) -> None:
        mgr = AmmoManager()
        mgr.state.set_unit(0, cap=40.0)
        mgr.consume(0, 20.0)
        given = mgr.resupply(0, 100.0)  # 不超容量上限
        assert given == pytest.approx(20.0)
        assert mgr.state.reserve[0] == pytest.approx(40.0)

    def test_ammo_request_process(self) -> None:
        mgr = AmmoManager()
        mgr.state.set_unit(0, cap=40.0)
        mgr.state.set_unit(1, cap=40.0)
        mgr.consume(0, 30.0)
        mgr.consume(1, 10.0)
        mgr.request_supply(0, 30.0, priority=1)
        mgr.request_supply(1, 10.0, priority=0)
        filled = mgr.process_requests(supply_capacity=40.0)  # 优先填 0 号
        assert filled == 2
        assert mgr.state.reserve[0] == pytest.approx(40.0)
        assert mgr.state.reserve[1] == pytest.approx(40.0)

    def test_fuel_consume(self) -> None:
        mgr = FuelManager()
        mgr.state.set_unit(0, cap=500.0)
        used = mgr.consume(0, 120.0)
        assert used == pytest.approx(120.0)
        assert mgr.state.reserve[0] == pytest.approx(380.0)

    def test_fuel_request(self) -> None:
        mgr = FuelManager()
        mgr.state.set_unit(0, cap=500.0)
        mgr.consume(0, 500.0)
        mgr.request_supply(0, 300.0)
        assert mgr.process_requests() == 1
        assert mgr.state.reserve[0] == pytest.approx(300.0)


# ===========================================================================
# simulation
# ===========================================================================


class TestSimulation:
    def test_command_queue(self) -> None:
        q = CommandQueue()
        q.push(Command(CommandType.STRIKE, "t-90-03", {"weapon": "m256"}))
        q.push(Command(CommandType.HOLD, "m1a2-01"))
        assert len(q) == 2
        cmds = q.drain()
        assert len(cmds) == 2
        assert cmds[0].type == CommandType.STRIKE
        assert cmds[0].target_id == "t-90-03"
        assert len(q) == 0

    def test_state_snapshot(self) -> None:
        st = SimulationState(sim_time=1.5, tick_count=3)
        snap = st.snapshot()
        assert snap.sim_time == 1.5
        assert snap.tick_count == 3
        assert snap.pools["armor"] is None

    def test_loop_step_advances(self) -> None:
        loop = SimulationLoop()
        loop.step(1.0)
        assert loop.state.sim_time == pytest.approx(1.0)
        assert loop.state.tick_count == 1

    def test_loop_run_headless(self) -> None:
        loop = SimulationLoop()
        ticks = loop.run_headless(10.0, dt=1.0)
        assert ticks == 10
        assert loop.state.sim_time == pytest.approx(10.0)

    def test_loop_hook_called(self) -> None:
        loop = SimulationLoop()
        calls: list[str] = []

        def move_hook(_loop: SimulationLoop, dt: float) -> None:
            _ = _loop  # hook 签名保留 loop 引用（后续子系统用）
            calls.append(f"move:{dt}")

        loop.attach("movement", move_hook)
        loop.step(0.5)
        assert calls == ["move:0.5"]

    def test_loop_drains_commands(self) -> None:
        loop = SimulationLoop()
        loop.commands.push(Command(CommandType.STRIKE, "t-90"))
        loop.step(1.0)
        kinds = [e.data["type"] for e in loop.events.drain()]
        assert "STRIKE" in kinds

    def test_loop_paused_no_advance(self) -> None:
        loop = SimulationLoop()
        loop.sim_paused = True
        loop.step(1.0)
        assert loop.state.tick_count == 0
        assert loop.state.sim_time == 0.0
