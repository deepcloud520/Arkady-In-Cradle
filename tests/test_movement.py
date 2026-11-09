"""movement 层单元测试：寻路 + Numba 移动积分。

覆盖：路标点推进、到达判定、朝向更新、抛射物重力、坡度速度、批量并行。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.entities.pools import MAX_ARMOR, MAX_PROJ
from Airia.lulilo.movement.integrate import (
    compute_speed,
    integrate_armor_movement,
    integrate_projectile_movement,
    integrate_single_armor,
)


# ---------------------------------------------------------------------------
# integrate_single_armor
# ---------------------------------------------------------------------------


class TestIntegrateSingle:
    """单装甲移动积分测试。"""

    def test_move_forward(self) -> None:
        """沿 X 方向前进。"""
        px, py, yaw, arrived = integrate_single_armor(
            0.0, 0.0, 5.0, 100.0, 0.0, speed=20.0, dt=1.0,
        )
        assert px == pytest.approx(20.0)
        assert py == pytest.approx(0.0)
        assert yaw == pytest.approx(0.0)  # 朝向 +X
        assert not arrived

    def test_move_diagonal(self) -> None:
        """对角线方向前进。"""
        px, py, yaw, arrived = integrate_single_armor(
            0.0, 0.0, 0.0, 10.0, 10.0, speed=14.142, dt=0.5,
        )
        # 方向 (1,1)/√2，步长 = 14.142*0.5 = 7.071
        assert px == pytest.approx(5.0, abs=0.1)
        assert py == pytest.approx(5.0, abs=0.1)
        assert yaw == pytest.approx(np.pi / 4, abs=0.01)

    def test_arrive_exact(self) -> None:
        """刚好到达目标点。"""
        px, py, yaw, arrived = integrate_single_armor(
            0.0, 0.0, 0.0, 10.0, 0.0, speed=10.0, dt=1.0,
        )
        assert px == pytest.approx(10.0)
        assert arrived

    def test_arrive_overshoot(self) -> None:
        """步长大于距离时停在目标点。"""
        px, py, yaw, arrived = integrate_single_armor(
            0.0, 0.0, 0.0, 5.0, 0.0, speed=100.0, dt=1.0,
        )
        assert px == pytest.approx(5.0)  # 停在目标点
        assert py == pytest.approx(0.0)
        assert arrived

    def test_yaw_updates(self) -> None:
        """朝向应指向运动方向。"""
        _, _, yaw, _ = integrate_single_armor(
            0.0, 0.0, 0.0, -10.0, 0.0, speed=10.0, dt=0.5,
        )
        assert yaw == pytest.approx(np.pi, abs=0.01)  # 朝向 -X

    def test_stationary_when_zero_speed(self) -> None:
        """速度为 0 时不应移动。"""
        px, py, yaw, arrived = integrate_single_armor(
            0.0, 0.0, 0.0, 100.0, 0.0, speed=0.0, dt=1.0,
        )
        assert px == pytest.approx(0.0)
        assert not arrived


# ---------------------------------------------------------------------------
# integrate_armor_movement (批量)
# ---------------------------------------------------------------------------


class TestIntegrateBatch:
    """批量装甲移动积分测试。"""

    @pytest.fixture
    def arrays(self) -> dict:
        N = 8
        pos = np.zeros((N, 3), dtype=np.float32)
        pos[:, 0] = np.arange(N, dtype=np.float32) * 10
        yaw = np.zeros(N, dtype=np.float32)
        active = np.ones(N, dtype=np.bool_)
        wp = np.zeros((N, 8, 2), dtype=np.float32)
        wp[:, 0, 0] = 200.0  # 所有单位目标 x=200
        wp[:, 0, 1] = np.arange(N, dtype=np.float32) * 10
        wp_count = np.ones(N, dtype=np.uint8)
        wp_idx = np.zeros(N, dtype=np.uint8)
        speed = np.full(N, 10.0, dtype=np.float32)
        vel = np.zeros((N, 3), dtype=np.float32)
        return {
            "pos": pos, "yaw": yaw, "active": active,
            "wp": wp, "wp_count": wp_count, "wp_idx": wp_idx,
            "speed": speed, "vel": vel,
        }

    def test_all_move_forward(self, arrays: dict) -> None:
        """所有活跃单位应向前移动。"""
        old_x = arrays["pos"][:, 0].copy()
        integrate_armor_movement(
            arrays["pos"], arrays["yaw"], arrays["active"],
            arrays["wp"], arrays["wp_count"], arrays["wp_idx"],
            arrays["speed"], arrays["vel"], dt=1.0,
        )
        assert np.all(arrays["pos"][:, 0] > old_x)

    def test_inactive_skipped(self, arrays: dict) -> None:
        """inactive 单位不应移动。"""
        arrays["active"][3] = False
        old_pos3 = arrays["pos"][3].copy()
        integrate_armor_movement(
            arrays["pos"], arrays["yaw"], arrays["active"],
            arrays["wp"], arrays["wp_count"], arrays["wp_idx"],
            arrays["speed"], arrays["vel"], dt=1.0,
        )
        assert np.all(arrays["pos"][3] == old_pos3)

    def test_no_waypoints_stationary(self, arrays: dict) -> None:
        """无路标点时不应移动。"""
        arrays["wp_count"][:] = 0
        old_pos = arrays["pos"].copy()
        integrate_armor_movement(
            arrays["pos"], arrays["yaw"], arrays["active"],
            arrays["wp"], arrays["wp_count"], arrays["wp_idx"],
            arrays["speed"], arrays["vel"], dt=1.0,
        )
        assert np.allclose(arrays["pos"], old_pos)

    def test_arrive_at_waypoint_advances_index(self, arrays: dict) -> None:
        """到达路标点后 wp_idx 应递增。"""
        # 把目标设得很近，确保 1s 内到达
        arrays["wp"][:, 0, 0] = arrays["pos"][:, 0] + 5.0
        arrays["wp"][:, 0, 1] = arrays["pos"][:, 1]
        arrays["wp_count"][:] = 2
        arrays["wp"][:, 1, 0] = arrays["pos"][:, 0] + 100.0
        arrays["speed"][:] = 20.0
        integrate_armor_movement(
            arrays["pos"], arrays["yaw"], arrays["active"],
            arrays["wp"], arrays["wp_count"], arrays["wp_idx"],
            arrays["speed"], arrays["vel"], dt=1.0,
        )
        # 到达第一个路标点后 wp_idx 应为 1
        assert np.all(arrays["wp_idx"] == 1)

    def test_last_waypoint_clears_path(self, arrays: dict) -> None:
        """到达最后一个路标点后应清除路径。"""
        arrays["wp"][:, 0, 0] = arrays["pos"][:, 0] + 3.0
        arrays["wp"][:, 0, 1] = arrays["pos"][:, 1]
        arrays["wp_count"][:] = 1
        arrays["speed"][:] = 20.0
        integrate_armor_movement(
            arrays["pos"], arrays["yaw"], arrays["active"],
            arrays["wp"], arrays["wp_count"], arrays["wp_idx"],
            arrays["speed"], arrays["vel"], dt=1.0,
        )
        assert np.all(arrays["wp_count"] == 0)
        assert np.all(arrays["wp_idx"] == 0)

    def test_full_capacity_no_crash(self) -> None:
        """MAX_ARMOR 全量移动不应崩溃。"""
        N = MAX_ARMOR
        pos = np.random.default_rng(42).uniform(0, 2000, (N, 3)).astype(np.float32)
        yaw = np.zeros(N, dtype=np.float32)
        active = np.ones(N, dtype=np.bool_)
        wp = np.zeros((N, 8, 2), dtype=np.float32)
        wp[:, 0, 0] = 500.0
        wp[:, 0, 1] = np.arange(N, dtype=np.float32) * 0.5
        wp_count = np.ones(N, dtype=np.uint8)
        wp_idx = np.zeros(N, dtype=np.uint8)
        speed = np.full(N, 15.0, dtype=np.float32)
        vel = np.zeros((N, 3), dtype=np.float32)

        integrate_armor_movement(pos, yaw, active, wp, wp_count, wp_idx, speed, vel, dt=0.016)
        assert np.all(pos[:, 0] > 0)  # 都向前移动了

    def test_velocity_written_when_moving(self) -> None:
        """移动积分应写入 vel = 本 tick 位移 / dt（火控前置量数据源）。"""
        N = 4
        pos = np.zeros((N, 3), dtype=np.float32)
        yaw = np.zeros(N, dtype=np.float32)
        active = np.ones(N, dtype=np.bool_)
        wp = np.zeros((N, 8, 2), dtype=np.float32)
        wp[:, 0, 0] = 50.0   # 目标 x=50，纯 +x 方向
        wp[:, 0, 1] = 0.0
        wp_count = np.ones(N, dtype=np.uint8)
        wp_idx = np.zeros(N, dtype=np.uint8)
        speed = np.full(N, 10.0, dtype=np.float32)
        vel = np.zeros((N, 3), dtype=np.float32)

        integrate_armor_movement(pos, yaw, active, wp, wp_count, wp_idx, speed, vel, dt=1.0)
        assert np.allclose(vel[:, 0], 10.0)
        assert np.allclose(vel[:, 1], 0.0)
        assert np.allclose(vel[:, 2], 0.0)

    def test_velocity_zero_when_stationary(self) -> None:
        """无路标点（静止）→ vel 归零（不残留旧速度）。"""
        N = 2
        pos = np.zeros((N, 3), dtype=np.float32)
        yaw = np.zeros(N, dtype=np.float32)
        active = np.ones(N, dtype=np.bool_)
        wp = np.zeros((N, 8, 2), dtype=np.float32)
        wp_count = np.zeros(N, dtype=np.uint8)  # 无路标点
        wp_idx = np.zeros(N, dtype=np.uint8)
        speed = np.full(N, 10.0, dtype=np.float32)
        vel = np.full((N, 3), 99.0, dtype=np.float32)

        integrate_armor_movement(pos, yaw, active, wp, wp_count, wp_idx, speed, vel, dt=1.0)
        assert np.allclose(vel, 0.0)


# ---------------------------------------------------------------------------
# integrate_projectile_movement
# ---------------------------------------------------------------------------


class TestIntegrateProjectile:
    """抛射物移动积分测试。"""

    def test_gravity_pulls_down(self) -> None:
        """重力应减小 Z 速度。"""
        M = 4
        pos = np.zeros((M, 3), dtype=np.float32)
        vel = np.zeros((M, 3), dtype=np.float32)
        vel[:, 2] = [10, 20, 0, -5]
        active = np.ones(M, dtype=np.bool_)

        integrate_projectile_movement(pos, vel, active, dt=0.1, gravity=9.81)

        # Z 速度应减少 0.981
        assert vel[0, 2] == pytest.approx(10 - 0.981)
        assert vel[1, 2] == pytest.approx(20 - 0.981)
        assert vel[2, 2] == pytest.approx(0 - 0.981)
        assert vel[3, 2] == pytest.approx(-5 - 0.981)

    def test_position_moves(self) -> None:
        """位置应按速度积分。"""
        pos = np.zeros((1, 3), dtype=np.float32)
        vel = np.array([[800.0, 0.0, 0.0]], dtype=np.float32)
        active = np.ones(1, dtype=np.bool_)

        integrate_projectile_movement(pos, vel, active, dt=0.01)

        assert pos[0, 0] == pytest.approx(8.0)
        assert pos[0, 1] == pytest.approx(0.0)

    def test_inactive_skipped(self) -> None:
        """inactive 弹丸不应移动。"""
        pos = np.array([[0, 0, 100]], dtype=np.float32)
        vel = np.array([[500, 0, 0]], dtype=np.float32)
        active = np.array([False])

        integrate_projectile_movement(pos, vel, active, dt=0.1)

        assert pos[0, 0] == pytest.approx(0.0)  # 没动

    def test_full_capacity_no_crash(self) -> None:
        """MAX_PROJ 全量移动不应崩溃。"""
        M = MAX_PROJ
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 2000, (M, 3)).astype(np.float32)
        vel = rng.uniform(-900, 900, (M, 3)).astype(np.float32)
        active = np.ones(M, dtype=np.bool_)

        integrate_projectile_movement(pos, vel, active, dt=0.01)
        # 不应崩溃


# ---------------------------------------------------------------------------
# compute_speed
# ---------------------------------------------------------------------------


class TestComputeSpeed:
    """坡度速度计算测试。"""

    def test_flat_is_full_speed(self) -> None:
        """平地 = 全速。"""
        assert compute_speed(20.0, 0.0) == pytest.approx(20.0)

    def test_gentle_slope_reduces(self) -> None:
        """缓坡应减速但可通行。"""
        s = compute_speed(20.0, 15.0, max_slope_deg=30.0)
        assert 0 < s < 20.0

    def test_max_slope_is_zero(self) -> None:
        """恰好等于最大坡度 → 速度为 0。"""
        assert compute_speed(20.0, 30.0, max_slope_deg=30.0) == 0.0

    def test_exceeded_slope_is_zero(self) -> None:
        """超过最大坡度 → 速度为 0。"""
        assert compute_speed(20.0, 45.0, max_slope_deg=30.0) == 0.0

    def test_custom_params(self) -> None:
        """自定义最大坡度和最小速度因子。"""
        s = compute_speed(10.0, 5.0, max_slope_deg=10.0, min_speed_factor=0.5)
        # 坡度 5/10 = 50% 减速，因子 = 1 - 0.5*0.5 = 0.75
        assert s == pytest.approx(7.5)
