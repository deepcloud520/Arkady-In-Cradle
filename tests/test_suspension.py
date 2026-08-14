"""悬挂模拟测试（docs/suspension-slope-design.md §3 / §4.2 E1）。

覆盖：
    - solve_suspension：斜坡三点平面拟合 → pitch/roll/底盘 z
    - 弹簧阻尼：子步稳定（k·dt_sub²<1）、阶跃收敛无过冲
    - 空中单位跳过（mesh_id=-1）
    - 桥上单位姿态（水平，z=桥高）
    - resolve_contact_mesh_ids：触点越出面片回退 STRtree
    - _sync_boxes_kernel：Rz(yaw)·Rx(roll)·Ry(pitch) 连乘与解析一致
    - build_armor_movement_hook：端到端（移动 + 悬挂 + 触点 resolve）

运行：uv run python -m pytest tests/test_suspension.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from Airis.lulilo.entities.pools import ArmorPool, BoxPool
from Airis.lulilo.movement.integrate import (
    build_armor_movement_hook,
    resolve_contact_mesh_ids,
    solve_suspension,
)
from Airis.lulilo.simulation.loop import SimulationLoop
from Airis.lulilo.simulation.state import SimulationState
from Airis.lulilo.terrain.dual_contour import DualContourBuilder
from Airis.lulilo.terrain.heightfield import MAT_BRIDGE, Heightfield
from Airis.lulilo.terrain.world_mesh import flatten_world_mesh


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _slope_mesh(w: int = 16, h: int = 16, cs: float = 2.0, slope: float = 0.5):
    """斜坡地形：continuous_ground z = slope*x（世界 x 线性坡），量化层全 0。"""
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
    xs = (np.arange(w) + 0.5) * cs
    gx, _ = np.meshgrid(xs, xs)
    continuous = (slope * gx).astype(np.float32)
    mesh = DualContourBuilder(hf, continuous_ground=continuous).build()
    return hf, mesh


def _bridge_mesh(w: int = 16, h: int = 16, cs: float = 2.0):
    """带桥地形：桥面 x∈[8,20], y∈[12,16]，z=4（抬升 2m）。"""
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
    for gy in range(6, 8):
        for gx in range(4, 10):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
    mesh = DualContourBuilder(hf).build()
    return hf, mesh


def _make_armor(n: int = 8) -> ArmorPool:
    return ArmorPool(capacity=n)


# ---------------------------------------------------------------------------
# 三点平面拟合 + 弹簧阻尼
# ---------------------------------------------------------------------------


class TestSuspensionCore:
    def test_slope_pitch_roll_end_to_end(self):
        """斜坡 z=0.5x，车朝 x：pitch→atan(0.5)，roll→0，z→0.5*6=3。"""
        _, mesh = _slope_mesh(slope=0.5)
        armor = _make_armor()
        idx = armor.spawn(
            x=6.0, y=6.0, z=0.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        speed = np.zeros(armor.capacity, dtype=np.float32)
        hook = build_armor_movement_hook(armor, mesh, speed)
        loop = SimulationLoop(state=SimulationState())
        loop.attach("movement", hook)

        for _ in range(25):
            loop.step(1.0)  # 地面 tick

        assert armor.pitch[idx] == pytest.approx(math.atan(0.5), abs=0.03)
        assert armor.roll[idx] == pytest.approx(0.0, abs=0.03)
        assert armor.pos[idx, 2] == pytest.approx(3.0, abs=0.2)
        assert int(armor.mesh_id[idx]) >= 0

    def test_bridge_unit_stays_level(self):
        """桥上单位：触点全在桥面 → pitch/roll=0，z=桥高 4。"""
        _, mesh = _bridge_mesh()
        armor = _make_armor()
        # 桥面世界 x∈[8,20], y∈[12,16]，放车中心 (12,14)，L=3/W=2 全在桥上
        idx = armor.spawn(
            x=12.0, y=14.0, z=4.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        speed = np.zeros(armor.capacity, dtype=np.float32)
        hook = build_armor_movement_hook(armor, mesh, speed)
        loop = SimulationLoop(state=SimulationState())
        loop.attach("movement", hook)
        for _ in range(15):
            loop.step(1.0)

        assert armor.pitch[idx] == pytest.approx(0.0, abs=0.03)
        assert armor.roll[idx] == pytest.approx(0.0, abs=0.03)
        assert armor.pos[idx, 2] == pytest.approx(4.0, abs=0.2)

    def test_air_unit_skipped(self):
        """mesh_id=-1（空中）：姿态不变、pos.z 不被覆盖。"""
        _, mesh = _slope_mesh()
        armor = _make_armor()
        idx = armor.spawn(
            x=6.0, y=6.0, z=100.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        armor.mesh_id[idx] = -1
        armor.pitch[idx] = 0.5
        armor.roll[idx] = -0.3

        pv, vstart, vcnt, norm = flatten_world_mesh(mesh)
        solve_suspension(
            armor.pos, armor.yaw, armor.half, armor.active, armor.mesh_id,
            armor.contact_z, armor.contact_mesh_id,
            armor.pitch, armor.roll, armor.suspension,
            armor.pitch_vel, armor.roll_vel,
            1.0, 8.0, pv, vstart, vcnt, norm,
        )
        assert armor.pitch[idx] == pytest.approx(0.5)
        assert armor.roll[idx] == pytest.approx(-0.3)
        assert armor.pos[idx, 2] == pytest.approx(100.0)

    def test_spring_converges_no_overshoot(self):
        """平地：初始 pitch=0.5 阶跃 target=0，临界阻尼无过冲收敛。"""
        hf, mesh = _bridge_mesh()  # 平地（无 continuous_ground → 量化层）
        # 放地面单位（不在桥上）
        armor = _make_armor()
        idx = armor.spawn(
            x=4.0, y=4.0, z=0.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        armor.mesh_id[idx] = mesh.locate_patch(4.0, 4.0, hint_z=0.0)
        armor.pitch[idx] = 0.5  # 人为初始姿态（不平）

        pv, vstart, vcnt, norm = flatten_world_mesh(mesh)
        prev = abs(armor.pitch[idx])
        for _ in range(20):
            solve_suspension(
                armor.pos, armor.yaw, armor.half, armor.active, armor.mesh_id,
                armor.contact_z, armor.contact_mesh_id,
                armor.pitch, armor.roll, armor.suspension,
                armor.pitch_vel, armor.roll_vel,
                1.0, 8.0, pv, vstart, vcnt, norm,
            )
            cur = abs(armor.pitch[idx])
            assert cur <= prev + 1e-6  # 无过冲（临界阻尼单调收敛）
            prev = cur
        assert armor.pitch[idx] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 触点 resolve（跨面片 fallback）
# ---------------------------------------------------------------------------


class TestContactResolve:
    def test_contact_mesh_resolved_on_slope(self):
        """车长跨多个面片：触点 miss → resolve_contact_mesh_ids 修正。"""
        _, mesh = _slope_mesh(slope=0.5)
        armor = _make_armor()
        idx = armor.spawn(
            x=6.0, y=6.0, z=0.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        # 先定位单位面片，触点初始 -1
        armor.mesh_id[idx] = mesh.locate_patch(6.0, 6.0, hint_z=0.0)
        armor.contact_mesh_id[idx] = -1
        armor.contact_z[idx] = 0.0

        pv, vstart, vcnt, norm = flatten_world_mesh(mesh)
        solve_suspension(
            armor.pos, armor.yaw, armor.half, armor.active, armor.mesh_id,
            armor.contact_z, armor.contact_mesh_id,
            armor.pitch, armor.roll, armor.suspension,
            armor.pitch_vel, armor.roll_vel,
            1.0, 8.0, pv, vstart, vcnt, norm,
        )
        # 触点越出单位面片 → 标记 -1
        assert int(armor.contact_mesh_id[idx, 0]) == -1

        resolve_contact_mesh_ids(
            armor.pos, armor.yaw, armor.half,
            armor.contact_z, armor.contact_mesh_id, mesh,
        )
        # 全部触点定位成功，且 z = slope*x（精确）
        L, W = 3.0, 2.0
        for j, (lx, ly) in enumerate(
            [(-L, -W), (-L, W), (L, -W), (L, W)]
        ):
            assert int(armor.contact_mesh_id[idx, j]) >= 0
            cx = 6.0 + lx
            cy = 6.0 + ly
            assert armor.contact_z[idx, j] == pytest.approx(0.5 * cx, abs=0.2)


# ---------------------------------------------------------------------------
# 碰撞箱 Rz·Rx·Ry 连乘
# ---------------------------------------------------------------------------


class TestBoxPitchRoll:
    def _r3(self, x: float) -> np.ndarray:
        c, s = math.cos(x), math.sin(x)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _r_y(self, x: float) -> np.ndarray:
        c, s = math.cos(x), math.sin(x)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _r_z(self, x: float) -> np.ndarray:
        c, s = math.cos(x), math.sin(x)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def test_sync_boxes_matches_analytic(self):
        """_sync_boxes_kernel（Numba 批量）与解析 Rz·Rx·Ry 一致。"""
        armor = ArmorPool(capacity=4)
        aidx = armor.spawn(
            x=10.0, y=20.0, z=5.0, half_x=2.0, half_y=3.0, half_z=1.0,
            yaw=0.5, health=100.0,
        )
        armor.pitch[aidx] = 0.3
        armor.roll[aidx] = 0.2

        box = BoxPool(capacity=16)
        bidx = box.spawn(aidx, 0, 2.0, 3.0, 1.0, local_x=2.0, local_y=1.0, local_z=0.5)
        bidx2 = box.spawn(aidx, 0, 1.0, 1.0, 0.7, local_x=-2.0, local_y=-1.0, local_z=0.6)
        box.sync_all_from_parent(armor)  # Numba 批量版

        # 解析：world = pos + Rz(yaw)·Rx(roll)·Ry(pitch)·local
        R = self._r_z(0.5) @ self._r3(0.2) @ self._r_y(0.3)
        for bi, loc in ((bidx, (2.0, 1.0, 0.5)), (bidx2, (-2.0, -1.0, 0.6))):
            world = np.array([10.0, 20.0, 5.0]) + R @ np.array(loc)
            assert box.pos[bi][0] == pytest.approx(world[0], abs=1e-4)
            assert box.pos[bi][1] == pytest.approx(world[1], abs=1e-4)
            assert box.pos[bi][2] == pytest.approx(world[2], abs=1e-4)

    def test_sync_boxes_zero_attitude_backward_compat(self):
        """pitch=roll=0 → 与旧纯 yaw 变换一致（回归）。"""
        armor = ArmorPool(capacity=4)
        aidx = armor.spawn(x=1.0, y=2.0, z=3.0, yaw=0.7)
        box = BoxPool(capacity=16)
        bidx = box.spawn(aidx, 0, 1.0, 1.0, 1.0, local_x=1.0, local_y=2.0, local_z=0.5)
        box.sync_all_from_parent(armor)

        cy, sy = math.cos(0.7), math.sin(0.7)
        wx = 1.0 + 1.0 * cy - 2.0 * sy
        wy = 2.0 + 1.0 * sy + 2.0 * cy
        wz = 3.0 + 0.5
        assert box.pos[bidx][0] == pytest.approx(wx, abs=1e-4)
        assert box.pos[bidx][1] == pytest.approx(wy, abs=1e-4)
        assert box.pos[bidx][2] == pytest.approx(wz, abs=1e-4)


# ---------------------------------------------------------------------------
# 端到端：movement hook 管线
# ---------------------------------------------------------------------------


class TestMovementHook:
    def test_hook_moves_and_suspends(self):
        """车沿路标点爬坡：位置前进 + 姿态跟随（管线完整跑通）。"""
        _, mesh = _slope_mesh(slope=0.5)
        armor = _make_armor()
        idx = armor.spawn(
            x=4.0, y=6.0, z=0.0, half_x=2.0, half_y=3.0, half_z=1.0, yaw=0.0
        )
        # 沿 x 方向爬坡 4m（仍在 y=6 一行）
        armor.set_waypoints(idx, [(8.0, 6.0)])
        speed = np.full(armor.capacity, 1.0, dtype=np.float32)
        hook = build_armor_movement_hook(armor, mesh, speed)
        loop = SimulationLoop(state=SimulationState())
        loop.attach("movement", hook)

        for _ in range(8):
            loop.step(1.0)
        # 1m/s × 8s → 到达 (8,6)（dist 4m < 8m，已到）
        assert armor.pos[idx, 0] == pytest.approx(8.0, abs=0.01)
        # 斜坡上 pitch 收敛
        assert armor.pitch[idx] == pytest.approx(math.atan(0.5), abs=0.08)
        # z 跟随斜坡（x=8 → z≈4）
        assert armor.pos[idx, 2] == pytest.approx(4.0, abs=0.3)
