"""detection 层单元测试：DDA 视线检测 + 传感器模型。

覆盖：LOS 阻挡/清晰、弹道拦截、雷达/红外/ESM 距离与概率、批量评估。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.detection.los import (
    LosChecker,
    line_of_sight,
    line_of_sight_batch,
    terrain_intercept,
)
from Airia.lulilo.detection.sensors import (
    RCS_APC,
    RCS_HELICOPTER,
    RCS_PERSON,
    RCS_TANK,
    RCS_TRUCK,
    IR_TANK_FIRING,
    IR_TANK_IDLE,
    IR_TANK_MOVING,
    esm_detection_probability,
    esm_detection_range,
    evaluate_sensor_detection,
    ir_detection_probability,
    ir_detection_range,
    radar_detection_probability,
    radar_detection_range,
)
from Airia.lulilo.terrain.heightfield import MAT_BRIDGE, MAT_TERRAIN, Heightfield


# ---------------------------------------------------------------------------
# 共享 fixture：带地面 + 一堵墙的高度场
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_hf() -> Heightfield:
    """20×20 平地高度场，地面 z=0。"""
    hf = Heightfield(20, 20, cs=10.0, ch=1.0, z_min=-50, z_max=200)
    for gy in range(20):
        for gx in range(20):
            hf.add_span(gx, gy, -50, 0, mat=MAT_TERRAIN)
    return hf


@pytest.fixture
def wall_hf() -> Heightfield:
    """20×20 高度场，中间有一堵 20m 高墙 (gx=10)。"""
    hf = Heightfield(20, 20, cs=10.0, ch=1.0, z_min=-50, z_max=200)
    for gy in range(20):
        for gx in range(20):
            hf.add_span(gx, gy, -50, 0, mat=MAT_TERRAIN)
    # 墙：gx=10，z 从 -50 到 20
    for gy in range(20):
        hf.add_span(10, gy, -50, 20, mat=MAT_BRIDGE)
    return hf


@pytest.fixture
def bridge_hf() -> Heightfield:
    """含桥面的高度场：地面 0~2m + 桥面 12~14m。"""
    hf = Heightfield(20, 20, cs=10.0, ch=1.0, z_min=-50, z_max=200)
    for gy in range(20):
        for gx in range(20):
            hf.add_span(gx, gy, -50, 2, mat=MAT_TERRAIN)
    for gy in range(5, 15):
        for gx in range(5, 15):
            hf.add_span(gx, gy, 12, 14, mat=MAT_BRIDGE)
    return hf


# ---------------------------------------------------------------------------
# DDA 视线检测
# ---------------------------------------------------------------------------


class TestLineOfSight:
    """line_of_sight Numba 函数测试。"""

    def test_same_point_sees(self, flat_hf: Heightfield) -> None:
        """同一点应可见。"""
        assert line_of_sight(
            10, 10, 5, 10, 10, 5,
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
        )

    def test_clear_above_ground(self, flat_hf: Heightfield) -> None:
        """地平线上方可互见。"""
        assert line_of_sight(
            10, 10, 5, 150, 10, 5,
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
        )

    def test_wall_blocks(self, wall_hf: Heightfield) -> None:
        """墙应阻挡视线。"""
        # 观察者在墙左边，目标在墙右边，z=5（低于墙顶 20）
        result = line_of_sight(
            10, 100, 5, 150, 100, 5,
            wall_hf.spans, wall_hf.count, wall_hf.width,
            wall_hf.cs, wall_hf.ch, wall_hf.z_min,
        )
        assert not result, "墙高 20m，z=5 的射线应被阻挡"

    def test_above_wall_sees(self, wall_hf: Heightfield) -> None:
        """高于墙的视线应可见。"""
        result = line_of_sight(
            10, 100, 25, 150, 100, 25,
            wall_hf.spans, wall_hf.count, wall_hf.width,
            wall_hf.cs, wall_hf.ch, wall_hf.z_min,
        )
        assert result, "z=25 高于墙顶 20，应可见"

    def test_observer_high_target_low(self, flat_hf: Heightfield) -> None:
        """高处观察者看低处目标。"""
        assert line_of_sight(
            10, 10, 50, 150, 10, 2,
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
        )

    def test_max_range_exceeded(self, flat_hf: Heightfield) -> None:
        """超距应返回 False。"""
        result = line_of_sight(
            0, 0, 5, 10000, 0, 5,
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
            max_range=100.0,
        )
        assert not result

    def test_bridge_underpass(self, bridge_hf: Heightfield) -> None:
        """桥下视线：桥下空间应可穿透。"""
        # 桥面板在 12~14m，射线在 z=8 穿过桥下
        los = LosChecker(bridge_hf)
        # 桥区域外 → 桥区域外，z=8 在桥下
        assert los.check(20, 50, 8, 180, 50, 8), "z=8 在桥面板下方，应可见"


class TestLosChecker:
    """LosChecker Python 封装测试。"""

    def test_check_wrapper(self, flat_hf: Heightfield) -> None:
        """LosChecker.check 应与 line_of_sight 一致。"""
        los = LosChecker(flat_hf)
        assert los.check(10, 10, 5, 150, 10, 5)

    def test_check_batch(self, flat_hf: Heightfield) -> None:
        """批量检测应返回正确形状。"""
        los = LosChecker(flat_hf)
        N = 10
        ox = np.full(N, 10.0, dtype=np.float32)
        oy = np.linspace(0, 190, N).astype(np.float32)
        oz = np.full(N, 5.0, dtype=np.float32)
        tx = np.full(N, 150.0, dtype=np.float32)
        ty = np.linspace(0, 190, N).astype(np.float32)
        tz = np.full(N, 5.0, dtype=np.float32)

        result = los.check_batch(ox, oy, oz, tx, ty, tz)
        assert result.shape == (N,)
        assert result.dtype == np.bool_


class TestTerrainIntercept:
    """弹道拦截点测试。"""

    def test_hits_ground(self, flat_hf: Heightfield) -> None:
        """向下射击应命中地面。"""
        t = terrain_intercept(
            50, 50, 30,   # 起点 (50,50,30)
            0, 0, -10,    # 方向：直直向下
            50.0,         # 最大搜索
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
        )
        # 地面在 z=0，从 z=30 以 -10/s 下降，约 3 秒命中
        assert 0 < t <= 50.0, f"应命中地面，t={t}"

    def test_miss_goes_past(self, flat_hf: Heightfield) -> None:
        """向上射击应未命中。"""
        t = terrain_intercept(
            50, 50, 30,
            0, 0, 10,     # 纯向上
            100.0,
            flat_hf.spans, flat_hf.count, flat_hf.width,
            flat_hf.cs, flat_hf.ch, flat_hf.z_min,
        )
        assert t > 100.0, "向上射击不应命中地面"


# ---------------------------------------------------------------------------
# 传感器模型
# ---------------------------------------------------------------------------


class TestRadar:
    """雷达模型测试。"""

    def test_rcs_ordering(self) -> None:
        """RCS 越大检测距离越远。"""
        r_tank = radar_detection_range(RCS_TANK)
        r_apc = radar_detection_range(RCS_APC)
        r_truck = radar_detection_range(RCS_TRUCK)
        r_person = radar_detection_range(RCS_PERSON)
        assert r_tank > r_apc > r_truck > r_person

    def test_detection_at_half_range(self) -> None:
        """半程处检测概率应接近 1。"""
        max_r = radar_detection_range(RCS_TANK)
        p = radar_detection_probability(max_r * 0.3, max_r)
        assert p > 0.95

    def test_detection_at_max_range(self) -> None:
        """最大距离处概率应接近 0.5。"""
        max_r = radar_detection_range(RCS_TANK)
        p = radar_detection_probability(max_r, max_r)
        assert 0.45 < p < 0.55

    def test_detection_past_range(self) -> None:
        """超过 1.5× 最大距离概率应为 0。"""
        max_r = radar_detection_range(RCS_TANK)
        p = radar_detection_probability(max_r * 2.0, max_r)
        assert p < 0.01

    def test_probability_zero_distance(self) -> None:
        """距离为 0 时应为 1。"""
        p = radar_detection_probability(0.0, 5000.0)
        assert p == pytest.approx(1.0)


class TestInfrared:
    """红外模型测试。"""

    def test_ir_ordering(self) -> None:
        """红外特征越大检测距离越远。"""
        r_firing = ir_detection_range(IR_TANK_FIRING)
        r_moving = ir_detection_range(IR_TANK_MOVING)
        r_idle = ir_detection_range(IR_TANK_IDLE)
        assert r_firing > r_moving > r_idle

    def test_detection_probability(self) -> None:
        """基本概率范围检查。"""
        max_r = ir_detection_range(IR_TANK_MOVING)
        p_near = ir_detection_probability(0.0, max_r)
        p_mid = ir_detection_probability(max_r * 0.7, max_r)
        p_far = ir_detection_probability(max_r * 1.3, max_r)
        assert p_near == pytest.approx(1.0)
        assert 0.4 < p_mid < 0.6
        assert p_far < 0.1


class TestEsm:
    """ESM 模型测试。"""

    def test_range_positive(self) -> None:
        """ESM 检测距离应为正数。"""
        r = esm_detection_range()
        assert r > 0

    def test_esm_longer_than_radar(self) -> None:
        """ESM（单程）检测距离应大于雷达（往返）。"""
        r_esm = esm_detection_range()
        r_radar = radar_detection_range(RCS_TANK)
        assert r_esm > r_radar, "ESM 单程应有更远距离"

    def test_probability(self) -> None:
        """ESM 检测概率曲线。"""
        max_r = esm_detection_range()
        p = esm_detection_probability(max_r * 0.5, max_r)
        assert p > 0.9


class TestEvaluateSensorDetection:
    """批量传感器评估测试。"""

    def test_shape_and_nonnegative(self) -> None:
        """输出缓存形状正确且非负。"""
        N, M = 5, 10
        rng = np.random.default_rng(42)
        sx = rng.uniform(0, 2000, N).astype(np.float32)
        sy = rng.uniform(0, 2000, N).astype(np.float32)
        sz = np.full(N, 10.0, dtype=np.float32)
        tx = rng.uniform(0, 2000, M).astype(np.float32)
        ty = rng.uniform(0, 2000, M).astype(np.float32)
        tz = np.full(M, 2.0, dtype=np.float32)
        stype = np.zeros(N, dtype=np.int32)
        trcs = np.full(M, RCS_TANK, dtype=np.float32)
        tir = np.full(M, 0.5, dtype=np.float32)
        sact = np.ones(N, dtype=np.bool_)
        tact = np.ones(M, dtype=np.bool_)
        cache = np.zeros((N, M), dtype=np.float32)

        evaluate_sensor_detection(
            sx, sy, sz, tx, ty, tz, stype, trcs, tir, sact, tact, cache,
        )
        assert cache.shape == (N, M)
        assert np.all(cache >= 0)

    def test_inactive_sensors_ignored(self) -> None:
        """inactive 传感器行应全为 0。"""
        N, M = 3, 5
        sx = np.array([100, 200, 300], dtype=np.float32)
        sy = np.zeros(N, dtype=np.float32)
        sz = np.full(N, 10.0, dtype=np.float32)
        tx = np.ones(M, dtype=np.float32) * 500
        ty = np.zeros(M, dtype=np.float32)
        tz = np.full(M, 2.0, dtype=np.float32)
        stype = np.zeros(N, dtype=np.int32)
        trcs = np.full(M, RCS_TANK, dtype=np.float32)
        tir = np.full(M, 0.5, dtype=np.float32)
        sact = np.array([True, False, True], dtype=np.bool_)
        tact = np.ones(M, dtype=np.bool_)
        cache = np.zeros((N, M), dtype=np.float32)

        evaluate_sensor_detection(
            sx, sy, sz, tx, ty, tz, stype, trcs, tir, sact, tact, cache,
        )
        assert np.all(cache[1, :] == 0.0), "inactive 传感器行应全零"

    def test_inactive_targets_ignored(self) -> None:
        """inactive 目标列应全为 0。"""
        N, M = 2, 4
        sx = np.array([100, 200], dtype=np.float32)
        sy = np.zeros(N, dtype=np.float32)
        sz = np.full(N, 10.0, dtype=np.float32)
        tx = np.arange(M, dtype=np.float32) * 50
        ty = np.zeros(M, dtype=np.float32)
        tz = np.full(M, 2.0, dtype=np.float32)
        stype = np.zeros(N, dtype=np.int32)
        trcs = np.full(M, RCS_TANK, dtype=np.float32)
        tir = np.full(M, 0.5, dtype=np.float32)
        sact = np.ones(N, dtype=np.bool_)
        tact = np.array([True, False, True, False], dtype=np.bool_)
        cache = np.zeros((N, M), dtype=np.float32)

        evaluate_sensor_detection(
            sx, sy, sz, tx, ty, tz, stype, trcs, tir, sact, tact, cache,
        )
        assert np.all(cache[:, 1] == 0.0)
        assert np.all(cache[:, 3] == 0.0)
