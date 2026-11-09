"""engine.detection 层：传感器模型 + 视线检测（AGENTS.md Phase 2）。"""

from Airia.lulilo.detection.los import LosChecker, line_of_sight, line_of_sight_batch, terrain_intercept
from Airia.lulilo.detection.sensors import (
    RCS_APC,
    RCS_HELICOPTER,
    RCS_PERSON,
    RCS_TANK,
    RCS_TRUCK,
    IR_APC,
    IR_HELICOPTER,
    IR_PERSON,
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

__all__ = [
    # LOS
    "LosChecker",
    "line_of_sight",
    "line_of_sight_batch",
    "terrain_intercept",
    # RCS 常量
    "RCS_TANK",
    "RCS_APC",
    "RCS_TRUCK",
    "RCS_HELICOPTER",
    "RCS_PERSON",
    # IR 常量
    "IR_TANK_IDLE",
    "IR_TANK_MOVING",
    "IR_TANK_FIRING",
    "IR_APC",
    "IR_HELICOPTER",
    "IR_PERSON",
    # 雷达
    "radar_detection_range",
    "radar_detection_probability",
    # 红外
    "ir_detection_range",
    "ir_detection_probability",
    # ESM
    "esm_detection_range",
    "esm_detection_probability",
    # 批量
    "evaluate_sensor_detection",
]
