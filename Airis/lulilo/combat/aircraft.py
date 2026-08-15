"""机型静态参数（Phase 7，docs/phase7-air-design.md §1.3）。

YAML + Pydantic 加载（仿 stochastic-warfare 数据组织，AGENTS.md §10.4）：
    定义文件是纯数据，引擎代码定义行为。本模型只声明字段与默认值，
    气动/飞控/AI 行为见 movement/aerodynamics.py / flight_control.py / simulation/air_ai.py。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from Airis.lulilo.entities.pools import AircraftRole


class AircraftConfig(BaseModel):
    """机型静态参数（Phase 7，docs/phase7-air-design.md §1.3 + phase7-air-pipeline.md §三）。

    字段全部带默认值——新机型 YAML 只需覆盖差异项，缺省即通用战斗机基线。
    气动内核消费：wing_area/cl_max/cd0/k_ind/max_g/v_min/v_max/thrust/mass。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: AircraftRole = AircraftRole.FIGHTER

    # ---- 气动（docs/phase7-air-pipeline.md §3.1）----
    wing_area_m2: float = Field(default=30.0, gt=0.0, description="机翼面积 (m²)")
    cl_max: float = Field(default=1.6, gt=0.0, description="最大升力系数")
    cd0: float = Field(default=0.02, ge=0.0, description="零升阻力系数")
    k_ind: float = Field(default=0.05, ge=0.0, description="诱导阻力因子")
    max_alpha_deg: float = Field(default=20.0, gt=0.0, description="最大迎角 (deg)")

    # ---- 飞行包线 ----
    v_min_m_s: float = Field(default=80.0, gt=0.0, description="失速速度 (m/s)")
    v_max_m_s: float = Field(default=750.0, gt=0.0, description="最大平飞速度 (m/s)")
    max_g: float = Field(default=9.0, gt=0.0, description="过载限制 (g)")
    climb_rate_m_s: float = Field(default=150.0, gt=0.0, description="最大爬升率 (m/s)")
    turn_rate_max_deg_s: float = Field(default=18.0, gt=0.0, description="最大持续转弯率 (deg/s)")
    thrust_max: float = Field(default=100000.0, gt=0.0, description="最大推力 (N)")
    mass_kg: float = Field(default=15000.0, gt=0.0, description="质量 (kg)")

    # ---- 载荷/燃油 ----
    aam_capacity: int = Field(default=8, ge=0, description="AAM 挂载数")
    agm_capacity: int = Field(default=0, ge=0, description="AGM/炸弹挂载数")
    fuel_capacity_s: float = Field(default=5400.0, gt=0.0, description="满油续航 (s，90min)")
    bingo_fuel: float = Field(default=0.3, ge=0.0, le=1.0, description="强制返航阈值 0~1")
    chaff_capacity: int = Field(default=30, ge=0, description="箔条数")
    flare_capacity: int = Field(default=30, ge=0, description="热焰弹数")

    # ---- 传感器（docs/phase7-air-pipeline.md §四）----
    radar_power_db: float = Field(default=70.0, description="机载雷达功率 (dBm，EmitterPool kind=0)")
    radar_freq_ghz: float = Field(default=10.0, gt=0.0, description="机载雷达频率 (GHz)")
    rwr_sensitivity: float = Field(default=1.0, gt=0.0, description="RWR 检测灵敏度")

    # ---- 毁伤 ----
    health_max: float = Field(default=100.0, gt=0.0, description="单血量上限")
