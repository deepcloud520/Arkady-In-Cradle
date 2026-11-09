"""武器参数模型与开火解算（AGENTS.md Phase 3：combat/weapons.py）。

职责（对应 docs/phase3-combat-design.md §八「参数来源映射」）：
- 武器/弹药定义（Pydantic 配置模型，YAML/CSV → 引擎）
- 武器注册表（WeaponRegistry：按 id 查询武器与兼容弹药）
- 开火链接口（WeaponSystem：给定目标与射距 → 散布半径 sigma + 开火解算）

设计要点（AGENTS.md 10.5 对比表）：
    我们的路线 = CSV → WeaponConfig（Pydantic），弹药独立或内嵌。
    骨架先提供 from_dict / from_yaml 两条加载路径，示例内置 M1A2 主炮 vs T-90。

散布内核（design doc §二）：sigma_m = base_accuracy_mrad * 0.001 * range_m，
再乘修正因子（乘员技能/目标运动/射手运动/能见度/姿态/武器状况/位置不确定/火控稳定）。
本模块只负责算出 sigma（统计层），物理命中交给 collision.py（物理层）。
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.entities.pools import ProjectileType

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class WeaponCategory(IntEnum):
    """武器分类。

    ⚠️ 升级预留：编号 5~10 已预留给 AUTOCANNON/SMALL_ARMS/GRENADE_LAUNCHER/
    MORTAR/AA_GUN/RECOILLESS（见 docs/ammunition-expansion-design.md §四）。
    本阶段不新增，避免破坏现有逻辑。
    """
    MAIN_GUN = 0     # 主炮
    SECONDARY = 1    # 副武器
    MG = 2           # 机枪
    ATGM = 3         # 反坦克导弹
    HOWITZER = 4     # 榴弹炮


class AmmoType(IntEnum):
    """弹药类型（决定穿透模型分支，design doc §四）。

    ⚠️ 升级预留：编号 6~34 已预留给 HE_FRAG/PELE/THERMOBARIC/APCR/APCBC/
    HEAT_MP/AMP/CANISTER/炮射导弹/步兵弹药/防空可编程弹（AHEAD/3P/VT 等）。
    完整规划见 docs/ammunition-expansion-design.md §一。

    Phase 6c 步骤③（docs/phase6-ai-pipeline.md §十一）**激活**部分预留值
    （用预留值、不重排，旧存档兼容）：HE_FRAG=6 / PELE=7 / APCR=9 / SAPHEI=34。
    """
    APFSDS = 0   # 尾翼稳定脱壳穿甲弹（动能）
    AP = 1       # 普通穿甲弹（动能）
    HEAT = 2     # 破甲弹（化学能）
    HE = 3       # 高爆弹
    ATGM = 4     # 反坦克导弹（化学能）
    HESH = 5     # 碎甲弹
    HE_FRAG = 6  # 高爆破片弹（球形爆破 + 破片锥；Phase 6c 激活，预留值）
    PELE = 7     # 惰性侵彻体（细锥射流 + 穿后碎裂；Phase 6c 激活，预留值）
    APCR = 9     # 硬芯穿甲弹（细锥射流，复用 DeMarre；Phase 6c 激活，预留值）
    SAPHEI = 34  # 半穿甲高爆燃烧弹（细锥射流 + 延迟引爆；Phase 6c 激活，预留值）
    SENSOR_FUZED = 35  # 末敏弹（攻顶，Phase 6b；追加不重排既有值，旧存档兼容）


class GuidanceType(IntEnum):
    """制导类型（决定抛射物是直射还是追踪）。"""
    NONE = 0             # 无制导（直射弹道）
    WIRE = 1             # 线导
    SAL = 2              # 半主动激光
    FIRE_AND_FORGET = 3  # 发射后不管


class SeekerType(IntEnum):
    """导引头类型（Phase 6b，docs/phase6-guidance-pipeline.md §6.5）。

    辐射特性表（决定「能否被反辐射跟踪」）：
        ACTIVE_RADAR     导引头强辐射 → 可被 ESM 检测 / ARM 跟踪（开导引头=暴露自己）
        SEMI_ACTIVE_LASER 不辐射（依赖外部激光照射器；照射器可被激光告警）
        IR_IMAGING / TV_EO / SENSOR_FUZED 全被动，无法被反辐射跟踪
        PASSIVE_RF       反辐射 ARM 本体（自己是检测方）
    """
    NONE = 0               # 无导引头（无制导，guid_law=-1）
    ACTIVE_RADAR = 1       # 主动雷达导引头：持续辐射，强辐射源（kind=1）
    SEMI_ACTIVE_LASER = 2  # 半主动激光：依赖外部激光照射器光斑（kind=4）
    IR_IMAGING = 3         # 红外成像：完全被动，检测红外 emitter（band=IR）
    PASSIVE_RF = 4         # 被动雷达（反辐射 ARM）：追踪敌方辐射源（band=RF）
    TV_EO = 5              # 电视/光电：被动（简化并入红外）
    SENSOR_FUZED = 6       # 毫米波+红外扫描（末敏弹，design §13.3；追加不重排）


# ---------------------------------------------------------------------------
# Pydantic 配置模型
# ---------------------------------------------------------------------------


class AmmoConfig(BaseModel):
    """弹药定义（design doc §八：pen_mm_rha / is_heat / drag_coefficient）。

    ⚠️ 升级预留：未来扩展字段（Lanz-Odermatt 参数 / 爆炸破片参数 / 引信可编程
    参数）规划见 docs/ammunition-expansion-design.md §三。

    Phase 6b（docs/phase6-guidance-pipeline.md §2.4）追加制导/导引头/末敏弹字段——
    全部带默认值，旧 YAML / 旧测试构造无需改动即通过校验（`extra=forbid` 下已一次加齐）。
    """

    ammo_id: str
    name: str = ""
    ammo_type: AmmoType = AmmoType.APFSDS
    pen_mm_rha: float = Field(default=0.0, ge=0.0, description="0m 标称穿深 (mm RHA)")
    muzzle_velocity: float = Field(default=0.0, ge=0.0, description="初速 (m/s)")
    projectile_type: ProjectileType = ProjectileType.BULLET
    guidance: GuidanceType = GuidanceType.NONE
    drag_coefficient: float = Field(default=0.15, ge=0.0, le=1.0, description="弹芯阻力")
    is_heat: bool = False  # 化学能弹（穿深不随距离衰减）

    # ---- 导弹控制律（docs §五/§12.1；推进剖面 × 制导律合一）----
    control_law_id: str = ""              # 导弹控制律注册表 id（guidance != NONE 时必填）
    nav_gain: float = 3.0                 # 制导增益 N（传给公共制导律求解器，per-ammo）
    seeker: SeekerType = SeekerType.NONE
    emitter_power_db: float = 60.0        # ACTIVE_RADAR 辐射功率
    # ---- 导引头性能（全 YAML 可配，体现导弹优劣，§6.5）----
    # —— emitter 方向性（发射侧）：f↑/D↑ → 主瓣窄 → 能量集中、旁瓣低 ——
    seeker_freq_ghz: float = 10.0         # 发射频率（f↑ → 主瓣窄、能量集中、旁瓣低）
    antenna_diameter_m: float = 0.3       # 天线口径（θ≈70°·λ/D，与 f 共同决定主瓣宽度）
    # —— seeker 视场（接收侧）：视场是接收方属性，直接配置，不走 GHz ——
    search_fov_deg: float = 30.0          # seeker 搜索视场（宽，未锁定）
    track_fov_deg: float = 5.0            # seeker 跟踪视场（窄，锁定后抗跳变）
    filter_quality: float = 0.7           # 滤波能力 0~1（坐标估计精度 + 抗干扰强度）
    # ---- ARM 记忆模式与末端引导（§12.1）----
    memory_mode: bool = True
    terminal_seeker: SeekerType = SeekerType.NONE
    terminal_range_m: float = 0.0         # 0=禁用
    ins_drift_m_s: float = 0.0            # 惯性漂移（本阶段 0，Phase 7 启用）
    # ---- 激光吊舱（SAL，§7.5）----
    beam_divergence_deg: float = 1.0      # 吊舱激光束发散角（决定光斑大小，本阶段仅描述）
    # ---- 通信（2026-08-14；对齐单位模型，人在回路门控，非专门数据链系统）----
    comm_threshold: float = 0.5           # 人在回路命令可达的通信门限（guid_comm ≥ 此值）
    # 注：guid_confidence 属友军孪生层（Phase 8），不存 AmmoConfig；
    #     制导「丢失」由 detection 决定（§5.2/§6.5）
    # ---- 末敏弹（§13.3）----
    efp_pen_mm: float = 0.0
    scan_altitude_m: float = 150.0
    scan_radius_m: float = 80.0
    scan_rps: float = 6.0                 # 预留（本阶段扫描全向，不消费）
    scan_duration_s: float = 5.0
    top_armor_factor: float = 0.3
    mmw_emitter_power_db: float = 40.0
    # ---- 双模扫描阈值（§2.3/§7.2，抗诱饵：双源都达标才引爆）----
    mmw_threshold: float = 5.0            # 毫米波反射阈值（ArmorPool.rcs）
    ir_threshold: float = 0.1             # 红外辐射阈值（EmitterPool 红外 emitter 的 cur_power，含残热）
    # ---- Phase 6c 步骤③ 爆炸/破片参数（docs/phase6-ai-pipeline.md §十一.2）----
    # 全部带默认值，旧 YAML / 旧测试构造无需改动即通过校验（extra=forbid 下一次加齐）
    blast_radius: float = 0.0             # 爆炸半径 (m)；>0 时命中后提交球形爆破事件（§七.3）
    blast_dmg: float = 0.0                # 球形爆破中心毁伤量（半径内线性衰减到 0）
    frag_dmg: float = 0.0                 # 单破片伤害；>0 时提交破片锥事件
    frag_count: int = 0                   # 破片数（破片锥参数；本阶段确定性结算）
    fuse_delay: float = 0.0               # 引信延迟 (s)：SAPHEI 穿后延迟引爆（预留，本阶段 0）
    # ---- Phase 7 空战（docs/phase7-air-pipeline.md §7.3）----
    prox_fuse_radius: float = 0.0         # 近炸引信半径 (m)：>0 时 AAM/SAM 与飞机相对距离 < 半径 → 近炸起爆

    # ---- 穿深算法分派（docs/penetration-algorithms-design.md §7.1）----
    # 全部带默认值，旧 YAML / 旧测试构造无需改动即通过校验（extra=forbid 下一次加齐）
    penetration_model: int = -1           # PenetrationModel 整数值；-1=由 ammo_type 自动派生
    velocity_alpha: float = 0.0           # 每米速度衰减系数 (1/m)；0=不衰减（旧行为）
    penetrator_length_mm: float = 0.0     # 弹芯初始长度 L (mm)；>0 且 Yp>0 → APFSDS 走 A-T
    penetrator_density: float = 17500.0   # ρp (kg/m³) WHA=17500 / DU=18600
    penetrator_strength_pa: float = 0.0   # Yp 杆动态流应力 (Pa)；Tate 1986: 1.7·σyp
    lo_beta: float = 0.67                 # 简化 Lanz-Odermatt β（WHA=0.67 / DU=0.72）
    lo_k: float = 3.0                     # 简化 Lanz-Odermatt k（RHA）

    model_config = {"extra": "forbid"}


class WeaponConfig(BaseModel):
    """武器定义（AGENTS.md 10.5：caliber/velocity/accuracy/compatible_ammo）。"""

    weapon_id: str
    name: str = ""
    category: WeaponCategory = WeaponCategory.MAIN_GUN
    caliber_mm: float = Field(default=120.0, gt=0.0)
    muzzle_velocity: float = Field(default=1500.0, gt=0.0, description="初速 (m/s)")
    base_accuracy_mrad: float = Field(
        default=0.2, gt=0.0, description="武器固有精度 1σ (mrad)，M256≈0.2 / 2A46M≈1.0"
    )
    drag_coefficient: float = Field(default=0.15, ge=0.0, le=1.0)
    fire_rate_rpm: float = Field(default=6.0, gt=0.0, description="射速 (发/分钟)")
    max_range_m: float = Field(default=4000.0, gt=0.0)
    fire_control_stabilization: float = Field(
        default=0.8, ge=0.0, le=1.0, description="火控稳定系数（行进间射击惩罚缩放）"
    )
    compatible_ammo: list[str] = Field(default_factory=list, description="兼容弹药 ammo_id 列表")

    model_config = {"extra": "forbid"}


class WeaponDefinition(BaseModel):
    """武器 + 所选弹药 的组合（一次开火解算的静态参数快照）。"""

    weapon: WeaponConfig
    ammo: AmmoConfig

    @property
    def pen_mm_rha(self) -> float:
        """当前弹药的 0m 穿深。"""
        return self.ammo.pen_mm_rha

    @property
    def is_heat(self) -> bool:
        return self.ammo.is_heat


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


class WeaponRegistry:
    """武器/弹药注册表（数据驱动：dict → Pydantic → 引擎）。

    骨架提供：
        - register / register_ammo / get / get_ammo
        - from_dict / from_yaml（YAML 加载管线占位，结构同 Reference CSV 思路）
        - 内置示例：M1A2 的 M256 主炮 + 兼容弹药、T-90 的 2A46M 主炮 + 兼容弹药
    """

    def __init__(self) -> None:
        self._weapons: dict[str, WeaponConfig] = {}
        self._ammo: dict[str, AmmoConfig] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, weapon: WeaponConfig | dict[str, Any]) -> None:
        """注册武器（接受 WeaponConfig 或可解析为它的 dict）。"""
        if isinstance(weapon, dict):
            weapon = WeaponConfig.model_validate(weapon)
        self._weapons[weapon.weapon_id] = weapon

    def register_ammo(self, ammo: AmmoConfig | dict[str, Any]) -> None:
        """注册弹药（接受 AmmoConfig 或可解析为它的 dict）。"""
        if isinstance(ammo, dict):
            ammo = AmmoConfig.model_validate(ammo)
        self._ammo[ammo.ammo_id] = ammo

    def register_definition(self, weapon: WeaponConfig, ammo: list[AmmoConfig]) -> None:
        """注册武器及其兼容弹药（批量）。"""
        self.register(weapon)
        for a in ammo:
            self.register_ammo(a)

    # ------------------------------------------------------------------
    # 索引查询（弹丸元数据下沉 SoA 用，2026-08-11）
    #
    # ProjectilePool.weapon/ammo 存「注册表索引」而非字符串 id，
    # 这里提供 id ↔ 索引 的往返查询。索引 = 注册顺序（dict 保持插入序）。
    # ------------------------------------------------------------------

    def weapon_index(self, weapon_id: str) -> int:
        """武器 id → 注册表索引（未注册返回 -1）。"""
        try:
            return list(self._weapons.keys()).index(weapon_id)
        except ValueError:
            return -1

    def ammo_index(self, ammo_id: str) -> int:
        """弹药 id → 注册表索引（未注册返回 -1）。"""
        try:
            return list(self._ammo.keys()).index(ammo_id)
        except ValueError:
            return -1

    def weapon_at(self, index: int) -> WeaponConfig | None:
        """注册表索引 → 武器（越界/无效返回 None）。"""
        if index < 0:
            return None
        try:
            return self._weapons[list(self._weapons.keys())[index]]
        except IndexError:
            return None

    def ammo_at(self, index: int) -> AmmoConfig | None:
        """注册表索引 → 弹药（越界/无效返回 None）。"""
        if index < 0:
            return None
        try:
            return self._ammo[list(self._ammo.keys())[index]]
        except IndexError:
            return None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, weapon_id: str) -> WeaponConfig:
        try:
            return self._weapons[weapon_id]
        except KeyError:
            raise KeyError(f"未知武器: {weapon_id}") from None

    def get_ammo(self, ammo_id: str) -> AmmoConfig:
        try:
            return self._ammo[ammo_id]
        except KeyError:
            raise KeyError(f"未知弹药: {ammo_id}") from None

    def get_definition(self, weapon_id: str, ammo_id: str | None = None) -> WeaponDefinition:
        """组合一个 WeaponDefinition。

        Args:
            weapon_id: 武器 id。
            ammo_id: 弹药 id；缺省取武器第一枚兼容弹药。

        Returns:
            武器 + 弹药 的组合快照。
        """
        weapon = self.get(weapon_id)
        if ammo_id is None:
            if not weapon.compatible_ammo:
                raise ValueError(f"武器 {weapon_id} 未声明兼容弹药")
            ammo_id = weapon.compatible_ammo[0]
        ammo = self.get_ammo(ammo_id)
        if ammo_id not in weapon.compatible_ammo:
            logger.warning("弹药 %s 不在 %s 的兼容列表，仍按配置使用", ammo_id, weapon_id)
        return WeaponDefinition(weapon=weapon, ammo=ammo)

    def all_weapons(self) -> list[WeaponConfig]:
        return list(self._weapons.values())

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """从嵌套 dict 构建注册表。

        data 结构:
            {
                "weapons": [{...WeaponConfig}, ...],
                "ammo": [{...AmmoConfig}, ...],
            }
        """
        reg = cls()
        for w in data.get("weapons", []):
            reg.register(w)
        for a in data.get("ammo", []):
            reg.register_ammo(a)
        return reg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """从 YAML 文件构建注册表（加载管线占位，Phase 3 后续接 CSV）。"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        reg = cls.from_dict(data)
        logger.info("从 %s 加载武器注册表（%d 武器 / %d 弹药）",
                    path.name, len(reg._weapons), len(reg._ammo))
        return reg

    # ------------------------------------------------------------------
    # 内置示例数据（M1A2 vs T-90，design doc §八）
    # ------------------------------------------------------------------

    @classmethod
    def with_builtin(cls) -> Self:
        """内置示例注册表：M256（M1A2）与 2A46M（T-90）主炮 + 兼容弹药。"""
        reg = cls()
        reg.register_definition(
            WeaponConfig(
                weapon_id="m256_120mm",
                name="M256 120mm 滑膛炮",
                category=WeaponCategory.MAIN_GUN,
                caliber_mm=120.0,
                muzzle_velocity=1570.0,
                base_accuracy_mrad=0.2,       # M1A2 先进 FCS（design doc §八）
                drag_coefficient=0.15,
                fire_rate_rpm=6.0,
                max_range_m=4000.0,
                fire_control_stabilization=0.9,  # 稳定器好 → 行进间惩罚减半
                compatible_ammo=["m829a3_apfsds", "m830a1_heat"],
            ),
            [
                AmmoConfig(
                    ammo_id="m829a3_apfsds",
                    name="M829A3 APFSDS",
                    ammo_type=AmmoType.APFSDS,
                    pen_mm_rha=700.0,
                    muzzle_velocity=1555.0,
                    projectile_type=ProjectileType.BULLET,
                    drag_coefficient=0.15,
                    # A-T 长杆参数（docs/penetration-algorithms-design.md §5.3/§7.1）
                    penetrator_length_mm=780.0,      # 弹芯长度 (mm)
                    penetrator_density=17500.0,      # WHA
                    penetrator_strength_pa=3.0e9,    # Yp = 1.7·σyp（WHA BHN≈420）
                    velocity_alpha=3.5e-5,           # ~50 m/s/km 减速
                ),
                AmmoConfig(
                    ammo_id="m830a1_heat",
                    name="M830A1 HEAT-MP",
                    ammo_type=AmmoType.HEAT,
                    pen_mm_rha=600.0,
                    muzzle_velocity=1140.0,
                    projectile_type=ProjectileType.MISSILE,
                    is_heat=True,
                ),
            ],
        )
        # ---- Phase 6c 步骤⑤：炮兵（docs/phase6-ai-pipeline.md §六）----
        # M109A6 155mm 自行榴弹炮：曲射弹药（HE_FRAG 球形爆破 + 破片锥，经提交池结算）。
        # max_range_m=15000 受 PROJECTILE_MAX_AGE=20s 约束（T(15km)≈18.3s < 20s）；
        # 超出 WORLD_BOUND 的飞行由 _recycle_stale 对 indirect 弹丸豁免（弹道必落点）。
        reg.register_definition(
            WeaponConfig(
                weapon_id="m109_155mm",
                name="M109A6 155mm 自行榴弹炮",
                category=WeaponCategory.HOWITZER,
                caliber_mm=155.0,
                muzzle_velocity=827.0,
                base_accuracy_mrad=0.8,       # 间瞄射击散布（mrad，1σ）
                drag_coefficient=0.35,        # 炮弹阻力系数
                fire_rate_rpm=4.0,
                max_range_m=15000.0,
                fire_control_stabilization=0.6,
                compatible_ammo=["m107_155mm_he_frag"],
            ),
            [
                AmmoConfig(
                    ammo_id="m107_155mm_he_frag",
                    name="M107 155mm HE-FRAG",
                    ammo_type=AmmoType.HE_FRAG,
                    pen_mm_rha=60.0,          # HE 对轻甲/破片侵彻（主杀伤 = 爆破）
                    muzzle_velocity=827.0,
                    projectile_type=ProjectileType.MISSILE,
                    is_heat=True,             # 化学能语义：穿深不随距离衰减
                    drag_coefficient=0.35,
                    # 落点毁伤（docs §七.1：球形爆破 + 破片锥，经提交池结算）
                    blast_radius=25.0,        # 爆炸半径 (m)
                    blast_dmg=150.0,          # 球形爆破中心毁伤量
                    frag_dmg=25.0,            # 单破片伤害
                    frag_count=120,           # 破片数
                ),
            ],
        )
        reg.register_definition(
            WeaponConfig(
                weapon_id="2a46m_125mm",
                name="2A46M 125mm 滑膛炮",
                category=WeaponCategory.MAIN_GUN,
                caliber_mm=125.0,
                muzzle_velocity=1700.0,
                base_accuracy_mrad=1.0,       # T-90 火控较逊（design doc §八）
                drag_coefficient=0.18,
                fire_rate_rpm=7.0,
                max_range_m=3500.0,
                fire_control_stabilization=0.5,  # 稳定器差 → 行进间惩罚大
                compatible_ammo=["3bm42_apfsds", "3bk18_heat"],
            ),
            [
                AmmoConfig(
                    ammo_id="3bm42_apfsds",
                    name="3BM42 APFSDS",
                    ammo_type=AmmoType.APFSDS,
                    pen_mm_rha=650.0,
                    muzzle_velocity=1700.0,
                    projectile_type=ProjectileType.BULLET,
                    drag_coefficient=0.18,
                    # A-T 长杆参数（docs/penetration-algorithms-design.md §5.3/§7.1）
                    penetrator_length_mm=660.0,      # 弹芯长度 (mm)
                    penetrator_density=17500.0,      # WHA
                    penetrator_strength_pa=3.0e9,    # Yp = 1.7·σyp
                    velocity_alpha=3.0e-5,           # ~45 m/s/km 减速
                ),
                AmmoConfig(
                    ammo_id="3bk18_heat",
                    name="3BK18 HEAT",
                    ammo_type=AmmoType.HEAT,
                    pen_mm_rha=550.0,
                    muzzle_velocity=905.0,
                    projectile_type=ProjectileType.MISSILE,
                    is_heat=True,
                ),
            ],
        )
        # ---- Phase 演示：T-14 机炮车（2A90 57mm 自动炮，低穿深高射速）----
        reg.register_definition(
            WeaponConfig(
                weapon_id="2a90_57mm",
                name="2A90 57mm 自动炮（T-14 机炮车）",
                category=WeaponCategory.MAIN_GUN,
                caliber_mm=57.0,
                muzzle_velocity=1000.0,
                base_accuracy_mrad=0.4,       # 自动炮火控良好
                drag_coefficient=0.2,
                fire_rate_rpm=120.0,          # 自动装填高射速
                max_range_m=6000.0,
                fire_control_stabilization=0.8,
                compatible_ammo=["57mm_apfsds"],
            ),
            [
                AmmoConfig(
                    ammo_id="57mm_apfsds",
                    name="57mm APFSDS",
                    ammo_type=AmmoType.APFSDS,
                    pen_mm_rha=300.0,         # 演示调参：能穿 MBT 炮塔/履带（薄部位），
                                              # 正面（复合 375 等效）仍不穿 → 部位损伤可见
                    muzzle_velocity=1000.0,
                    projectile_type=ProjectileType.BULLET,
                    drag_coefficient=0.2,
                ),
            ],
        )
        return reg


# ---------------------------------------------------------------------------
# 开火解算（散布内核的"统计层"入口）
# ---------------------------------------------------------------------------

# 姿态修正（design doc §二 表格）：依托掩体的暴露面积修正
POSTURE_HIT_FACTOR: dict[str, float] = {
    "DUG_IN": 0.6,
    "FORTIFIED": 0.4,
    "DEFENSIVE": 0.85,
    "MOVING": 1.0,
}


class FireSolution(BaseModel):
    """一次开火的解算结果（统计层输出，交给物理层发射）。"""

    weapon_id: str
    ammo_id: str
    range_m: float
    sigma_m: float                 # 最终 1σ 散布半径（含全部修正）
    aim_x: float = 0.0
    aim_y: float = 0.0
    aim_z: float = 0.0
    muzzle_velocity: float = 0.0


class WeaponSystem:
    """开火链的武器侧（design doc §七 开火链第 2~3 步）。

    compute_sigma → 把散布半径交给抛射物生成（collision 物理层决定落点）。
    """

    def __init__(self, registry: WeaponRegistry | None = None) -> None:
        self.registry = registry or WeaponRegistry.with_builtin()

    def compute_sigma_m(
        self,
        range_m: float,
        base_accuracy_mrad: float,
        *,
        crew_skill: float = 0.7,
        target_speed: float = 0.0,
        shooter_speed: float = 0.0,
        visibility: float = 1.0,
        posture: str = "MOVING",
        weapon_condition: float = 1.0,
        uncertainty_m: float = 0.0,
        fire_control_stabilization: float = 0.5,
    ) -> float:
        """计算最终 1σ 散布半径（design doc §二 全部修正因子）。

        sigma_m = base_accuracy_mrad * 0.001 * range_m × Π(修正因子)

        Args:
            range_m: 射距 (m)。
            base_accuracy_mrad: 武器固有精度 (mrad)。
            crew_skill: 乘员技能 [0,1]，M1A2≈0.9 / T-90≈0.7。
            target_speed: 目标运动速度 (m/s)。
            shooter_speed: 射手运动速度 (m/s)。
            visibility: 能见度 [0,1]。
            posture: 姿态键（见 POSTURE_HIT_FACTOR）。
            weapon_condition: 武器状况 [0,1]。
            uncertainty_m: 目标位置不确定度 (m)。
            fire_control_stabilization: 火控稳定 [0,1]。

        Returns:
            最终 sigma（米）。
        """
        sigma = base_accuracy_mrad * 0.001 * range_m

        # 因子语义（design doc §2.2/§2.3）：各因子为命中概率 P(hit) 的乘子，
        # 因子越大 → P 越高 → 散布应越小，故注入 sigma 时取倒数（除法）。

        # 乘员技能：0.5 + 0.3 * skill
        sigma /= 0.5 + 0.3 * crew_skill
        # 目标运动：1.0 - 0.15 * (speed/10)，下限 0.5
        sigma /= max(0.5, 1.0 - 0.15 * (target_speed / 10.0))
        # 射手运动：1.0 - 0.25 * (speed/10) × (1 - 稳定系数)，下限 0.3
        motion_penalty = 1.0 - 0.25 * (shooter_speed / 10.0) * (1.0 - fire_control_stabilization)
        sigma /= max(0.3, motion_penalty)
        # 能见度：0.3 + 0.7 * vis
        sigma /= 0.3 + 0.7 * visibility
        # 姿态
        sigma /= POSTURE_HIT_FACTOR.get(posture, 1.0)
        # 武器状况：0.5 + 0.5 * condition
        sigma /= 0.5 + 0.5 * weapon_condition
        # 位置不确定：1.0 - 0.01 * uncertainty，下限 0.3
        sigma /= max(0.3, 1.0 - 0.01 * uncertainty_m)

        return float(sigma)

    def solve(self, weapon_id: str, ammo_id: str | None, range_m: float, **mods: Any) -> FireSolution:
        """解算一次开火（统计层输出 FireSolution）。

        Args:
            weapon_id: 武器 id。
            ammo_id: 弹药 id；None 取默认。
            range_m: 射距 (m)。
            **mods: 透传给 compute_sigma_m 的修正因子。

        Returns:
            FireSolution（sigma + 瞄准点，物理层据此生成抛射物）。
        """
        definition = self.registry.get_definition(weapon_id, ammo_id)
        sigma = self.compute_sigma_m(
            range_m,
            definition.weapon.base_accuracy_mrad,
            fire_control_stabilization=definition.weapon.fire_control_stabilization,
            **mods,
        )
        return FireSolution(
            weapon_id=weapon_id,
            ammo_id=definition.ammo.ammo_id,
            range_m=range_m,
            sigma_m=sigma,
            muzzle_velocity=definition.ammo.muzzle_velocity,
        )
