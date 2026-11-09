"""engine.combat 层：武器参数 / 毁伤解析 / 碰撞检测 / 开火链—— Phase 3。

模块：
    weapons.py     武器/弹药 Pydantic 模型 + 注册表 + 散布解算（WeaponSystem）
    collision.py   Numba 碰撞检测（统一胶囊体 vs AABB）+ 空间哈希（CollisionDetector）
    damage.py      穿透/斜距/跳弹/后效毁伤（Numba 内核 + DamageResolver）
    engagement.py  开火链交战管理器（FiringControl：散布→发射→碰撞→毁伤→消耗→事件）
    guidance.py    导弹控制律注册表 + PN/PP 公共求解器 + 内置控制回调（Phase 6b）
    sfm.py        末敏弹双模扫描（scan_armor_detect + SfmScanner，Phase 6b 步骤 8）
    sal.py        激光制导炸弹（SALSystem：吊舱 + 光斑存活，Phase 6b 步骤 9）

开火链（docs/phase3-combat-design.md §七）：
    WeaponSystem（统计层散布）→ FiringControl.fire（抛射物发射）
    → CollisionDetector（物理层碰撞）→ DamageResolver（穿透/毁伤）
    → AmmoManager（弹药消耗）→ EventBus（命中事件，叙事层消费）

制导管线（docs/phase6-guidance-pipeline.md，Phase 6b）：
    MissileControlRegistry（guid_law 函数索引）→ 控制回调（stage/override 状态机输入）
    → integrate_guided_projectile_movement（按 stage 分派物理 + 回调算总加速度）
"""

from Airia.lulilo.combat.collision import CollisionDetector, CollisionHit, FormationResolver
from Airia.lulilo.combat.damage import (
    ARMOR_EFFECTIVENESS,
    ArmorType,
    DamageResolver,
)
from Airia.lulilo.combat.engagement import FireRecord, FiringControl, UnitCombatConfig
from Airia.lulilo.combat.guidance import (
    GuidanceStage,
    ManualOverride,
    MissileControlRegistry,
    control_boost_coast_pn,
    control_boost_coast_pp,
    control_dual_pulse_pn,
    control_sfm_ballistic,
    guidance_proportional_nav,
    guidance_pure_pursuit,
)
from Airia.lulilo.combat.sal import SALSystem
from Airia.lulilo.combat.sfm import SfmScanner
from Airia.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    FireSolution,
    GuidanceType,
    SeekerType,
    WeaponCategory,
    WeaponConfig,
    WeaponDefinition,
    WeaponRegistry,
    WeaponSystem,
)

__all__ = [
    "ARMOR_EFFECTIVENESS",
    "AmmoConfig",
    "AmmoType",
    "ArmorType",
    "CollisionDetector",
    "CollisionHit",
    "FormationResolver",
    "DamageResolver",
    "FireRecord",
    "FireSolution",
    "FiringControl",
    "GuidanceStage",
    "GuidanceType",
    "ManualOverride",
    "MissileControlRegistry",
    "SALSystem",
    "SeekerType",
    "SfmScanner",
    "UnitCombatConfig",
    "WeaponCategory",
    "WeaponConfig",
    "WeaponDefinition",
    "WeaponRegistry",
    "WeaponSystem",
    "control_boost_coast_pn",
    "control_boost_coast_pp",
    "control_dual_pulse_pn",
    "control_sfm_ballistic",
    "guidance_proportional_nav",
    "guidance_pure_pursuit",
]
