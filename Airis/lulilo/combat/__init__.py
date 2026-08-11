"""engine.combat 层：武器参数 / 毁伤解析 / 碰撞检测 / 开火链—— Phase 3。

模块：
    weapons.py     武器/弹药 Pydantic 模型 + 注册表 + 散布解算（WeaponSystem）
    collision.py   Numba 碰撞检测（统一胶囊体 vs AABB）+ 空间哈希（CollisionDetector）
    damage.py      穿透/斜距/跳弹/后效毁伤（Numba 内核 + DamageResolver）
    engagement.py  开火链交战管理器（FiringControl：散布→发射→碰撞→毁伤→消耗→事件）

开火链（docs/phase3-combat-design.md §七）：
    WeaponSystem（统计层散布）→ FiringControl.fire（抛射物发射）
    → CollisionDetector（物理层碰撞）→ DamageResolver（穿透/毁伤）
    → AmmoManager（弹药消耗）→ EventBus（命中事件，叙事层消费）
"""

from Airis.lulilo.combat.collision import CollisionDetector, CollisionHit, FormationResolver
from Airis.lulilo.combat.damage import (
    ARMOR_EFFECTIVENESS,
    ArmorType,
    DamageResolver,
)
from Airis.lulilo.combat.engagement import FireRecord, FiringControl, UnitCombatConfig
from Airis.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    FireSolution,
    GuidanceType,
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
    "GuidanceType",
    "UnitCombatConfig",
    "WeaponCategory",
    "WeaponConfig",
    "WeaponDefinition",
    "WeaponRegistry",
    "WeaponSystem",
]
