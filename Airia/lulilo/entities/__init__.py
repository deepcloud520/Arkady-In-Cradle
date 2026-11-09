"""engine.entities 层：SoA 对象池（AGENTS.md 3.1）。"""

from Airia.lulilo.entities.side import (
    SIDE_NAME,
    Side,
    SideNameMapper,
    allegiance,
    hostile,
    is_neutral,
    side_name,
)
from Airia.lulilo.entities.pools import (
    MAX_ARMOR,
    MAX_BOXES,
    MAX_EMITTERS,
    MAX_FORMATIONS,
    MAX_PROJ,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
    ProjectileType,
)

__all__ = [
    "SIDE_NAME",
    "Side",
    "SideNameMapper",
    "allegiance",
    "hostile",
    "is_neutral",
    "side_name",
    "MAX_ARMOR",
    "MAX_BOXES",
    "MAX_EMITTERS",
    "MAX_FORMATIONS",
    "MAX_PROJ",
    "ArmorPool",
    "BoxPool",
    "EmitterBand",
    "EmitterPool",
    "FormationAIPool",
    "PartType",
    "ProjectilePool",
    "ProjectileType",
]
