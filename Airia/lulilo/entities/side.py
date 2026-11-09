"""Side：绝对阵营（世界模型客观事实）+ 关系派生（敌友中立）。

设计约定（2026-08-15，取代旧 ArmorPool.team 的相对友/敌/中立语义）：

- ArmorPool.side / FormationAIPool.side 只存**客观绝对阵营**（这是谁的部队），
  不存相对敌友；「敌友中立」是**关系**，由 allegiance() / hostile() 现算，
  不入池、不入存档。
- 同一阵营内不同指挥链（如美1装 vs 英旅）由 formation_id（已有）/
  未来 coalition_id 表达，side 不承担该职责。
- Phase 8 换 networkx 联盟图时，hostile() / allegiance() 签名不变，
  只换实现（内部查 DataLinkNetwork 联盟关系）。
"""

from __future__ import annotations

from enum import IntEnum


class Side(IntEnum):
    """绝对阵营 id（0~255，SoA uint8 存储）。"""

    NATO = 0      # 北约 / 蓝方
    PACT = 1      # 华约（苏联 / 红军）
    NEUTRAL = 2   # 中立 / 平民
    # 3..255 预留（Phase 8 多阵营：PRC、IRAN……）


# 阵营 id → 显示名（SideNameMapper 默认表；渲染/叙事层消费，引擎核心不依赖）
SIDE_NAME: dict[Side, str] = {
    Side.NATO: "北约",
    Side.PACT: "华约",
    Side.NEUTRAL: "中立",
}

# 阵营 id → 默认单位型号（SideModelMapper 默认表；渲染/叙事层消费，引擎核心不依赖。
# 唯一数据源：HUD / DetailPanel / 叙事 recorder 一律从这里查型号，禁止各处手写分支。
# Phase 8 接入 per-unit name 字段后，此处退化为兜底默认。）
SIDE_MODEL: dict[Side, str] = {
    Side.NATO: "M1A2",
    Side.PACT: "T-14",
    Side.NEUTRAL: "载具",
}

# 阵营 id × 角色 id → 单位型号覆盖表（在 SIDE_MODEL 之上按 role 细分；
# 键 = (int(Side), int(UnitRole))——side.py 在 entities 层，不依赖 simulation 层的
# UnitRole，故 role 用整型常量并在此注释对齐：0=坦克 1=炮兵 2=侦察 3=防空 4=攻顶 5=空中。
# 2026-08-18 修复：演示里炮兵（role=ARTILLERY）此前落到阵营默认型号（M1A2/T-14 MBT），
# HUD / 详情 / 叙事简报都把 M109 显示成主战坦克；此处按 role 覆盖为自行火炮型号。
# 空中（role=5，adapter.aircraft_name 查询时传）按阵营给战斗机默认型号。）
SIDE_ROLE_MODEL: dict[tuple[int, int], str] = {
    (int(Side.NATO), 1): "M109",   # 北约炮兵（演示装配 M109 155mm 自行炮）
    (int(Side.PACT), 1): "2S19",   # 华约炮兵（155mm 自行炮对应物）
    (int(Side.NATO), 5): "F-16C",  # 北约战斗机（Phase 7 空战演示）
    (int(Side.PACT), 5): "Su-27",  # 华约战斗机
}


class SideNameMapper:
    """阵营 id → 显示名映射器（可注入自定义表，如剧情中的化名阵营）。"""

    def __init__(self, names: dict[int, str] | None = None) -> None:
        self._names: dict[int, str] = {int(s): n for s, n in SIDE_NAME.items()}
        if names:
            self._names.update({int(k): v for k, v in names.items()})

    def name(self, side: int) -> str:
        """阵营 id → 显示名；未知 id 返回 "未知"。"""
        return self._names.get(int(side), "未知")

    def __call__(self, side: int) -> str:
        return self.name(side)


def side_name(side: int) -> str:
    """便捷函数：默认映射表的显示名（未知 id → "未知"）。"""
    try:
        return SIDE_NAME[Side(int(side))]
    except (ValueError, KeyError):
        return "未知"


class SideModelMapper:
    """阵营 id × 角色 id → 默认单位型号映射器（与 SideNameMapper 同构，可注入自定义表）。

    消费方约定：HUD / DetailPanel（render/adapter.py）与叙事层
    （ui/main_window.py 生成 unit_names）统一从本类查询型号，
    不允许在渲染/UI 层各自硬编码型号名。

    查询顺序：role 覆盖表（SIDE_ROLE_MODEL）优先 → 阵营默认表（SIDE_MODEL）
    → "载具"（未知）。role 缺省 0（坦克）→ 纯阵营默认语义，向后兼容。
    """

    def __init__(
        self,
        models: dict[int, str] | None = None,
        role_models: dict[tuple[int, int], str] | None = None,
    ) -> None:
        self._models: dict[int, str] = {int(s): m for s, m in SIDE_MODEL.items()}
        if models:
            self._models.update({int(k): v for k, v in models.items()})
        self._role_models: dict[tuple[int, int], str] = {
            (int(s), int(r)): m for (s, r), m in SIDE_ROLE_MODEL.items()
        }
        if role_models:
            self._role_models.update(
                {(int(s), int(r)): m for (s, r), m in role_models.items()}
            )

    def model(self, side: int, role: int = 0) -> str:
        """阵营 id × 角色 id → 默认型号；role 覆盖优先，回退阵营默认；未知 → "载具"。"""
        side_i, role_i = int(side), int(role)
        role_model = self._role_models.get((side_i, role_i))
        if role_model is not None:
            return role_model
        return self._models.get(side_i, "载具")

    def __call__(self, side: int, role: int = 0) -> str:
        return self.model(side, role)


def side_model(side: int, role: int = 0) -> str:
    """便捷函数：默认映射表的单位型号（role 覆盖优先；未知 id → "载具"）。"""
    side_i, role_i = int(side), int(role)
    role_model = SIDE_ROLE_MODEL.get((side_i, role_i))
    if role_model is not None:
        return role_model
    try:
        return SIDE_MODEL[Side(side_i)]
    except (ValueError, KeyError):
        return "载具"


def is_neutral(side: int) -> bool:
    """中立判定（中立方不敌视任何人，也不被任何人敌视）。"""
    return int(side) == int(Side.NEUTRAL)


def hostile(a_side: int, b_side: int) -> bool:
    """a、b 两个绝对阵营是否敌对。

    - 同阵营 → False（友军，含同阵营不同指挥链；Phase 8 联盟图可拆）；
    - 任一中立 → False；
    - 异阵营（且均非中立）→ True。
    """
    if int(a_side) == int(b_side):
        return False
    return not (is_neutral(a_side) or is_neutral(b_side))


def allegiance(a_side: int, a_chain: int, b_side: int, b_chain: int) -> str:
    """绝对阵营/指挥链 → 相对敌友（"ally" / "enemy" / "neutral"）。

    供叙事层 side_map / 触发器过滤等派生场景使用（armor_idx → 敌友字符串）。
    a_chain / b_chain 为指挥链 id（formation_id / 未来 coalition_id）；
    本期同阵营一律 "ally"；Phase 8 换 networkx 联盟图时签名不变。
    """
    if is_neutral(a_side) or is_neutral(b_side):
        return "neutral"
    if int(a_side) == int(b_side):
        return "ally"
    return "enemy"
