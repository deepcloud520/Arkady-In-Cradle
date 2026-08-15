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
