"""单位详情面板（SidePanel Tab4，Phase 5b 新增，三级信息详细度）。

对应 docs/phase5-ui-pipeline-design.md §4.4：
    - 级别 0：无交互（全局概况）——双方兵力 / 平均弹药燃料 / 模拟时间
    - 级别 1：鼠标悬浮（悬浮基础）——单位名 / 阵营 / 状态 / 损坏摘要 / 位置
    - 级别 2：点击选中（选中完整）——级别 1 + 血量 / 航向 / 部件逐项 / 后勤 / 最近事件

2026-08-18 增强：
    - 选择句柄覆盖地面装甲 + 空中单位（decode_handle 分流，飞机有独立详情视图）；
    - 地面部件逐项显示**每部位装甲厚度**（adapter.part_armor_mm）；
    - 地面单位新增**编队 AI 状态**区（意图/命令/姿态/通信质量/成员数）。

数据来源全部通过 WorldAdapter（渲染层永不直读 Real SoA）；最近事件
由外部注入 event_source 回调（消费 CombatEventRecorder）。
"""
from __future__ import annotations

import math
from typing import Callable

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from Airia.render.adapter import PART_NAMES, WorldAdapter, decode_handle, is_air_handle
from Airia.ui import palette as pal
from Airia.ui.metrics import Metrics, for_viewport

# 阵营 → 部位损毁描述（级别 2 部件逐项）
_PART_HINT = {
    "引擎": "失速",
    "履带": "无法机动",
    "炮塔": "无法开火",
    "弹药架": "殉爆风险",
    "车体": "结构受损",
}

# 编队 AI 状态中文（simulation/ai.py：FormationIntent / UnitOrder）
_FORMATION_INTENT_CN = {0: "机动", 1: "进攻", 2: "防御", 3: "撤退", 4: "待命"}
_UNIT_ORDER_CN = {0: "进攻", 1: "防御", 2: "机动", 3: "撤退", 4: "待命"}
_STANCE_CN = {0: "进攻", 1: "防御"}
# 飞机飞行状态中文（entities/pools.py AircraftState）
_AIRCRAFT_STATE_CN = {
    0: "起飞", 1: "巡航", 2: "交战", 3: "返航", 4: "降落", 5: "坠毁", 6: "驻停",
}


class DetailPanel(QWidget):
    """三级信息详细度详情面板。"""

    def __init__(
        self,
        event_source: Callable[[int], list[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._adapter: WorldAdapter | None = None
        self._hover = -1
        self._selection = -1
        self._event_source = event_source

        self._palette = pal.current()
        self._metrics: Metrics = for_viewport(1440, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._view = QTextBrowser(self)
        self._view.setOpenExternalLinks(False)
        layout.addWidget(self._view)
        self.apply_palette(self._palette)
        self._refresh()

    # ------------------------------------------------------------------
    # 配色 / 字号（C4I 浮层：浏览器底透明，让卡片透出地图）
    # ------------------------------------------------------------------

    def apply_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_style()
        self._refresh()

    def set_metrics(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._apply_style()

    def _apply_style(self) -> None:
        p = self._palette
        size = max(8, int(round(self._metrics.font_sm)))
        # 凹陷文档面（面板里的"内容控件"），而不是透明文本压在卡片上
        self._view.setStyleSheet(
            "QTextBrowser {"
            f" background: {p.bar_bg}; color: {p.text};"
            f" border: 1px solid {p.border_dark};"
            f" padding: 4px; font-size: {size}px; }}"
        )
        self.setStyleSheet("QWidget { background: transparent; }")

    # ------------------------------------------------------------------
    # 数据注入（docs §2.1 / §4.4 信号槽）
    # ------------------------------------------------------------------

    def set_adapter(self, adapter: WorldAdapter | None) -> None:
        self._adapter = adapter
        self._refresh()

    def set_hover(self, idx: int) -> None:
        self._hover = idx
        self._refresh()

    def set_selection(self, idx: int) -> None:
        self._selection = idx
        self._refresh()

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._adapter is None:
            self._view.setHtml("<p style='color:#66666e'>等待模拟数据...</p>")
            return
        if self._selection >= 0:
            html = self._level2_handle(self._selection)
        elif self._hover >= 0:
            html = self._level1_handle(self._hover)
        else:
            html = self._level0()
        self._view.setHtml(html)

    def _level1_handle(self, handle: int) -> str:
        """悬浮（handle 分流：空中单位 → 飞机简版，地面 → 原逻辑）。"""
        if is_air_handle(handle):
            return self._level1_aircraft(handle)
        return self._level1(handle)

    def _level2_handle(self, handle: int) -> str:
        """选中（handle 分流：空中单位 → 飞机详情，地面 → 原逻辑）。"""
        if is_air_handle(handle):
            return self._level2_aircraft(handle)
        return self._level2(handle)

    def _level0(self) -> str:
        s = self._adapter.get_stats()
        t = s["sim_time"]
        mm, ss = int(t // 60), int(t % 60)
        return (
            "<h4>全局概况</h4>"
            f"<p>友军装甲：<b>{s['n_ally']}</b>　敌军装甲：<b>{s['n_enemy']}</b>"
            + (f"　中立：{s['n_neutral']}" if s["n_neutral"] else "")
            + "</p>"
            f"<p>平均弹药存量：<b>{s['avg_ammo']:.0f}%</b>"
            f"　平均燃料存量：<b>{s['avg_fuel']:.0f}%</b></p>"
            f"<p>模拟时间：<b>{mm:02d}:{ss:02d}</b>"
            f"（tick {s['tick_count']}）</p>"
        )

    def _level1(self, i: int) -> str:
        a = self._adapter
        name = a.unit_name(i)
        side = a.side_name(i)
        damaged = a.damaged_parts(i)
        dmg = f"<span style='color:#e0b040'>⚠ {' '.join(damaged)}</span>" if damaged else "无"
        x, y = float(a.armor_pos[i][0]), float(a.armor_pos[i][1])
        return (
            f"<h4>📌 {name}（{side}）</h4>"
            "<p>状态：<b>ACTIVE</b></p>"
            f"<p>损坏：{dmg}</p>"
            f"<p>位置：({x:.1f}, {y:.1f})</p>"
            "<hr><p style='color:#88888f'>（提示：点击查看完整信息）</p>"
        )

    def _level2(self, i: int) -> str:
        a = self._adapter
        name = a.unit_name(i)
        side = a.side_name(i)
        hp = float(a.armor_health[i])
        yaw_deg = math.degrees(float(a.armor_yaw[i]))
        x, y, z = (float(a.armor_pos[i][k]) for k in range(3))
        speed = a.speed_at(i)

        # 部件逐项 ✓/✕（仅显示该单位实际装配的部位——炮兵无引擎/履带/弹药架
        # 不再显示；已毁部位（槽位已释放）由 present_parts 从单位标记补回。
        # 2026-08-18：每部位附带标称装甲厚度（adapter.part_armor_mm，mm RHA）
        present = set(a.present_parts(i))
        damaged = set(a.damaged_parts(i))
        part_rows = []
        for pname in PART_NAMES:
            if pname not in present:
                continue  # 未装配部位不显示（2026-08-18 修复）
            if pname in damaged:
                hint = _PART_HINT.get(pname, "受损")
                part_rows.append(
                    f"<span style='color:#e05a4a'>✕ {pname}（{hint}）</span>"
                )
            else:
                mm = a.part_armor_mm(i, pname)
                mm_txt = f" {mm:.0f}mm" if mm is not None else ""
                part_rows.append(f"<span style='color:#4ab85a'>✓ {pname}{mm_txt}</span>")
        parts = "　".join(part_rows) or "<span style='color:#88888f'>（无部位数据）</span>"

        # 编队 AI 状态（2026-08-18：意图/命令/姿态/通信质量/成员数）
        fs = a.unit_formation_status(i)
        if fs is not None:
            ai_txt = (
                f"<p>编队 <b>#{fs['formation_id']}</b>　意图：<b>{_FORMATION_INTENT_CN.get(fs['intent'], fs['intent'])}</b>"
                f"　命令：<b>{_UNIT_ORDER_CN.get(fs['order'], '—')}</b>"
                f"　姿态：<b>{_STANCE_CN.get(fs['stance'], '—')}</b></p>"
                f"<p>通信质量：<b>{fs['comm'] * 100:.0f}%</b>　成员数：<b>{fs['member_count']}</b></p>"
            )
        else:
            ai_txt = "<p>编队 AI：<b>（无编队）</b></p>"
        ai_block = (
            "<p>── 编队 AI ──</p>" + ai_txt
        )

        # 后勤
        ammo = a.ammo_at(i)
        fuel = a.fuel_at(i)
        ammo_txt = f"{ammo[0]:.0f}/{ammo[1]:.0f} 发" if ammo else "N/A"
        fuel_pct = f"{fuel[0] / fuel[1] * 100:.0f}%" if fuel and fuel[1] > 0 else "N/A"

        # 最近事件
        events = self._event_source(i) if self._event_source else []
        ev_txt = "".join(f"<li>{e}</li>" for e in events) or "<li>（无）</li>"

        return (
            f"<h4>📌 {name}（{side}）<span style='color:#e0b040'>[已选中]</span></h4>"
            f"<p>状态：<b>ACTIVE</b>　阵营：<b>{side}</b></p>"
            f"<p>血量：<b>{hp:.0f}/100</b>　置信度：<b>100%</b></p>"
            f"<p>位置：({x:.1f}, {y:.1f}, {z:.1f})　航向：<b>{yaw_deg:.0f}°</b></p>"
            f"<p>速度：<b>{speed:.0f} m/s</b></p>"
            "<p>── 部件状态（装甲厚度 mm RHA）──</p>"
            f"<p>{parts}</p>"
            + ai_block
            + "<p>── 后勤 ──</p>"
            f"<p>弹药：<b>{ammo_txt}</b>　燃料：<b>{fuel_pct}</b></p>"
            "<p>── 最近事件 ──</p>"
            f"<ul>{ev_txt}</ul>"
        )

    # ------------------------------------------------------------------
    # 空中单位视图（2026-08-18：handle = AIR_HANDLE_OFFSET + aircraft_idx）
    # ------------------------------------------------------------------

    def _level1_aircraft(self, handle: int) -> str:
        a = self._adapter
        name = a.aircraft_name(handle)
        side = a.aircraft_side_name(handle)
        st = a.aircraft_state_at(handle)
        st_cn = _AIRCRAFT_STATE_CN.get(st, "未知") if st is not None else "未知"
        alt = a.aircraft_alt_at(handle)
        spd = a.aircraft_speed_at(handle)
        return (
            f"<h4>✈ {name}（{side}）</h4>"
            f"<p>状态：<b>{st_cn}</b></p>"
            f"<p>高度：<b>{alt:.0f} m</b>　速度：<b>{spd:.0f} m/s</b></p>"
            "<hr><p style='color:#88888f'>（提示：点击查看完整信息）</p>"
        )

    def _level2_aircraft(self, handle: int) -> str:
        a = self._adapter
        name = a.aircraft_name(handle)
        side = a.aircraft_side_name(handle)
        st = a.aircraft_state_at(handle)
        st_cn = _AIRCRAFT_STATE_CN.get(st, "未知") if st is not None else "未知"
        hp = a.aircraft_health_at(handle)
        hp_txt = f"{hp:.0f}/100" if hp is not None else "—"
        alt = a.aircraft_alt_at(handle)
        spd = a.aircraft_speed_at(handle)
        ammo = a.aircraft_ammo_at(handle)
        ammo_txt = (
            f"AAM <b>{ammo[0]}</b> / AGM <b>{ammo[1]}</b>" if ammo else "—"
        )
        fuel = a.aircraft_fuel_at(handle)
        fuel_txt = f"{fuel:.0f}%" if fuel is not None else "—"
        # 空中编队（air_formation 池暂无 intent/comm 视图 → 只显示编队号）
        af = a.aircraft_formation_status(handle)
        fid_txt = f"<b>#{af['formation_id']}</b>" if af is not None else "（无编队）"
        return (
            f"<h4>✈ {name}（{side}）<span style='color:#e0b040'>[已选中]</span></h4>"
            f"<p>状态：<b>{st_cn}</b>　血量：<b>{hp_txt}</b></p>"
            f"<p>高度：<b>{alt:.0f} m</b>　速度：<b>{spd:.0f} m/s</b></p>"
            f"<p>武器挂载：{ammo_txt}　燃料：{fuel_txt}</p>"
            f"<p>编队：{fid_txt}</p>"
        )
