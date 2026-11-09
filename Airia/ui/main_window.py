"""主窗口（C4I 浮层版，2026-11-03 重构）。

布局（docs/ui-cmo-redesign.md §3.6.2）：
    地图铺满 + 卡片式半透明浮层，取代旧的「QSplitter 分栏 + 4 Tab」：

    ┌ 菜单栏 ─────────────────────────────────────────────┐
    │  ┌ 时间栏（左上浮层）────────────────────────┐        │
    │  地图（GLWidget，铺满）          ┌ UNIT STATUS ┐      │
    │  ┌ 实时通话（左侧弹窗，通话时出现）│ 详情卡片列  │      │
    │  └ MESSAGE LOG（底部卡片）───────────────────┘        │
    └ 状态栏 ─────────────────────────────────────────────┘

用户 2026-11-03 定稿的四条口径：
    ① 缩略图用无版权装备图片（``ui/unit_thumb.py``）
    ② 系统/教程对话 → MESSAGE LOG（``append_system``）
    ③ 实时通话 → 左侧弹窗 + 头像/Sound only + STT（``ui/comms_popup.py``）
    ④ 简报 → 中央独立窗口（``ui/windows.py``）

窗口形态：**可缩放窗口**（用户可拖边缘 / 最大化铺满），不是独占全屏。
所有尺寸走 ``ui/metrics.py``，resize 时重算，不重建控件树。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.simulation.loop import SimulationLoop
from Airia.narrative.engine import NarrativeEngine
from Airia.narrative.recorder import CombatEventRecorder
from Airia.render.adapter import WorldAdapter, decode_handle
from Airia.render.gl_widget import GLWidget
from Airia.render.hud.log_widget import LogWidget
from Airia.ui import palette as pal
from Airia.ui.archive_panel import ArchivePanel
from Airia.ui.briefing_panel import BriefingPanel
from Airia.ui.cockpit import Cockpit
from Airia.ui.comms_popup import CommsPopup
from Airia.ui.controls import ControlMap
from Airia.ui.detail_panel import DetailPanel
from Airia.ui.metrics import UI_SCALES, Metrics, for_viewport
from Airia.ui.message_panel import MessageLogPanel
from Airia.ui.timebar import TimeBar, TimeBarState, format_cmo_line, format_selected
from Airia.ui.windows import ArchiveWindow, BriefingWindow

from Airia.Arkady_In_Cradle import world1

logger = get_logger(__name__)

# 项目根（story/ 目录定位）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 演示场景起始时刻（当地时间 07:45:44，东欧 UTC+2）
_SCENE_BASE_SECONDS = 7 * 3600 + 45 * 60 + 44
_SCENE_DURATION_S = 24 * 3600.0
_ZULU_OFFSET_S = -2 * 3600

# 事件类型 → 中文标签（日志 / 档案 / 详情复用）
_KIND_CN = {
    "fire": "开火",
    "hit": "命中",
    "destroyed": "击毁",
    "miss": "脱靶",
    "detection": "探测",
    "command": "指令",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("[D8] Arkady in Cradle —— A fanfic work of Neuroverse")
        self.setMinimumSize(1024, 640)

        # UI 状态（响应式）
        self._ui_scale = 1.0
        self._ui_theme = "dark"
        self._last_metrics: Metrics = for_viewport(1440, 900, self._ui_scale)

        # ── UI 布局（先建 GLWidget，其 terrain 供模拟层采样单位高度）──
        self._build_ui()
        self._init_geometry()

        # ── 模拟层（Phase 3 无头引擎）+ 单位上下文 ──
        self.simulation, unit_names, side_map = self._create_simulation()

        # ── 叙事层（Phase 4，装配事件记录器供档案/简报窗口）──
        recorder = CombatEventRecorder(
            db_path=":memory:",
            unit_registry=unit_names,
            side_map=side_map,
            mission_id="demo_001",
        )
        self.narrative = NarrativeEngine(
            command_queue=self.simulation.commands,
            recorder=recorder,
            loop=self.simulation,
        )
        self.narrative.set_unit_context(unit_registry=unit_names, side_map=side_map)
        self._load_story()

        # ── 渲染数据源（Phase 5 透传模式）──
        self.adapter: WorldAdapter | None = None

        # ── 中央窗口实例（懒创建，复用）──
        self._archive_window: ArchiveWindow | None = None
        self._briefing_window: BriefingWindow | None = None

        # ── 信号槽（拾取 → 详情面板；通话选项 → 叙事引擎）──
        self.gl_widget.unit_hovered.connect(self.detail_panel.set_hover)
        self.gl_widget.unit_selected.connect(self.detail_panel.set_selection)
        self.comms.option_selected.connect(self._on_option_selected)

        # ── Phase 5e：键盘快捷键（ui/controls.py）──
        # 事件过滤器只装在 GLWidget 上：焦点在战场时响应热键；
        # 焦点在按钮/输入框时按键走 Qt 默认行为（空格=点击按钮等）。
        self.gl_widget.installEventFilter(self)
        QShortcut(QKeySequence("Tab"), self, activated=self._toggle_sidebar)

        # ── 主循环（帧定时器）──
        self._paused = False            # 对话暂停（叙事层驱动）
        self._manual_paused = False     # Phase 5e：手动暂停（空格键，独立）
        self._active_dialogue_id: str | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_clock = time.monotonic()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_frame)
        self._tick_timer.start(16)

        # Phase 5e：渲染层速度估计状态（位置差分 + EMA）
        self._prev_pos: np.ndarray | None = None
        self._speed_ema: np.ndarray | None = None

        # 状态栏刷新（100ms）
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(100)

        # 主题菜单需等 GLWidget.initializeGL（ThemeManager 就绪）后填充
        QTimer.singleShot(0, self._populate_theme_menu)

        # 系统/教程消息（② 定稿：系统对话进 MESSAGE LOG，不阻塞推演）
        self._push_system_intro()

    # ------------------------------------------------------------------
    # 窗口几何（可缩放窗口：按可用区域 85% 居中）
    # ------------------------------------------------------------------

    def _init_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        avail = screen.availableGeometry()   # 扣掉任务栏，避免底部被压住
        w = max(1024, int(avail.width() * 0.85))
        h = max(640, int(avail.height() * 0.85))
        self.resize(w, h)
        self.move(
            avail.center().x() - w // 2,
            avail.center().y() - h // 2,
        )

    # ------------------------------------------------------------------
    # 模拟层装配（演示编队）
    # ------------------------------------------------------------------

    def _create_simulation(
        self,
    ) -> tuple[SimulationLoop, dict[int, str], dict[int, str]]:
        return world1.create_simulation(self.gl_widget)

    # ------------------------------------------------------------------
    # 叙事层加载
    # ------------------------------------------------------------------

    def _load_story(self) -> None:
        """加载 story/ 对话树与触发器（剧情文案仍由 story/ 文档定稿）。"""
        dialogues_dir = PROJECT_ROOT / "story" / "dialogues"
        triggers_dir = PROJECT_ROOT / "story" / "triggers"
        if dialogues_dir.is_dir():
            n = self.narrative.load_dialogues(dialogues_dir)
            logger.info("已加载 %d 棵对话树", n)
        if triggers_dir.is_dir():
            n = self.narrative.load_triggers(triggers_dir)
            logger.info("已加载 %d 个触发器", n)

        self.narrative.set_flag("has_scout_report")

    # ------------------------------------------------------------------
    # UI 布局
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.gl_widget = GLWidget()
        self.comms = CommsPopup()
        self.timebar = TimeBar()
        # MESSAGE LOG = 真面板（可折叠/拖动/关闭），默认折叠只留标题栏
        self.log_widget = LogWidget()
        self.message_panel = MessageLogPanel(self.log_widget)
        self.detail_panel = DetailPanel(event_source=self._recent_events_for)
        self.archive_panel = ArchivePanel(timeline_source=self._timeline_for)
        self.briefing_panel = BriefingPanel(generate_cb=self._generate_briefing)

        self.cockpit = Cockpit(
            map_widget=self.gl_widget,
            side_widget=self.detail_panel,
            message_panel=self.message_panel,
            comms=self.comms,
            timebar=self.timebar,
            user_scale=self._ui_scale,
        )
        self.setCentralWidget(self.cockpit)

        self._build_menus()
        self._apply_ui_theme(self._ui_theme)

        # 沉浸感：不显示常驻状态栏（F3 可临时唤出，看 FPS/相机等调试信息）
        self.statusBar().hide()

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        view_menu = menubar.addMenu("视图")
        self._terrain_menu = view_menu.addMenu("地形风格")
        self._theme_actions: dict[str, QAction] = {}

        ui_menu = view_menu.addMenu("界面配色")
        self._ui_theme_group = QActionGroup(self)
        self._ui_theme_group.setExclusive(True)
        self._ui_theme_actions: dict[str, QAction] = {}
        for tid in ("dark", "light"):
            act = QAction("深色（战术终端）" if tid == "dark" else "浅色（指挥所）",
                          self, checkable=True)
            act.setChecked(tid == self._ui_theme)
            act.triggered.connect(lambda _=False, i=tid: self._apply_ui_theme(i))
            self._ui_theme_group.addAction(act)
            ui_menu.addAction(act)
            self._ui_theme_actions[tid] = act

        scale_menu = view_menu.addMenu("UI 缩放")
        self._scale_group = QActionGroup(self)
        self._scale_group.setExclusive(True)
        for s in UI_SCALES:
            act = QAction(f"{int(round(s * 100))}%", self, checkable=True)
            act.setChecked(abs(s - self._ui_scale) < 1e-6)
            act.triggered.connect(lambda _=False, v=s: self._set_ui_scale(v))
            self._scale_group.addAction(act)
            scale_menu.addAction(act)

        view_menu.addSeparator()
        self._sidebar_action = QAction("显示侧栏", self, checkable=True)
        self._sidebar_action.setChecked(True)
        self._sidebar_action.setShortcut(QKeySequence("Tab"))
        self._sidebar_action.triggered.connect(self._toggle_sidebar)
        view_menu.addAction(self._sidebar_action)

        self._status_action = QAction("状态栏（调试）", self, checkable=True)
        self._status_action.setShortcut(QKeySequence("F3"))
        self._status_action.triggered.connect(self._toggle_status_bar)
        view_menu.addAction(self._status_action)

        narrative_menu = menubar.addMenu("叙事")
        skip_act = QAction("跳过当前对话", self)
        skip_act.triggered.connect(self._skip_dialogue)
        narrative_menu.addAction(skip_act)
        # 真正战役：禁用一切暂停（玩家/教程都不行），见 narrative/pause.py
        self._campaign_action = QAction("战役模式（禁用暂停）", self, checkable=True)
        self._campaign_action.triggered.connect(self._toggle_campaign_mode)
        narrative_menu.addAction(self._campaign_action)

        panel_menu = menubar.addMenu("面板")
        archive_act = QAction("战斗档案", self)
        archive_act.triggered.connect(self._show_archive)
        panel_menu.addAction(archive_act)
        briefing_act = QAction("战斗简报", self)
        briefing_act.triggered.connect(self._show_briefing)
        panel_menu.addAction(briefing_act)
        log_act = QAction("消息日志", self)
        log_act.setShortcut(QKeySequence("L"))
        log_act.triggered.connect(self._toggle_message_panel)
        panel_menu.addAction(log_act)

        help_menu = menubar.addMenu("帮助")
        help_act = QAction("快捷键", self)
        help_act.triggered.connect(
            lambda: self.statusBar().showMessage(ControlMap.HELP_TEXT, 8000)
        )
        help_menu.addAction(help_act)

    # ------------------------------------------------------------------
    # 界面配色 / 缩放 / 侧栏
    # ------------------------------------------------------------------

    def _apply_ui_theme(self, name: str) -> None:
        p = pal.set_current(name)
        self._ui_theme = p.name
        # 调色板 + 样式表一起换：调色板让原生控件（滚动条/菜单/按钮面）保持
        # 真控件质感，样式表只负责配色与渐变（2026-11-03 修正平涂问题）。
        app = QApplication.instance()
        if app is not None:
            app.setPalette(p.qpalette())
        self.setStyleSheet(pal.qss(p))
        self.cockpit.set_palette(p)
        self.comms.apply_palette(p)
        self.message_panel.set_palette(p)
        self.log_widget.set_palette(p)
        self.detail_panel.apply_palette(p)
        self.briefing_panel.apply_palette(p)
        self.archive_panel.apply_palette(p)
        for win in (
            getattr(self, "_archive_window", None),
            getattr(self, "_briefing_window", None),
        ):
            if win is not None:
                win.apply_palette(p)
        for tid, act in getattr(self, "_ui_theme_actions", {}).items():
            act.setChecked(tid == p.name)
        self.statusBar().showMessage(f"界面配色：{p.name}", 2500)

    def _set_ui_scale(self, scale: float) -> None:
        self._ui_scale = float(scale)
        self.cockpit.set_user_scale(self._ui_scale)

    def _toggle_sidebar(self) -> None:
        state = self.cockpit.toggle_sidebar()
        self._sidebar_action.setChecked(state)

    def _toggle_status_bar(self) -> None:
        bar = self.statusBar()
        if bar.isVisible():
            bar.hide()
        else:
            bar.show()
        self._status_action.setChecked(bar.isVisible())

    def _system(self, text: str) -> None:
        """系统/教程消息：进 MESSAGE LOG 面板（折叠时累计未读数）。"""
        self.message_panel.push_system(text)

    def _toggle_message_panel(self) -> None:
        self.cockpit.toggle_message_panel()

    # ------------------------------------------------------------------
    # 中央窗口（档案 / 简报）
    # ------------------------------------------------------------------

    def _show_archive(self) -> None:
        if self._archive_window is None:
            self._archive_window = ArchiveWindow(
                timeline_source=self._timeline_for, parent=self
            )
        self._archive_window.show()
        self._archive_window.raise_()

    def _show_briefing(self) -> None:
        if self._briefing_window is None:
            self._briefing_window = BriefingWindow(
                generate_cb=self._generate_briefing, parent=self
            )
        self._briefing_window.show()
        self._briefing_window.raise_()

    # ------------------------------------------------------------------
    # 地形主题菜单（GL shader 主题，与界面配色是两根轴）
    # ------------------------------------------------------------------

    def _populate_theme_menu(self) -> None:
        tm = self.gl_widget.theme_manager
        if tm is None:
            return
        for tid in tm.discover():
            act = QAction(tid, self, checkable=True)
            if tm.current is not None and tm.current.id == tid:
                act.setChecked(True)
            act.triggered.connect(lambda _=False, i=tid: self._switch_theme(i))
            self._terrain_menu.addAction(act)
            self._theme_actions[tid] = act

    def _switch_theme(self, theme_id: str) -> None:
        tm = self.gl_widget.theme_manager
        if tm is None:
            return
        # 菜单槽触发时 GL context 不保证 current：整个 switch（编译新 program
        # + 回调里重建 VAO / 写 uniform）都必须包在 makeCurrent() 内。
        self.gl_widget.makeCurrent()
        try:
            theme = tm.switch(theme_id)
        finally:
            self.gl_widget.doneCurrent()
        for tid, act in self._theme_actions.items():
            act.setChecked(tid == theme.id)
        self.statusBar().showMessage(f"地形风格已切换：{theme.name}", 3000)

    # ------------------------------------------------------------------
    # 叙事交互
    # ------------------------------------------------------------------

    def _toggle_campaign_mode(self) -> None:
        """真正战役模式：进入后玩家/教程都不能暂停推演。"""
        active = not self.narrative.pause_locked
        self.narrative.set_campaign_mode(active)
        self._manual_paused = self.narrative.pause.manual
        self._paused = self.narrative.is_paused
        self._campaign_action.setChecked(active)
        self._system(
            "已进入战役模式：推演不可暂停"
            if active
            else "已退出战役模式：可暂停"
        )

    def _skip_dialogue(self) -> None:
        state = self.narrative.skip_active_dialogue()
        self._paused = self.narrative.is_paused
        self._manual_paused = self.narrative.pause.manual
        self.comms.set_state(state)
        self._system("对话已跳过，推演恢复")
        self.statusBar().showMessage("对话已跳过，模拟恢复", 3000)

    def _on_option_selected(self, dialogue_id: str, option_index: int) -> None:
        state = self.narrative.on_player_choice(dialogue_id, option_index)
        self._paused = self.narrative.is_paused
        self.comms.set_state(state)

    # ------------------------------------------------------------------
    # Phase 5e：键盘快捷键
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        if (
            obj is self.gl_widget
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
        ):
            action = ControlMap.key_action(event)
            if action is not None:
                if action == ControlMap.CANCEL_SELECTION:
                    return False  # 交还 GLWidget 处理（取消选中）
                self._handle_action(action)
                return True
        return super().eventFilter(obj, event)

    def _handle_action(self, action: str) -> None:
        if action == ControlMap.TOGGLE_PAUSE:
            # 暂停是权限受限操作：真正战役中一律拒绝（玩家 / 教程都不行）
            accepted = self.narrative.request_manual_pause()
            if not accepted:
                self._system("战役进行中：暂停不可用")
                self.statusBar().showMessage("战役进行中：暂停不可用", 3000)
                return
            self._manual_paused = self.narrative.pause.manual
            self._system("推演暂停" if self._manual_paused else "推演继续")
            self.statusBar().showMessage(
                "⏸ 手动暂停" if self._manual_paused else "▶ 恢复运行", 2000
            )
        elif action == ControlMap.ADVANCE_DIALOGUE:
            if self.narrative.has_active_dialogue:
                state = self.narrative.advance_dialogue(None)
                self._paused = self.narrative.is_paused
                self.comms.set_state(state)
        elif action == ControlMap.SKIP_DIALOGUE:
            self._skip_dialogue()
        elif action.startswith(ControlMap.OPTION_PREFIX):
            try:
                idx = int(action[len(ControlMap.OPTION_PREFIX):])
            except ValueError:
                return
            self._choose_option(idx)
        elif action.startswith(ControlMap.THEME_PREFIX):
            tid = action[len(ControlMap.THEME_PREFIX):]
            tm = self.gl_widget.theme_manager
            if tm is not None and tid in tm.discover():
                self._switch_theme(tid)
        elif action == ControlMap.SHOW_HELP:
            self.statusBar().showMessage(ControlMap.HELP_TEXT, 8000)

    def _choose_option(self, option_index: int) -> None:
        did = self._active_dialogue_id
        if did is None:
            return
        state = self.narrative.on_player_choice(did, option_index)
        self._paused = self.narrative.is_paused
        self.comms.set_state(state)

    # ------------------------------------------------------------------
    # 面板数据源
    # ------------------------------------------------------------------

    def _recent_events_for(self, armor_idx: int) -> list[str]:
        rec = getattr(self, "narrative", None)
        if rec is None or rec.recorder is None:
            return []
        r = rec.recorder
        evs = r.query(target_id=armor_idx, limit=10) + r.query(
            source_id=armor_idx, limit=10
        )
        evs.sort(key=lambda e: e.sim_time)
        return [
            f"[{e.sim_time:.1f}s] {_KIND_CN.get(e.event_type, e.event_type)}"
            for e in evs[-5:]
        ]

    def _timeline_for(self) -> list[str]:
        rec = getattr(self, "narrative", None)
        if rec is None or rec.recorder is None:
            return []
        lines = []
        for e in rec.recorder.timeline()[-200:]:
            src = e.source_name or f"#{e.source_id}"
            tgt = e.target_name or f"#{e.target_id}"
            kind = _KIND_CN.get(e.event_type, e.event_type)
            lines.append(f"[{e.sim_time:6.1f}s] {kind} {src}→{tgt}")
        return lines

    def _generate_briefing(self):
        rec = getattr(self, "narrative", None)
        if rec is None:
            return None
        return rec.generate_briefing()

    # ------------------------------------------------------------------
    # 系统 / 教程消息（② 定稿）
    # ------------------------------------------------------------------

    def _push_system_intro(self) -> None:
        self._system("战术终端已就绪：右侧 UNIT STATUS，Tab 显隐。")
        self._system("左键拖拽旋转视角，滚轮缩放；单击选中，Esc 取消。")
        self._system("L 查看消息日志；F3 调试状态栏；F1 快捷键。")

    # ------------------------------------------------------------------
    # 主循环（docs §2.3 数据流）
    # ------------------------------------------------------------------

    def _on_frame(self) -> None:
        # 1. 模拟层推进（暂停策略统一由 narrative/pause.py 决定 → loop.sim_paused）
        if not self.simulation.sim_paused:
            self.simulation.step(1 / 60)

        # 2. 孪生适配器（Phase 5 透传；Phase 8 传感器融合）
        events = self.simulation.events.drain()
        snapshot = self.simulation.state.snapshot()
        armor = snapshot.pools.get("armor")
        speed = self._estimate_speed(armor.pos if armor is not None else None)
        self.adapter = WorldAdapter.from_snapshot(snapshot, armor_speed=speed)

        # 3. 叙事层消费
        narrative_state = self.narrative.tick(snapshot, events)
        self._paused = narrative_state.paused
        # 与暂停策略保持同步（战役上锁会清空手动暂停）
        self._manual_paused = self.narrative.pause.manual
        self._campaign_action.setChecked(self.narrative.pause_locked)
        d = narrative_state.active_dialogue
        self._active_dialogue_id = (
            d.dialogue_id if d is not None and not d.is_finished else None
        )

        # 4. UI 刷新（注入 adapter，永不注入原始 snapshot）
        self.gl_widget.set_adapter(self.adapter)
        self.detail_panel.set_adapter(self.adapter)
        self.comms.set_state(narrative_state)
        self._sync_metrics()
        self._update_timebar(snapshot)

        if events:
            event_dicts = [
                {"time": snapshot.sim_time, "kind": ev.kind, "data": ev.data}
                for ev in events
            ]
            self.message_panel.push_events(event_dicts)
            self.gl_widget.feed_effects(event_dicts)

        # 5. 渲染刷新
        self.gl_widget.update()

        self._frame_count += 1
        now = time.monotonic()
        if now - self._fps_clock >= 1.0:
            self._fps = self._frame_count / (now - self._fps_clock)
            self._frame_count = 0
            self._fps_clock = now

    def _sync_metrics(self) -> None:
        """窗口尺寸变化后把 metrics 分发给各浮层（相等则跳过）。"""
        m = self.cockpit.metrics
        if m == self._last_metrics:
            return
        self._last_metrics = m
        self.detail_panel.set_metrics(m)
        self.comms.set_metrics(m)
        self.message_panel.set_metrics(m)
        self.log_widget.set_metrics(m)
        self.timebar.set_metrics(m)

    def _update_timebar(self, snapshot) -> None:
        """时间栏数据（CMO 风格单行：Time / Local / Zulu / 剩余 / Camera Alt）。"""
        sim_t = float(snapshot.sim_time)
        local = int(_SCENE_BASE_SECONDS + sim_t) % 86400
        zulu = (local + _ZULU_OFFSET_S) % 86400
        remain = max(0.0, _SCENE_DURATION_S - sim_t)

        local_str = f"{local // 3600:02d}:{local % 3600 // 60:02d}:{local % 60:02d}"
        zulu_str = f"{zulu // 3600:02d}:{zulu % 3600 // 60:02d}:{zulu % 60:02d}"
        line = format_cmo_line(
            local_str, zulu_str, remain, self.gl_widget.camera.eye().z
        )

        kind, text = self._selected_text()
        self.timebar.set_state(
            TimeBarState(
                clock=local_str,
                line=line,
                selected=text,
                selected_kind=kind,
            )
        )

    def _selected_text(self) -> tuple[str, str]:
        """选中状态 → (kind, Selected 行文本)。§2.7 三态：悬浮不参与。"""
        handle = getattr(self.gl_widget, "selected_handle", -1)
        if self.adapter is None or handle < 0:
            return "none", format_selected("none")
        unit_kind, idx = decode_handle(handle)
        if unit_kind == 1:   # 空中单位
            name = self.adapter.aircraft_name(handle)
            return "single", format_selected("single", name)
        name = self.adapter.unit_name(idx)
        fs = self.adapter.unit_formation_status(idx)
        formation = f"编队 #{fs['formation_id']}" if fs is not None else ""
        return "single", format_selected("single", name, formation)

    # ------------------------------------------------------------------
    # Phase 5e：渲染层速度估计（位置差分 + EMA，不动引擎）
    # ------------------------------------------------------------------

    def _estimate_speed(self, pos: np.ndarray | None) -> np.ndarray | None:
        if pos is None or len(pos) == 0:
            self._prev_pos = None
            self._speed_ema = None
            return None
        if self._prev_pos is None or self._prev_pos.shape != pos.shape:
            self._prev_pos = pos.copy()
            self._speed_ema = np.zeros(len(pos), dtype=np.float32)
            return self._speed_ema
        delta = np.hypot(
            pos[:, 0] - self._prev_pos[:, 0], pos[:, 1] - self._prev_pos[:, 1]
        )
        raw = (delta / (1.0 / 60.0)).astype(np.float32)
        self._speed_ema = self._speed_ema * 0.6 + raw * 0.4
        self._prev_pos = pos.copy()
        return self._speed_ema

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------

    def _update_status_bar(self) -> None:
        cam = self.gl_widget.camera
        tm = self.gl_widget.theme_manager
        theme_name = tm.current.name if tm is not None and tm.current else "-"
        sim = self.simulation.state
        if self.narrative.pause_locked:
            state = "▶ 战役中（不可暂停）"
        elif self._manual_paused:
            state = "⏸ 手动暂停"
        elif self._paused:
            state = "⏸ 对话暂停"
        else:
            state = "▶ 运行中"
        n_units = self.adapter.unit_count() if self.adapter else 0
        m = self.cockpit.metrics
        self.statusBar().showMessage(
            f"{state} | 模拟 t={sim.sim_time:6.1f}s tick={sim.tick_count} | "
            f"FPS={self._fps:5.1f} | 单位={n_units} | 地形={theme_name} | "
            f"界面={self._ui_theme} | 布局={m.mode} k={m.k:.2f} | "
            f"相机 d={cam.distance:.0f} yaw={cam.yaw:.0f}° pitch={cam.pitch:.0f}°"
        )

