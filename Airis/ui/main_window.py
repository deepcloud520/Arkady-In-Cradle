"""主窗口（Phase 5：管线集成 + 主题菜单 + 主循环 + 快捷键）。

对应 docs/phase5-ui-pipeline-design.md §1 / §2.3：
    - 水平 QSplitter：GLWidget（左，拉伸 3）+ SidePanel（右，拉伸 1）
    - 菜单栏：主题切换（ThemeManager.switch 热切换）、叙事（跳过对话）
    - 主循环（QTimer 帧定时器驱动）：
        Simulation.step → WorldAdapter → NarrativeEngine.tick → UI 刷新
    - Phase 5e：
        * ui/controls.py 快捷键（空格暂停 / 回车推进对话 / 数字选选项 /
          S 跳过 / Ctrl+1/2/3 主题 / F1 帮助）——GLWidget 事件过滤器分发
        * 手动暂停（_manual_paused，与对话暂停独立）
        * 单位速度：渲染层位置差分 + EMA 平滑（不动引擎，adapter.armor_speed）
        * 战场特效事件喂入 GLWidget（弹道线 / 命中标记缓冲）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import QMainWindow, QSplitter, QTabWidget

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.simulation.loop import SimulationLoop
from Airis.narrative.engine import NarrativeEngine
from Airis.narrative.recorder import CombatEventRecorder
from Airis.render.adapter import WorldAdapter
from Airis.render.gl_widget import GLWidget
from Airis.render.hud.log_widget import LogWidget
from Airis.ui.archive_panel import ArchivePanel
from Airis.ui.briefing_panel import BriefingPanel
from Airis.ui.controls import ControlMap
from Airis.ui.detail_panel import DetailPanel
from Airis.ui.narrative_overlay import NarrativeOverlay

from Airis.Arkady_In_Cradle import world1
logger = get_logger(__name__)

# 项目根（story/ 目录定位）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("[D8] Arkady in Cradle —— A fanfic work of Neuroverse")
        self.setGeometry(100, 100, 1280, 800)

        # ── UI 布局（先建 GLWidget，其 terrain 供模拟层采样单位高度）──
        self._build_ui()

        # ── 模拟层（Phase 3 无头引擎）+ 单位上下文 ──
        self.simulation, unit_names, side_map = self._create_simulation()

        # ── 叙事层（Phase 4，装配事件记录器供档案/简报面板）──
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

        # ── 信号槽（拾取 → 详情面板；对话选项 → 叙事引擎）──
        self.gl_widget.unit_hovered.connect(self.detail_panel.set_hover)
        self.gl_widget.unit_selected.connect(self.detail_panel.set_selection)
        self.narrative_overlay.option_selected.connect(self._on_option_selected)

        # ── Phase 5e：键盘快捷键（ui/controls.py）──
        # 事件过滤器只装在 GLWidget 上：焦点在战场时响应热键；
        # 焦点在按钮/输入框时按键走 Qt 默认行为（空格=点击按钮等）。
        self.gl_widget.installEventFilter(self)

        # ── 主循环（帧定时器）──
        self._paused = False            # 对话暂停（叙事层驱动）
        self._manual_paused = False     # Phase 5e：手动暂停（空格键，独立）
        self._active_dialogue_id: str | None = None   # Phase 5e：当前对话 id（快捷键选选项用）
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
        return  # 暂不加载 story/ 对话树（Phase 4 叙事层演示）
        dialogues_dir = PROJECT_ROOT / "story" / "dialogues"
        triggers_dir = PROJECT_ROOT / "story" / "triggers"
        if dialogues_dir.is_dir():
            n = self.narrative.load_dialogues(dialogues_dir)
            logger.info("已加载 %d 棵对话树", n)
        if triggers_dir.is_dir():
            n = self.narrative.load_triggers(triggers_dir)
            logger.info("已加载 %d 个触发器", n)

    # ------------------------------------------------------------------
    # UI 布局
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 菜单栏
        menubar = self.menuBar()
        self._theme_menu = menubar.addMenu("主题")
        self._theme_actions: dict[str, QAction] = {}

        narrative_menu = menubar.addMenu("叙事")
        skip_act = QAction("跳过当前对话", self)
        skip_act.triggered.connect(self._skip_dialogue)
        narrative_menu.addAction(skip_act)

        # 中央：水平 QSplitter（左 GLWidget / 右 SidePanel）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.gl_widget = GLWidget()
        splitter.addWidget(self.gl_widget)

        # SidePanel = QTabWidget（对话 / 档案 / 简报 / 详情，docs §1.1）
        self.side_panel = QTabWidget()
        self.narrative_overlay = NarrativeOverlay()
        self.archive_panel = ArchivePanel(timeline_source=self._timeline_for)
        self.briefing_panel = BriefingPanel(generate_cb=self._generate_briefing)
        self.detail_panel = DetailPanel(event_source=self._recent_events_for)
        self.side_panel.addTab(self.narrative_overlay, "对话")
        self.side_panel.addTab(self.archive_panel, "档案")
        self.side_panel.addTab(self.briefing_panel, "简报")
        self.side_panel.addTab(self.detail_panel, "详情")
        self.side_panel.setMinimumWidth(250)
        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self.setStyleSheet("background-color: #0a0a0f; color: #d0d0d8;")

        # LogWidget 叠加在 GLWidget 左下角（透明，docs §4.3）
        self.log_widget = LogWidget(self.gl_widget)
        self.gl_widget.overlay_widgets.append(self.log_widget)
        self.log_widget.show()

    # ------------------------------------------------------------------
    # 主题菜单
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
            self._theme_menu.addAction(act)
            self._theme_actions[tid] = act

    def _switch_theme(self, theme_id: str) -> None:
        tm = self.gl_widget.theme_manager
        if tm is None:
            return
        # 菜单槽触发时 GL context 不保证 current：整个 switch（编译新 program
        # + 回调里重建 VAO / 写 uniform）都必须包在 makeCurrent() 内，否则
        # 在错误/无上下文下创建的对象会导致地形渲染变黑（2026-08-13 修复）。
        self.gl_widget.makeCurrent()
        try:
            theme = tm.switch(theme_id)
        finally:
            self.gl_widget.doneCurrent()
        for tid, act in self._theme_actions.items():
            act.setChecked(tid == theme.id)
        self.statusBar().showMessage(f"主题已切换：{theme.name}", 3000)

    # ------------------------------------------------------------------
    # 叙事交互
    # ------------------------------------------------------------------

    def _skip_dialogue(self) -> None:
        """跳过当前对话，恢复模拟。"""
        state = self.narrative.skip_active_dialogue()
        self._paused = False
        self.narrative_overlay.update_state(state)
        self.statusBar().showMessage("对话已跳过，模拟恢复", 3000)

    def _on_option_selected(self, dialogue_id: str, option_index: int) -> None:
        """对话选项点击 → NarrativeEngine.on_player_choice（docs §2.1）。"""
        state = self.narrative.on_player_choice(dialogue_id, option_index)
        self._paused = state.paused
        self.narrative_overlay.update_state(state)

    # ------------------------------------------------------------------
    # Phase 5e：键盘快捷键（ui/controls.py 映射 → 动作分发）
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        """GLWidget 按键过滤：ControlMap 映射动作，命中则分发并消费。

        Esc（取消选中）不在此拦截——GLWidget 原有的 keyPressEvent 处理，
        返回 False 让事件继续传播。
        """
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
        """按动作名分发到具体处理槽。"""
        if action == ControlMap.TOGGLE_PAUSE:
            self._manual_paused = not self._manual_paused
            self.statusBar().showMessage(
                "⏸ 手动暂停" if self._manual_paused else "▶ 恢复运行", 2000
            )
        elif action == ControlMap.ADVANCE_DIALOGUE:
            if self.narrative.has_active_dialogue:
                state = self.narrative.advance_dialogue(None)
                self._paused = state.paused
                self.narrative_overlay.update_state(state)
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
        """快捷键选择对话选项（等价点击选项按钮 → Command 入队）。"""
        did = self._active_dialogue_id
        if did is None:
            return
        state = self.narrative.on_player_choice(did, option_index)
        self._paused = state.paused
        self.narrative_overlay.update_state(state)

    # ------------------------------------------------------------------
    # 面板数据源（消费 narrative/recorder；延迟防御：_build_ui 阶段
    # narrative 尚未创建，此时返回空）
    # ------------------------------------------------------------------

    def _recent_events_for(self, armor_idx: int) -> list[str]:
        """某单位最近事件摘要（DetailPanel 级别 2）。"""
        rec = getattr(self, "narrative", None)
        if rec is None or rec.recorder is None:
            return []
        r = rec.recorder
        evs = r.query(target_id=armor_idx, limit=10) + r.query(source_id=armor_idx, limit=10)
        evs.sort(key=lambda e: e.sim_time)
        return [
            f"[{e.sim_time:.1f}s] {_KIND_CN.get(e.event_type, e.event_type)}"
            for e in evs[-5:]
        ]

    def _timeline_for(self) -> list[str]:
        """事件时间线（ArchivePanel）。"""
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
        """生成战斗简报（BriefingPanel）。"""
        rec = getattr(self, "narrative", None)
        if rec is None:
            return None
        return rec.generate_briefing()

    # ------------------------------------------------------------------
    # 主循环（docs §2.3 数据流）
    # ------------------------------------------------------------------

    def _on_frame(self) -> None:
        # 1. 模拟层推进（对话暂停 / 手动暂停 / 模拟暂停时冻结）
        if (
            not (self._manual_paused or self._paused)
            and not self.simulation.sim_paused
        ):
            self.simulation.step(1 / 60)

        # 2. 孪生适配器（Phase 5 透传；Phase 8 传感器融合）
        events = self.simulation.events.drain()
        snapshot = self.simulation.state.snapshot()
        # Phase 5e：渲染层位置差分速度估计（EMA 平滑，不动引擎）
        armor = snapshot.pools.get("armor")
        speed = self._estimate_speed(armor.pos if armor is not None else None)
        self.adapter = WorldAdapter.from_snapshot(snapshot, armor_speed=speed)

        # 3. 叙事层消费
        narrative_state = self.narrative.tick(snapshot, events)
        self._paused = narrative_state.paused
        d = narrative_state.active_dialogue
        self._active_dialogue_id = (
            d.dialogue_id if d is not None and not d.is_finished else None
        )

        # 4. UI 刷新（注入 adapter，永不注入原始 snapshot）
        self.gl_widget.set_adapter(self.adapter)
        self.detail_panel.set_adapter(self.adapter)
        self.narrative_overlay.update_state(narrative_state)
        if events:
            event_dicts = [
                {"time": snapshot.sim_time, "kind": ev.kind, "data": ev.data}
                for ev in events
            ]
            self.log_widget.append_events(event_dicts)
            # Phase 5e：弹道线 / 命中标记特效缓冲（与日志同一份事件）
            self.gl_widget.feed_effects(event_dicts)

        # 5. 渲染刷新
        self.gl_widget.update()

        # FPS 统计
        self._frame_count += 1
        now = time.monotonic()
        if now - self._fps_clock >= 1.0:
            self._fps = self._frame_count / (now - self._fps_clock)
            self._frame_count = 0
            self._fps_clock = now

    # ------------------------------------------------------------------
    # Phase 5e：渲染层速度估计（位置差分 + EMA，不动引擎）
    # ------------------------------------------------------------------

    def _estimate_speed(self, pos: np.ndarray | None) -> np.ndarray | None:
        """由相邻两帧位置差分估计水平速度（m/s），EMA 平滑。

        返回与 pos 等长的 float32 数组；首帧 / 形状变化时返回 0 并重置。
        对话/手动暂停期间位置不变 → 速度自然归 0。
        """
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
        raw = (delta / (1.0 / 60.0)).astype(np.float32)  # step(1/60) 每帧
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
        if self._manual_paused:
            state = "⏸ 手动暂停"
        elif self._paused:
            state = "⏸ 对话暂停"
        else:
            state = "▶ 运行中"
        n_units = self.adapter.unit_count() if self.adapter else 0
        self.statusBar().showMessage(
            f"{state} | 模拟 t={sim.sim_time:6.1f}s tick={sim.tick_count} | "
            f"FPS={self._fps:5.1f} | 单位={n_units} | 主题={theme_name} | "
            f"相机 d={cam.distance:.0f} yaw={cam.yaw:.0f}° pitch={cam.pitch:.0f}°"
        )