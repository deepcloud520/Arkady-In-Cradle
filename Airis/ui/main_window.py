"""主窗口（Phase 5 骨架：管线集成 + 主题菜单 + 主循环）。

对应 docs/phase5-ui-pipeline-design.md §1 / §2.3：
    - 水平 QSplitter：GLWidget（左，拉伸 3）+ 占位 SidePanel（右，拉伸 1）
    - 菜单栏：主题切换（ThemeManager.switch 热切换）、叙事（跳过对话）
    - 主循环（QTimer 帧定时器驱动）：
        Simulation.step → WorldAdapter → NarrativeEngine.tick → UI 刷新
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QSplitter, QTabWidget

from Airis.lulilo.coordinates.grid import world_to_grid
from Airis.lulilo.entities.pools import ArmorPool, BoxPool, PartType, ProjectilePool
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.movement.integrate import build_armor_movement_hook
from Airis.lulilo.movement.pathfinding import GraphPathfinder
from Airis.lulilo.simulation.loop import SimulationLoop
from Airis.lulilo.simulation.state import SimulationState
from Airis.lulilo.terrain.dual_contour import DualContourBuilder
from Airis.narrative.engine import NarrativeEngine
from Airis.narrative.recorder import CombatEventRecorder
from Airis.render.adapter import WorldAdapter
from Airis.render.gl_widget import GLWidget
from Airis.render.hud.log_widget import LogWidget
from Airis.ui.archive_panel import ArchivePanel
from Airis.ui.briefing_panel import BriefingPanel
from Airis.ui.detail_panel import DetailPanel
from Airis.ui.narrative_overlay import NarrativeOverlay

logger = logging.getLogger(__name__)

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
        self.setWindowTitle("Arkady in Cradle — Phase 5 骨架")
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

        # ── 主循环（帧定时器）──
        self._paused = False
        self._fps = 0.0
        self._frame_count = 0
        self._fps_clock = time.monotonic()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_frame)
        self._tick_timer.start(16)

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
        """创建模拟层 + 演示编队 + 后勤 + 移动/悬挂管线，返回 (loop, unit_names, side_map)。

        注意：ArmorPool 暂无 side 字段（Phase 8 接入），这里用动态附加
        属性演示敌我着色；WorldAdapter.from_snapshot 通过 getattr 读取。

        🔥 2026-08-13 接入移动+悬挂（docs/suspension-slope-design.md §2）：
        - 逻辑 WorldMesh = DualContourBuilder（连续场平滑 → 地面双线性斜面，
          悬挂 pitch/roll 在 fBm 起伏上生效；build_adjacency=False 省扫描线）
        - 编队设路标点（NavGraph 寻路，失败降级直线）→ movement hook 每帧
          推进：integrate_3d → resolve_mesh → solve_suspension → resolve_contact
        """
        state = SimulationState()
        armor = ArmorPool(capacity=64)
        box = BoxPool(capacity=256)
        proj = ProjectilePool(capacity=128)
        ammo = AmmoManager(capacity=64)
        fuel = FuelManager(capacity=64)
        state.armor_pool = armor
        state.box_pool = box
        state.projectile_pool = proj
        state.ammo = ammo
        state.fuel = fuel
        armor.side = np.zeros(armor.capacity, dtype=np.uint8)   # 动态附加（演示）

        terrain = self.gl_widget.terrain
        hf = terrain.heightfield
        cs = hf.cs

        # 逻辑 WorldMesh（连续场平滑，悬挂/移动 z 采样源；构建较慢，加载期一次）
        logger.info("构建逻辑 WorldMesh（DualContour，约 4s）…")
        world_mesh = DualContourBuilder(
            hf,
            continuous_ground=terrain.continuous_ground,
            build_adjacency=False,  # 邻接仅供寻路（Phase C 未做），省 3s+
        ).build()
        logger.info("逻辑 WorldMesh 就绪：%d 面片", len(world_mesh.patches))
        # 寻路服务（NavGraph 独立于 WorldMesh，0.3s 构建）
        pathfinder = GraphPathfinder(terrain.nav_graph, cell_size=cs)
        # 每车速度（m/s，沿路标点）
        speed = np.zeros(armor.capacity, dtype=np.float32)

        def _ground_z(wx: float, wy: float) -> float:
            gx, gy = world_to_grid(wx, wy, cs)
            gx = min(max(gx, 0), hf.width - 1)
            gy = min(max(gy, 0), hf.height - 1)
            return float(hf.surface_z(gx, gy, 0))

        def _route(
            start_w: tuple[float, float], end_w: tuple[float, float]
        ) -> list[tuple[float, float]]:
            """寻路路标点（≤8 个，wp 上限）；失败降级直线。"""
            pts = pathfinder.find_path(start_w, end_w)
            if not pts:
                pts = [end_w]
            if len(pts) > 8:
                idxs = [int(k * (len(pts) - 1) / 7) for k in range(8)]
                pts = [pts[i] for i in idxs]
            return pts

        def _squad(
            x0: float, y0: float, side: int, yaw: float,
            count: int = 8, target: tuple[float, float] = (128.0, 128.0),
        ) -> int:
            first = -1
            for i in range(count):
                wx = x0 + i * 6
                wy = y0 + i * 6
                aidx = armor.spawn(
                    x=wx, y=wy, z=_ground_z(wx, wy),
                    half_x=1.5, half_y=3.0, half_z=0.8,
                    yaw=yaw,
                )
                armor.side[aidx] = side
                # 后勤演示数据（Phase 5 DetailPanel 级别 0/2 显示）
                ammo.state.set_unit(aidx, cap=40.0, initial=40.0 * (0.8 + 0.2 * i / count))
                fuel.state.set_unit(aidx, cap=100.0, initial=100.0 * (0.85 + 0.15 * i / count))
                # 路标点（寻路 → 悬挂在 fBm 起伏/桥上生效）
                armor.set_waypoints(aidx, _route((wx, wy), target))
                speed[aidx] = 5.0
                if first < 0:
                    first = aidx
                # 车体 + 炮塔两个碰撞箱（Phase 3 部位毁伤演示）
                hull_b = box.spawn(aidx, int(PartType.HULL), 1.5, 3.0, 0.8,
                                   local_x=0.0, local_y=0.0, local_z=0.0)
                armor.bind_hull(aidx, hull_b)  # 绑定 HULL box（part-health-design.md §2.2）
                box.spawn(aidx, int(PartType.TURRET), 1.2, 1.2, 0.7,
                          local_x=0.0, local_y=0.0, local_z=0.6)
            return first

        # 两队相向推进到峡谷长桥两端附近（穿越 fBm 起伏 + 可能上桥）
        _squad(60, 60, 0, 0.0, target=(150.0, 128.0))
        enemy_first = _squad(196, 196, 1, 3.14159, target=(110.0, 128.0))
        box.sync_all_from_parent(armor)

        # 演示损坏部位标记：敌军首车炮塔毁伤（HUD ⚠ 标记 Phase 5 即可生效）
        for bi in range(box.capacity):
            if box.parent[bi] == enemy_first and box.part[bi] == int(PartType.TURRET):
                box.active[bi] = False
                break

        # 单位名称 + 阵营映射（渲染层 DetailPanel / 叙事层 recorder 消费）
        unit_names: dict[int, str] = {}
        side_map: dict[int, str] = {}
        for idx in range(armor.capacity):
            if armor.active[idx]:
                side_int = int(armor.side[idx])
                model = "T-90" if side_int == 0 else "T-72B3"
                unit_names[idx] = f"{model} #{idx:02d}"
                side_map[idx] = "ally" if side_int == 0 else "enemy"

        # 移动+悬挂管线（每帧推进；弹簧 dt=1/60 稳定域内）
        loop = SimulationLoop(state=state)
        loop.attach("movement", build_armor_movement_hook(armor, world_mesh, speed))
        return loop, unit_names, side_map

    # ------------------------------------------------------------------
    # 叙事层加载
    # ------------------------------------------------------------------

    def _load_story(self) -> None:
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
        # 1. 模拟层推进（对话暂停 / 模拟暂停时冻结）
        if not self._paused and not self.simulation.sim_paused:
            self.simulation.step(1 / 60)

        # 2. 孪生适配器（Phase 5 透传；Phase 8 传感器融合）
        events = self.simulation.events.drain()
        snapshot = self.simulation.state.snapshot()
        self.adapter = WorldAdapter.from_snapshot(snapshot)

        # 3. 叙事层消费
        narrative_state = self.narrative.tick(snapshot, events)
        self._paused = narrative_state.paused

        # 4. UI 刷新（注入 adapter，永不注入原始 snapshot）
        self.gl_widget.set_adapter(self.adapter)
        self.detail_panel.set_adapter(self.adapter)
        self.narrative_overlay.update_state(narrative_state)
        if events:
            self.log_widget.append_events(
                [
                    {"time": snapshot.sim_time, "kind": ev.kind, "data": ev.data}
                    for ev in events
                ]
            )

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
    # 状态栏
    # ------------------------------------------------------------------

    def _update_status_bar(self) -> None:
        cam = self.gl_widget.camera
        tm = self.gl_widget.theme_manager
        theme_name = tm.current.name if tm is not None and tm.current else "-"
        sim = self.simulation.state
        state = "⏸ 对话暂停" if self._paused else "▶ 运行中"
        n_units = self.adapter.unit_count() if self.adapter else 0
        self.statusBar().showMessage(
            f"{state} | 模拟 t={sim.sim_time:6.1f}s tick={sim.tick_count} | "
            f"FPS={self._fps:5.1f} | 单位={n_units} | 主题={theme_name} | "
            f"相机 d={cam.distance:.0f} yaw={cam.yaw:.0f}° pitch={cam.pitch:.0f}°"
        )