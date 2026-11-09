"""战斗事件记录 + 简报生成（AGENTS.md Phase 4：narrative/recorder.py）。

对应 docs/phase4-narrative-design.md §3.6 + §九：
    - RingBuffer：内存环形缓冲区（最近 1000 条）
    - CombatEventRecord：一条事件记录（与 SQLite combat_events 表对应）
    - CombatEventRecorder：环形缓冲区 + SQLite 持久化 + 简报生成
    - BriefingGenerator：多维度聚合（战果统计 / 时间线 / 单位战报 / 关键事件）
    - CombatBriefing / UnitStatsReport：渲染层消费的数据模型

设计原则（docs §9）：
    - 事件驱动 + 数据与表现分离（参考 CMO Message Log + DCS Debrief）
    - on_events() 只写环形缓冲区（O(1)，不阻塞 Tick）；flush() 批量写 SQLite
    - 战争迷雾：observed_by_player 标记，简报只展示已观测事件
    - combat_events 表只 INSERT 不 UPDATE/DELETE（事件日志语义）

SQLite 使用 stdlib sqlite3（无第三方依赖）。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Airia.lulilo.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 事件记录模型
# ---------------------------------------------------------------------------


@dataclass
class CombatEventRecord:
    """一条战斗事件记录（与 SQLite combat_events 表一一对应，docs §9.1）。"""

    id: int = -1                    # SQLite 自增主键（内存中为 -1）
    sim_time: float = 0.0
    tick_count: int = 0
    event_type: str = ""            # fire | hit | destroyed | detection | command | miss
    source_id: int = -1             # 发起者 armor_idx（-1=系统/玩家）
    source_name: str = ""
    source_side: str = ""           # ally | enemy | neutral
    target_id: int = -1
    target_name: str = ""
    target_side: str = ""
    weapon_id: str = ""
    ammo_id: str = ""
    hit_part: str = ""              # HULL | TURRET | ENGINE | TRACK | AMMO_RACK
    penetration: bool = False
    pen_mm: float = 0.0
    angle_deg: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    extra_json: str = "{}"          # 扩展字段 JSON
    observed_by_player: int = 1     # 0=未观测 / 1=已观测 / 2=推测


# ---------------------------------------------------------------------------
# 环形缓冲区
# ---------------------------------------------------------------------------


class RingBuffer:
    """固定容量环形缓冲区（Python list 实现，docs §9.3）。"""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self._buf: list[CombatEventRecord] = []
        self._capacity = capacity

    def append(self, record: CombatEventRecord) -> None:
        if len(self._buf) >= self._capacity:
            self._buf.pop(0)  # 丢弃最旧
        self._buf.append(record)

    def drain(self) -> list[CombatEventRecord]:
        """取走全部并清空（写入 SQLite 前 drain）。"""
        records, self._buf = self._buf, []
        return records

    def peek(self) -> list[CombatEventRecord]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)


# ---------------------------------------------------------------------------
# 单位统计 / 简报数据模型
# ---------------------------------------------------------------------------


@dataclass
class UnitStatsReport:
    """单位服役记录（docs §9.6）。"""

    armor_idx: int = -1
    unit_name: str = ""
    side: str = ""
    shots_fired: int = 0
    hits_landed: int = 0
    hits_received: int = 0
    kills: int = 0
    destroyed: bool = False
    destroyed_by: str = ""
    ammo_consumed: float = 0.0
    survival_time: float = 0.0
    notable_events: list[str] = field(default_factory=list)


@dataclass
class CombatBriefing:
    """战斗简报（BriefingGenerator 生成，渲染层消费，docs §9.6）。"""

    mission_id: str = ""
    duration_s: float = 0.0
    outcome: str = "stalemate"          # victory / defeat / stalemate

    allies_initial: int = 0
    allies_remaining: int = 0
    allies_lost: int = 0
    enemies_initial: int = 0
    enemies_destroyed: int = 0
    enemies_remaining: int = 0
    total_shots_fired: int = 0
    total_ammo_consumed: float = 0.0

    timeline: list[str] = field(default_factory=list)
    unit_reports: list[UnitStatsReport] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    friendly_fire_incidents: list[str] = field(default_factory=list)
    arcadia_notes: str = ""

    def to_text(self) -> str:
        """渲染为纯文本（控制台 / .txt 导出）。"""
        lines: list[str] = []
        lines.append(f"===== 战斗简报: {self.mission_id} =====")
        lines.append(f"耗时 {self.duration_s:.1f}s | 结果: {self.outcome}")
        lines.append(
            f"我方: {self.allies_remaining}/{self.allies_initial} 存活 "
            f"(损失 {self.allies_lost})"
        )
        lines.append(
            f"敌方: 击毁 {self.enemies_destroyed}, 剩余 {self.enemies_remaining}"
        )
        lines.append(f"总开火: {self.total_shots_fired} | 弹药消耗: {self.total_ammo_consumed:.0f}")
        if self.highlights:
            lines.append("\n[关键事件]")
            lines.extend(f"  - {h}" for h in self.highlights)
        if self.friendly_fire_incidents:
            lines.append("\n[友军误击!]")
            lines.extend(f"  - {h}" for h in self.friendly_fire_incidents)
        if self.timeline:
            lines.append("\n[时间线]")
            lines.extend(f"  {t}" for t in self.timeline[:50])
        if self.arcadia_notes:
            lines.append(f"\n[Arcadia] {self.arcadia_notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 简报生成器（纯函数，无状态）
# ---------------------------------------------------------------------------


class BriefingGenerator:
    """简报生成器：从事件列表生成 CombatBriefing（docs §9.5）。

    设计原则：
        - 多维度聚合：战果统计 + 时间线 + 单位战报 + 关键事件高亮
        - 战争迷雾：observed_only=True 时过滤未观测（=0）事件
        - Arcadia 认知：subjective_notes 注入主观解读
    """

    def __init__(self) -> None:
        self._part_names = {
            0: "车体", 1: "炮塔", 2: "引擎", 3: "履带", 4: "弹药架",
            "HULL": "车体", "TURRET": "炮塔", "ENGINE": "引擎",
            "TRACK": "履带", "AMMO_RACK": "弹药架",
        }

    def generate(
        self,
        events: list[CombatEventRecord],
        unit_names: dict[int, str],
        duration_s: float,
        mission_id: str = "",
        observed_only: bool = True,
        subjective_notes: str = "",
        side_map: dict[int, str] | None = None,
    ) -> CombatBriefing:
        """生成一份战斗简报。

        Args:
            events: 全部事件（已从 SQLite/缓冲区读取）。
            unit_names: armor_idx → 单位名。
            duration_s: 战斗耗时。
            mission_id: 任务标识。
            observed_only: 是否只统计玩家已观测事件（战争迷雾）。
            subjective_notes: Arcadia 主观解读文本（可选）。
            side_map: armor_idx → 阵营（未在事件中出现单位的阵营推断用）。
        """
        side_map = side_map or {}
        if observed_only:
            events = [e for e in events if e.observed_by_player >= 1]

        b = CombatBriefing(
            mission_id=mission_id,
            duration_s=duration_s,
            arcadia_notes=subjective_notes,
        )

        # 战果统计
        allies_seen: set[int] = set()
        enemies_seen: set[int] = set()
        destroyed_units: set[int] = set()
        for e in events:
            if e.source_side == "ally" and e.source_id >= 0:
                allies_seen.add(e.source_id)
            elif e.source_side == "enemy" and e.source_id >= 0:
                enemies_seen.add(e.source_id)
            if e.target_side == "ally" and e.target_id >= 0:
                allies_seen.add(e.target_id)
            elif e.target_side == "enemy" and e.target_id >= 0:
                enemies_seen.add(e.target_id)
            if e.event_type == "destroyed":
                destroyed_units.add(e.target_id)
                if e.target_side == "enemy":
                    b.enemies_destroyed += 1
                elif e.target_side == "ally":
                    b.allies_lost += 1
            if e.event_type == "fire":
                b.total_shots_fired += 1

        # 用 unit_names + side_map 推导初始数量（fallback：事件中出现的单位）
        for idx in unit_names:
            side = self._side_of(events, idx, side_map)
            if side == "ally":
                allies_seen.add(idx)
            elif side == "enemy":
                enemies_seen.add(idx)

        b.allies_initial = len(allies_seen)
        b.enemies_initial = len(enemies_seen)
        b.allies_remaining = max(0, b.allies_initial - b.allies_lost)
        b.enemies_remaining = max(0, b.enemies_initial - b.enemies_destroyed)

        # 时间线
        b.timeline = self._build_timeline(events, unit_names)

        # 单位战报
        b.unit_reports = self._per_unit_stats(events, unit_names, duration_s)

        # 关键事件高亮 + 友军误击
        b.highlights = self._detect_highlights(events, unit_names)
        b.friendly_fire_incidents = self._detect_friendly_fire(events, unit_names)

        # 结局判定
        b.outcome = self._outcome(b)
        return b

    # ------------------------------------------------------------------
    # 内部聚合
    # ------------------------------------------------------------------

    @staticmethod
    def _side_of(
        events: list[CombatEventRecord],
        idx: int,
        side_map: dict[int, str] | None = None,
    ) -> str:
        """推断某单位的阵营：优先 side_map，其次从事件流推断。"""
        if side_map and idx in side_map:
            return side_map[idx]
        for e in events:
            if e.source_id == idx and e.source_side:
                return e.source_side
            if e.target_id == idx and e.target_side:
                return e.target_side
        return ""

    def _build_timeline(
        self, events: list[CombatEventRecord], unit_names: dict[int, str]
    ) -> list[str]:
        """时间线：按 sim_time 排序的关键事件摘要文本。"""
        lines: list[str] = []
        for e in sorted(events, key=lambda x: x.sim_time):
            t = f"T+{e.sim_time:6.1f}s"
            if e.event_type == "fire":
                src = self._name(e.source_id, unit_names, e.source_name)
                tgt = self._name(e.target_id, unit_names, e.target_name)
                lines.append(f"{t} {src} 向 {tgt} 开火（{e.ammo_id or e.weapon_id}）")
            elif e.event_type == "hit":
                src = self._name(e.source_id, unit_names, e.source_name)
                tgt = self._name(e.target_id, unit_names, e.target_name)
                part = self._part(e.hit_part)
                pen = "穿透" if e.penetration else "未击穿"
                lines.append(f"{t} {src} 命中 {tgt}（{part}，{pen}，{e.pen_mm:.0f}mm）")
            elif e.event_type == "destroyed":
                tgt = self._name(e.target_id, unit_names, e.target_name)
                cause = self._cause_text(e.extra_json)
                lines.append(f"{t} {tgt} 被击毁（{cause}）")
            elif e.event_type == "detection":
                src = self._name(e.source_id, unit_names, e.source_name)
                tgt = self._name(e.target_id, unit_names, e.target_name)
                lines.append(f"{t} {src} 探测到 {tgt}")
            elif e.event_type == "command":
                lines.append(f"{t} 指令: {e.extra_json}")
            elif e.event_type == "miss":
                src = self._name(e.source_id, unit_names, e.source_name)
                lines.append(f"{t} {src} 脱靶")
        return lines

    def _per_unit_stats(
        self,
        events: list[CombatEventRecord],
        unit_names: dict[int, str],
        duration_s: float,
    ) -> list[UnitStatsReport]:
        """每个单位的服役记录。"""
        units: dict[int, UnitStatsReport] = {}
        for e in events:
            if e.source_id >= 0:
                r = units.setdefault(
                    e.source_id,
                    UnitStatsReport(
                        armor_idx=e.source_id,
                        unit_name=e.source_name or unit_names.get(e.source_id, f"#{e.source_id}"),
                        side=e.source_side,
                    ),
                )
                if e.event_type == "fire":
                    r.shots_fired += 1
                elif e.event_type == "hit" and e.penetration:
                    r.hits_landed += 1
                elif e.event_type == "destroyed":
                    r.kills += 1
            if e.target_id >= 0:
                r = units.setdefault(
                    e.target_id,
                    UnitStatsReport(
                        armor_idx=e.target_id,
                        unit_name=e.target_name or unit_names.get(e.target_id, f"#{e.target_id}"),
                        side=e.target_side,
                    ),
                )
                if e.event_type == "hit":
                    r.hits_received += 1
                if e.event_type == "destroyed":
                    r.destroyed = True
                    r.destroyed_by = self._name(
                        e.source_id, unit_names, e.source_name
                    ) or "未知"
                    r.survival_time = e.sim_time
        for r in units.values():
            if not r.destroyed:
                r.survival_time = duration_s
        return list(units.values())

    def _detect_highlights(
        self, events: list[CombatEventRecord], unit_names: dict[int, str]
    ) -> list[str]:
        """关键事件高亮：首次命中 / 首次击毁 / 弹药殉爆等。"""
        highlights: list[str] = []
        first_hit = False
        first_kill = False
        for e in sorted(events, key=lambda x: x.sim_time):
            if e.event_type == "hit" and not first_hit:
                src = self._name(e.source_id, unit_names, e.source_name)
                tgt = self._name(e.target_id, unit_names, e.target_name)
                highlights.append(f"首次命中: {src} → {tgt}")
                first_hit = True
            if e.event_type == "destroyed" and not first_kill:
                tgt = self._name(e.target_id, unit_names, e.target_name)
                src = self._name(e.source_id, unit_names, e.source_name)
                cause = e.extra_json
                highlights.append(f"首次击毁: {tgt} 被 {src} 击毁（{cause}）")
                first_kill = True
            if e.event_type == "destroyed" and "cookoff" in e.extra_json:
                tgt = self._name(e.target_id, unit_names, e.target_name)
                highlights.append(f"弹药殉爆: {tgt} 弹药架被引爆！")
        return highlights

    def _detect_friendly_fire(
        self, events: list[CombatEventRecord], unit_names: dict[int, str]
    ) -> list[str]:
        """友军误击检测：ally 打 ally。"""
        incidents: list[str] = []
        for e in events:
            if (
                e.event_type in ("fire", "hit", "destroyed")
                and e.source_side == "ally"
                and e.target_side == "ally"
                and e.source_id != e.target_id
            ):
                src = self._name(e.source_id, unit_names, e.source_name)
                tgt = self._name(e.target_id, unit_names, e.target_name)
                incidents.append(f"{src} → {tgt}（{e.event_type}）")
        return incidents

    def _outcome(self, b: CombatBriefing) -> str:
        """结局判定：全歼敌方 = victory；我方全灭 = defeat；否则 stalemate。"""
        if b.enemies_initial > 0 and b.enemies_destroyed >= b.enemies_initial:
            return "victory"
        if b.allies_initial > 0 and b.allies_remaining <= 0:
            return "defeat"
        return "stalemate"

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _name(idx: int, unit_names: dict[int, str], fallback: str) -> str:
        if fallback:
            return fallback
        return unit_names.get(idx, f"#{idx}")

    @staticmethod
    def _cause_text(extra_json: str) -> str:
        """从 extra_json 提取可读的击毁原因。"""
        cause_names = {
            "health": "结构损毁",
            "cookoff": "弹药殉爆",
            "destroyed": "被击毁",
            "burnout": "起火焚毁",
        }
        try:
            d = json.loads(extra_json or "{}")
        except (json.JSONDecodeError, TypeError):
            return extra_json or "未知"
        cause = d.get("cause", "")
        return cause_names.get(cause, cause or "未知")

    def _part(self, part: str | int) -> str:
        return self._part_names.get(part, str(part))


# ---------------------------------------------------------------------------
# CombatEventRecorder（环形缓冲区 + SQLite + 简报生成）
# ---------------------------------------------------------------------------


class CombatEventRecorder:
    """战斗事件记录器。

    用法（docs §9.4）:
        recorder = CombatEventRecorder("logs/mission_001.db", unit_registry=..., side_map=...)
        recorder.on_events(events, snapshot)   # 每帧 drain 后调用
        briefing = recorder.generate_briefing()  # 任务结束生成简报
        recorder.close()

    SQLite 结构（docs §9.2）：
        combat_events   事件日志（不可变追加）
        mission_summary 任务摘要（简报 JSON 缓存）
        unit_stats      单位统计（任务结束聚合写入）
    """

    # 事件 → 列名映射（EventBus data 字段 → CombatEventRecord 字段）
    _DATA_MAP = {
        "fire": {"shooter": "source_id", "target": "target_id"},
        "hit": {"shooter": "source_id", "target": "target_id"},
        "destroyed": {"armor_idx": "target_id"},
        "miss": {"shooter": "source_id"},
        "detection": {"source": "source_id", "target": "target_id"},
    }

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        ring_size: int = 1000,
        unit_registry: dict[int, str] | None = None,
        side_map: dict[int, str] | None = None,
        mission_id: str = "mission_001",
        *,
        flush_every: int = 256,
    ) -> None:
        """初始化记录器。

        Args:
            db_path: SQLite 路径（":memory:" = 仅内存，不落盘）。
            ring_size: 环形缓冲区容量。
            unit_registry: armor_idx → 单位名。
            side_map: armor_idx → "ally"/"enemy"/"neutral"。
            mission_id: 任务标识。
            flush_every: 缓冲区累计多少条后自动 flush 到 SQLite。
        """
        self.db_path = str(db_path)
        self.mission_id = mission_id
        self.unit_registry: dict[int, str] = dict(unit_registry or {})
        self.side_map: dict[int, str] = dict(side_map or {})
        self._ring = RingBuffer(ring_size)
        self._flush_every = max(1, flush_every)
        self._pending_count = 0
        # 击杀归属追踪：target_id → 最近一次命中它的 source_id
        # （模拟层 destroyed 事件不含 shooter，需从 hit 事件推断，参考 DCS 最后一击者）
        self._last_hitter: dict[int, int] = {}
        self._conn: sqlite3.Connection | None = None
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """建表（docs §9.2 SQL）。"""
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS combat_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_time        REAL NOT NULL,
                tick_count      INTEGER NOT NULL,
                event_type      TEXT NOT NULL,
                source_id       INTEGER DEFAULT -1,
                source_name     TEXT DEFAULT '',
                source_side     TEXT DEFAULT '',
                target_id       INTEGER DEFAULT -1,
                target_name     TEXT DEFAULT '',
                target_side     TEXT DEFAULT '',
                weapon_id       TEXT DEFAULT '',
                ammo_id         TEXT DEFAULT '',
                hit_part        TEXT DEFAULT '',
                penetration     INTEGER DEFAULT 0,
                pen_mm          REAL DEFAULT 0.0,
                angle_deg       REAL DEFAULT 0.0,
                pos_x           REAL DEFAULT 0.0,
                pos_y           REAL DEFAULT 0.0,
                pos_z           REAL DEFAULT 0.0,
                extra_json      TEXT DEFAULT '{}',
                observed_by_player INTEGER DEFAULT 1,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_time   ON combat_events(sim_time);
            CREATE INDEX IF NOT EXISTS idx_events_type   ON combat_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_source ON combat_events(source_id);
            CREATE INDEX IF NOT EXISTS idx_events_target ON combat_events(target_id);

            CREATE TABLE IF NOT EXISTS mission_summary (
                id              INTEGER PRIMARY KEY,
                mission_id      TEXT NOT NULL,
                start_time      REAL,
                end_time        REAL,
                total_ticks     INTEGER,
                outcome         TEXT,
                summary_json    TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS unit_stats (
                id              INTEGER PRIMARY KEY,
                mission_id      TEXT NOT NULL,
                armor_idx       INTEGER NOT NULL,
                unit_name       TEXT,
                side            TEXT,
                shots_fired     INTEGER DEFAULT 0,
                hits_landed     INTEGER DEFAULT 0,
                hits_received   INTEGER DEFAULT 0,
                kills           INTEGER DEFAULT 0,
                destroyed       INTEGER DEFAULT 0,
                destroyed_by    TEXT DEFAULT '',
                ammo_consumed   REAL DEFAULT 0.0,
                fuel_consumed   REAL DEFAULT 0.0,
                survival_time   REAL,
                UNIQUE(mission_id, armor_idx)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库（自动 flush 残余）。"""
        if self._conn is not None:
            try:
                self.flush()
            finally:
                self._conn.close()
                self._conn = None

    def reset(self) -> None:
        """清空全部事件（新任务开始，docs §3.6）。"""
        self._ring.drain()
        self._pending_count = 0
        self._last_hitter.clear()
        cur = self._conn.cursor()
        cur.execute("DELETE FROM combat_events")
        cur.execute("DELETE FROM mission_summary")
        cur.execute("DELETE FROM unit_stats")
        self._conn.commit()

    # ------------------------------------------------------------------
    # 事件写入
    # ------------------------------------------------------------------

    def on_events(self, events: list[Any], snapshot: Any | None = None) -> int:
        """每帧调用：EventBus 事件 → CombatEventRecord → 环形缓冲区。

        Args:
            events: EventBus.drain() 返回的 SimulationEvent 列表。
            snapshot: 模拟快照（可选，用于补全位置/名称）。

        Returns:
            写入的事件条数。
        """
        if not events:
            return 0

        sim_time = float(getattr(snapshot, "sim_time", 0.0)) if snapshot else 0.0
        tick = int(getattr(snapshot, "tick_count", 0)) if snapshot else 0

        for ev in events:
            kind = getattr(ev, "kind", None)
            if not kind:
                continue
            data = dict(getattr(ev, "data", {}) or {})
            rec = self._to_record(kind, data, sim_time, tick, snapshot)
            if rec is not None:
                self._ring.append(rec)
                self._pending_count += 1

        # 自动 flush（达到阈值）
        if self._pending_count >= self._flush_every:
            self.flush()
        return len(events)

    def _to_record(
        self,
        kind: str,
        data: dict[str, Any],
        sim_time: float,
        tick: int,
        snapshot: Any | None,
    ) -> CombatEventRecord | None:
        """SimulationEvent → CombatEventRecord（字段映射 + 名称/阵营补全）。"""
        rec = CombatEventRecord(sim_time=sim_time, tick_count=tick, event_type=kind)

        # 位置（从 snapshot 的 armor pool 读取）
        if snapshot is not None:
            self._fill_pos(rec, data, snapshot)

        if kind == "fire":
            rec.source_id = int(data.get("shooter", -1))
            rec.target_id = int(data.get("target", -1))
            rec.weapon_id = str(data.get("weapon", ""))
            rec.ammo_id = str(data.get("ammo", ""))
        elif kind == "hit":
            rec.source_id = int(data.get("shooter", -1))
            rec.target_id = int(data.get("target", -1))
            rec.weapon_id = str(data.get("weapon", ""))
            rec.ammo_id = str(data.get("ammo", ""))
            rec.hit_part = self._part_str(data.get("part"))
            rec.penetration = bool(data.get("penetrated", False))
            rec.pen_mm = float(data.get("armor_eff", 0.0))
            rec.angle_deg = float(data.get("margin", 0.0))
            # 更新击杀归属追踪（最近命中者）
            if rec.target_id >= 0 and rec.source_id >= 0:
                self._last_hitter[rec.target_id] = rec.source_id
        elif kind == "destroyed":
            rec.target_id = int(data.get("armor_idx", -1))
            # 击杀归属：优先事件自带 shooter，否则用最近命中者（_last_hitter）
            shooter = data.get("shooter", -1)
            if shooter is None or int(shooter) < 0:
                shooter = self._last_hitter.get(rec.target_id, -1)
            rec.source_id = int(shooter)
            rec.extra_json = json.dumps({"cause": str(data.get("cause", ""))}, ensure_ascii=False)
        elif kind == "miss":
            rec.source_id = int(data.get("shooter", -1))
        elif kind == "detection":
            rec.source_id = int(data.get("source", -1))
            rec.target_id = int(data.get("target", -1))
            rec.extra_json = json.dumps(
                {"sensor": str(data.get("sensor", ""))}, ensure_ascii=False
            )
        elif kind == "command":
            rec.target_id = int(data.get("target", -1))
            rec.extra_json = json.dumps(
                {"type": str(data.get("type", "")), "params": data.get("params", {})},
                ensure_ascii=False,
            )
        else:
            # 未知事件类型：写入 extra（保留原始数据）
            rec.extra_json = json.dumps(data, ensure_ascii=False, default=str)

        # 名称 / 阵营补全
        self._fill_names(rec)
        return rec

    def _fill_pos(
        self, rec: CombatEventRecord, data: dict[str, Any], snapshot: Any
    ) -> None:
        """从 snapshot 的 armor pool 读取事件位置。"""
        pool = getattr(snapshot, "pools", {}).get("armor") if hasattr(snapshot, "pools") else None
        if pool is None:
            return
        idx = rec.source_id if rec.source_id >= 0 else rec.target_id
        if idx >= 0 and idx < len(getattr(pool, "pos", ())):
            p = pool.pos[idx]
            rec.pos_x = float(p[0])
            rec.pos_y = float(p[1])
            rec.pos_z = float(p[2])

    def _fill_names(self, rec: CombatEventRecord) -> None:
        """补全 source/target 的名称与阵营。"""
        if rec.source_id >= 0:
            rec.source_name = self.unit_registry.get(rec.source_id, rec.source_name)
            rec.source_side = self.side_map.get(rec.source_id, rec.source_side)
        if rec.target_id >= 0:
            rec.target_name = self.unit_registry.get(rec.target_id, rec.target_name)
            rec.target_side = self.side_map.get(rec.target_id, rec.target_side)

    @staticmethod
    def _part_str(part: Any) -> str:
        """命中部位 → 字符串（int 枚举 → 名称）。"""
        names = {0: "HULL", 1: "TURRET", 2: "ENGINE", 3: "TRACK", 4: "AMMO_RACK"}
        if isinstance(part, int):
            return names.get(part, str(part))
        return str(part or "")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def flush(self) -> int:
        """将环形缓冲区残余批量写入 SQLite（Tick 间隙/阶段切换调用）。

        Returns:
            写入的行数。
        """
        records = self._ring.drain()
        if not records or self._conn is None:
            return 0
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO combat_events (
                sim_time, tick_count, event_type,
                source_id, source_name, source_side,
                target_id, target_name, target_side,
                weapon_id, ammo_id, hit_part,
                penetration, pen_mm, angle_deg,
                pos_x, pos_y, pos_z,
                extra_json, observed_by_player
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r.sim_time, r.tick_count, r.event_type,
                    r.source_id, r.source_name, r.source_side,
                    r.target_id, r.target_name, r.target_side,
                    r.weapon_id, r.ammo_id, r.hit_part,
                    int(r.penetration), r.pen_mm, r.angle_deg,
                    r.pos_x, r.pos_y, r.pos_z,
                    r.extra_json, r.observed_by_player,
                )
                for r in records
            ],
        )
        self._conn.commit()
        self._pending_count = 0
        return len(records)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query(
        self,
        event_type: str | None = None,
        source_id: int | None = None,
        target_id: int | None = None,
        time_range: tuple[float, float] | None = None,
        observed_only: bool = True,
        limit: int = 500,
    ) -> list[CombatEventRecord]:
        """从 SQLite 查询事件（docs §3.6 query）。

        查询前自动 flush 环形缓冲区，保证未落盘的事件也能查到。
        """
        self.flush()
        sql = "SELECT * FROM combat_events WHERE 1=1"
        params: list[Any] = []
        if event_type:
            sql += " AND event_type=?"
            params.append(event_type)
        if source_id is not None:
            sql += " AND source_id=?"
            params.append(source_id)
        if target_id is not None:
            sql += " AND target_id=?"
            params.append(target_id)
        if time_range:
            sql += " AND sim_time>=? AND sim_time<=?"
            params.extend(time_range)
        if observed_only:
            sql += " AND observed_by_player>=1"
        sql += " ORDER BY sim_time LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def timeline(self, observed_only: bool = True) -> list[CombatEventRecord]:
        """按时间排序的完整事件时间线。"""
        return self.query(observed_only=observed_only, limit=100000)

    def all_events(self) -> list[CombatEventRecord]:
        """读取全部事件（含未观测，供简报生成器过滤）。"""
        return self.query(observed_only=False, limit=100000)

    def _row_to_record(self, row: tuple) -> CombatEventRecord:
        cols = [
            "id", "sim_time", "tick_count", "event_type",
            "source_id", "source_name", "source_side",
            "target_id", "target_name", "target_side",
            "weapon_id", "ammo_id", "hit_part",
            "penetration", "pen_mm", "angle_deg",
            "pos_x", "pos_y", "pos_z",
            "extra_json", "observed_by_player", "created_at",
        ]
        d = dict(zip(cols, row))
        return CombatEventRecord(
            id=int(d["id"]),
            sim_time=float(d["sim_time"]),
            tick_count=int(d["tick_count"]),
            event_type=str(d["event_type"]),
            source_id=int(d["source_id"]),
            source_name=str(d["source_name"]),
            source_side=str(d["source_side"]),
            target_id=int(d["target_id"]),
            target_name=str(d["target_name"]),
            target_side=str(d["target_side"]),
            weapon_id=str(d["weapon_id"]),
            ammo_id=str(d["ammo_id"]),
            hit_part=str(d["hit_part"]),
            penetration=bool(d["penetration"]),
            pen_mm=float(d["pen_mm"]),
            angle_deg=float(d["angle_deg"]),
            pos_x=float(d["pos_x"]),
            pos_y=float(d["pos_y"]),
            pos_z=float(d["pos_z"]),
            extra_json=str(d["extra_json"]),
            observed_by_player=int(d["observed_by_player"]),
        )

    # ------------------------------------------------------------------
    # 简报生成
    # ------------------------------------------------------------------

    def generate_briefing(self, subjective_notes: str = "") -> CombatBriefing:
        """从事件流生成战斗简报（docs §9.5）。

        流程：
            1. flush() 残余 → SQLite
            2. 读取全部事件
            3. BriefingGenerator 多维度聚合
            4. 写入 mission_summary（缓存）
        """
        self.flush()
        events = self.all_events()
        generator = BriefingGenerator()
        briefing = generator.generate(
            events,
            unit_names=self.unit_registry,
            duration_s=self._duration_s(),
            mission_id=self.mission_id,
            observed_only=True,
            subjective_notes=subjective_notes,
            side_map=self.side_map,
        )
        # 缓存到 mission_summary
        self._save_summary(briefing)
        return briefing

    def _duration_s(self) -> float:
        rows = self._conn.execute(
            "SELECT COALESCE(MAX(sim_time), 0.0) FROM combat_events"
        ).fetchone()
        return float(rows[0] if rows else 0.0)

    def _save_summary(self, briefing: CombatBriefing) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO mission_summary (mission_id, start_time, end_time, outcome, summary_json)
            VALUES (?, 0.0, ?, ?, ?)
            """,
            (
                self.mission_id,
                briefing.duration_s,
                briefing.outcome,
                json.dumps(
                    {
                        "allies_remaining": briefing.allies_remaining,
                        "allies_lost": briefing.allies_lost,
                        "enemies_destroyed": briefing.enemies_destroyed,
                        "enemies_remaining": briefing.enemies_remaining,
                        "timeline": briefing.timeline[:100],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        self._conn.commit()

    def unit_stats(self, armor_idx: int) -> UnitStatsReport | None:
        """查询单个单位的服役记录（从 SQLite unit_stats 或事件聚合）。"""
        # 从事件实时聚合（简单可靠）
        events = self.query(
            source_id=armor_idx, observed_only=False, limit=100000
        ) + self.query(target_id=armor_idx, observed_only=False, limit=100000)
        if not events:
            return None
        gen = BriefingGenerator()
        reports = gen._per_unit_stats(events, self.unit_registry, self._duration_s())
        for r in reports:
            if r.armor_idx == armor_idx:
                return r
        return None
