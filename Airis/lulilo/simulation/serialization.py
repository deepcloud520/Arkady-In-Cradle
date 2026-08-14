"""存档 / 读档（AGENTS.md Phase 3 交付物：simulation/serialization.py）。

序列化：msgpack（快，AGENTS.md 4.4 I/O 与序列化）。
大规模数据（SoA 池）用 numpy 原生 bytes 编码（tobytes + dtype/shape），
避免逐元素 Python 对象开销，float32 精度无损。

设计要点：
    - 存档结构 = 版本号 + sim_time/tick_count + 各池数组 + 后勤状态 + 补给请求
    - numpy 数组 → {dtype, shape, data(bytes)}；读回 np.frombuffer + reshape
    - 恢复对象时 free_stack 由 active 掩码反推（未占用索引），保证 spawn 不复用活跃索引
    - 补给请求（SupplyRequest）→ dict 往返，保持队列语义
    - 验证标准：round-trip 后各数组逐元素相等（无数据丢失）

用法:
    blob = serialize_battlefield(state, armor, boxes, projectiles, ammo, fuel)
    data = deserialize_battlefield(blob)      # 返回 BattlefieldSave
    data.armor.health   # 恢复后的池
"""

from __future__ import annotations

from dataclasses import dataclass, field

import msgpack
import numpy as np

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoState, SupplyRequest
from Airis.lulilo.logistics.fuel import FuelState
from Airis.lulilo.simulation.state import SimulationState

logger = get_logger(__name__)

SAVE_VERSION = 1


# ---------------------------------------------------------------------------
# numpy 数组编码 / 解码
# ---------------------------------------------------------------------------


def encode_ndarray(a: np.ndarray) -> dict:
    """numpy 数组 → msgpack 兼容 dict（dtype/shape/bytes）。"""
    return {
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "data": np.ascontiguousarray(a).tobytes(),
    }


def decode_ndarray(d: dict) -> np.ndarray:
    """msgpack dict → numpy 数组（无损还原）。"""
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"]).copy()


def encode_requests(requests: list[SupplyRequest]) -> list[dict]:
    """补给请求 → dict 列表。"""
    return [
        {
            "target_idx": r.target_idx,
            "amount": r.amount,
            "priority": r.priority,
            "item": r.item,
        }
        for r in requests
    ]


def decode_requests(data: list[dict]) -> list[SupplyRequest]:
    """dict 列表 → 补给请求。"""
    return [SupplyRequest(**d) for d in data]


# ---------------------------------------------------------------------------
# 池序列化辅助
# ---------------------------------------------------------------------------


def _pack_pool_arrays(arrays: dict[str, np.ndarray]) -> dict:
    return {name: encode_ndarray(arr) for name, arr in arrays.items()}


def _restore_armor(data: dict) -> ArmorPool:
    cap = data["capacity"]
    pool = ArmorPool(capacity=cap)
    for name in ("pos", "half", "yaw", "health", "active", "wp", "wp_count", "wp_idx"):
        setattr(pool, name, decode_ndarray(data[name]))
    # Phase 6b：rcs/ecm（旧存档缺失 → 保持构造默认）
    if "rcs" in data:
        pool.rcs = decode_ndarray(data["rcs"])
    if "ecm" in data:
        pool.ecm = decode_ndarray(data["ecm"])
    _rebuild_free_stack(pool, pool.active)
    return pool


def _restore_boxes(data: dict) -> BoxPool:
    cap = data["capacity"]
    pool = BoxPool(capacity=cap)
    for name in ("local", "pos", "half", "parent", "part", "active"):
        setattr(pool, name, decode_ndarray(data[name]))
    _rebuild_free_stack(pool, pool.active)
    return pool


# Phase 6b 制导字段（docs/phase6-guidance-pipeline.md §2.5）
_PROJECTILE_GUIDANCE_FIELDS = (
    "guid_target_pos",
    "guid_law",
    "guid_nav_gain",
    "guid_stage",
    "guid_memory_pos",
    "guid_comm",
    "guid_confidence",
    "guid_override",
    "guid_search_fov_cos",
    "guid_track_fov_cos",
    "guid_filter_quality",
)


def _restore_projectiles(data: dict) -> ProjectilePool:
    cap = data["capacity"]
    pool = ProjectilePool(capacity=cap)
    for name in ("pos", "vel", "radius", "dmg", "kind", "active", "owner", "weapon", "ammo", "age"):
        setattr(pool, name, decode_ndarray(data[name]))
    # Phase 6b：制导字段（旧存档缺失 → 保持构造默认）
    for name in _PROJECTILE_GUIDANCE_FIELDS:
        if name in data:
            setattr(pool, name, decode_ndarray(data[name]))
    _rebuild_free_stack(pool, pool.active)
    return pool


def _restore_emitters(data: dict) -> EmitterPool:
    cap = data["capacity"]
    pool = EmitterPool(capacity=cap)
    for name in (
        "pos",
        "power_db",
        "cur_power",
        "decay_per_s",
        "band",
        "kind",
        "active",
        "parent_kind",
        "parent_idx",
        "normal",
    ):
        setattr(pool, name, decode_ndarray(data[name]))
    _rebuild_free_stack(pool, pool.active)
    # 读档后无战术关机保留（级联失效语义由下一次 sync_from_parent 重建）
    pool._deactivated = set()
    return pool


def _rebuild_free_stack(pool: object, active: np.ndarray) -> None:
    """由 active 掩码重建 free_stack（未占用索引），避免 spawn 复用活跃索引。"""
    pool._free_stack = [int(i) for i in range(active.shape[0]) if not active[i]]


def _restore_ammo(data: dict) -> AmmoState:
    st = AmmoState(capacity=data["capacity"])
    st.capacity = decode_ndarray(data["capacity_arr"])
    st.reserve = decode_ndarray(data["reserve"])
    st.active = decode_ndarray(data["active"])
    return st


def _restore_fuel(data: dict) -> FuelState:
    st = FuelState(capacity=data["capacity"])
    st.capacity = decode_ndarray(data["capacity_arr"])
    st.reserve = decode_ndarray(data["reserve"])
    st.active = decode_ndarray(data["active"])
    return st


# ---------------------------------------------------------------------------
# 存档 / 读档 主入口
# ---------------------------------------------------------------------------


def serialize_battlefield(
    state: SimulationState,
    armor: ArmorPool | None = None,
    boxes: BoxPool | None = None,
    projectiles: ProjectilePool | None = None,
    ammo: AmmoState | None = None,
    fuel: FuelState | None = None,
    ammo_requests: list[SupplyRequest] | None = None,
    fuel_requests: list[SupplyRequest] | None = None,
    emitters: EmitterPool | None = None,
) -> bytes:
    """把战场全部可序列化状态打包为 msgpack bytes。

    Args:
        state: SimulationState（取 sim_time / tick_count）。
        armor / boxes / projectiles / emitters: 各对象池（缺省从 state 读取）。
        ammo / fuel: 后勤状态（缺省从 state 读取）。
        ammo_requests / fuel_requests: 挂起补给请求（缺省为空）。

    Returns:
        msgpack bytes。
    """
    armor = armor or (state.armor_pool if state.armor_pool is not None else None)
    boxes = boxes or (state.box_pool if state.box_pool is not None else None)
    projectiles = projectiles or (
        state.projectile_pool if state.projectile_pool is not None else None
    )
    emitters = emitters or (
        state.emitter_pool if state.emitter_pool is not None else None
    )
    ammo = ammo or (state.ammo.state if state.ammo is not None else None)
    fuel = fuel or (state.fuel.state if state.fuel is not None else None)

    payload: dict = {
        "version": SAVE_VERSION,
        "sim_time": float(state.sim_time),
        "tick_count": int(state.tick_count),
    }

    if armor is not None:
        payload["armor"] = {
            "capacity": armor.capacity,
            **_pack_pool_arrays(
                {
                    "pos": armor.pos,
                    "half": armor.half,
                    "yaw": armor.yaw,
                    "health": armor.health,
                    "active": armor.active,
                    "wp": armor.wp,
                    "wp_count": armor.wp_count,
                    "wp_idx": armor.wp_idx,
                    # Phase 6b：电磁特征
                    "rcs": armor.rcs,
                    "ecm": armor.ecm,
                }
            ),
        }
    if boxes is not None:
        payload["boxes"] = {
            "capacity": boxes.capacity,
            **_pack_pool_arrays(
                {
                    "local": boxes.local,
                    "pos": boxes.pos,
                    "half": boxes.half,
                    "parent": boxes.parent,
                    "part": boxes.part,
                    "active": boxes.active,
                }
            ),
        }
    if projectiles is not None:
        payload["projectiles"] = {
            "capacity": projectiles.capacity,
            **_pack_pool_arrays(
                {
                    "pos": projectiles.pos,
                    "vel": projectiles.vel,
                    "radius": projectiles.radius,
                    "dmg": projectiles.dmg,
                    "kind": projectiles.kind,
                    "active": projectiles.active,
                    "owner": projectiles.owner,
                    "weapon": projectiles.weapon,
                    "ammo": projectiles.ammo,
                    "age": projectiles.age,
                    # Phase 6b：制导字段
                    "guid_target_pos": projectiles.guid_target_pos,
                    "guid_law": projectiles.guid_law,
                    "guid_nav_gain": projectiles.guid_nav_gain,
                    "guid_stage": projectiles.guid_stage,
                    "guid_memory_pos": projectiles.guid_memory_pos,
                    "guid_comm": projectiles.guid_comm,
                    "guid_confidence": projectiles.guid_confidence,
                    "guid_override": projectiles.guid_override,
                    "guid_search_fov_cos": projectiles.guid_search_fov_cos,
                    "guid_track_fov_cos": projectiles.guid_track_fov_cos,
                    "guid_filter_quality": projectiles.guid_filter_quality,
                }
            ),
        }
    if emitters is not None:
        payload["emitters"] = {
            "capacity": emitters.capacity,
            **_pack_pool_arrays(
                {
                    "pos": emitters.pos,
                    "power_db": emitters.power_db,
                    "cur_power": emitters.cur_power,
                    "decay_per_s": emitters.decay_per_s,
                    "band": emitters.band,
                    "kind": emitters.kind,
                    "active": emitters.active,
                    "parent_kind": emitters.parent_kind,
                    "parent_idx": emitters.parent_idx,
                    "normal": emitters.normal,
                }
            ),
        }
    if ammo is not None:
        payload["ammo"] = {
            "capacity": int(ammo.capacity.shape[0]),
            "capacity_arr": encode_ndarray(ammo.capacity),
            "reserve": encode_ndarray(ammo.reserve),
            "active": encode_ndarray(ammo.active),
        }
    if fuel is not None:
        payload["fuel"] = {
            "capacity": int(fuel.capacity.shape[0]),
            "capacity_arr": encode_ndarray(fuel.capacity),
            "reserve": encode_ndarray(fuel.reserve),
            "active": encode_ndarray(fuel.active),
        }

    payload["ammo_requests"] = encode_requests(ammo_requests or [])
    payload["fuel_requests"] = encode_requests(fuel_requests or [])

    return msgpack.packb(payload, use_bin_type=True)


@dataclass
class BattlefieldSave:
    """读档结果：恢复的运行时对象集合。"""

    sim_time: float = 0.0
    tick_count: int = 0
    armor: ArmorPool | None = None
    boxes: BoxPool | None = None
    projectiles: ProjectilePool | None = None
    emitters: EmitterPool | None = None  # Phase 6b
    ammo: AmmoState | None = None
    fuel: FuelState | None = None
    ammo_requests: list[SupplyRequest] = field(default_factory=list)
    fuel_requests: list[SupplyRequest] = field(default_factory=list)

    def to_state(self) -> SimulationState:
        """把读档结果组装为 SimulationState（供 SimulationLoop 使用）。"""
        st = SimulationState(sim_time=self.sim_time, tick_count=self.tick_count)
        st.armor_pool = self.armor
        st.box_pool = self.boxes
        st.projectile_pool = self.projectiles
        st.emitter_pool = self.emitters
        st.ammo = self.ammo
        st.fuel = self.fuel
        return st


def deserialize_battlefield(data: bytes) -> BattlefieldSave:
    """从 msgpack bytes 恢复战场（无损）。

    Raises:
        ValueError: 版本不兼容。
    """
    payload = msgpack.unpackb(data, raw=False)
    if payload.get("version") != SAVE_VERSION:
        raise ValueError(
            f"存档版本不兼容: {payload.get('version')} != {SAVE_VERSION}"
        )

    save = BattlefieldSave(
        sim_time=float(payload.get("sim_time", 0.0)),
        tick_count=int(payload.get("tick_count", 0)),
        ammo_requests=decode_requests(payload.get("ammo_requests", [])),
        fuel_requests=decode_requests(payload.get("fuel_requests", [])),
    )

    if "armor" in payload:
        save.armor = _restore_armor(payload["armor"])
    if "boxes" in payload:
        save.boxes = _restore_boxes(payload["boxes"])
    if "projectiles" in payload:
        save.projectiles = _restore_projectiles(payload["projectiles"])
    if "emitters" in payload:
        save.emitters = _restore_emitters(payload["emitters"])
    if "ammo" in payload:
        save.ammo = _restore_ammo(payload["ammo"])
    if "fuel" in payload:
        save.fuel = _restore_fuel(payload["fuel"])

    logger.info(
        "读档完成: t=%.1fs / %d ticks / %s",
        save.sim_time,
        save.tick_count,
        _summarize(save),
    )
    return save


def _summarize(save: BattlefieldSave) -> str:
    parts: list[str] = []
    if save.armor is not None:
        parts.append(f"装甲 {int(np.sum(save.armor.active))}")
    if save.projectiles is not None:
        parts.append(f"弹丸 {int(np.sum(save.projectiles.active))}")
    return ", ".join(parts) if parts else "空"


def save_to_file(blob: bytes, path: str) -> None:
    """写存档到文件（utf-8 无关，二进制写入）。"""
    import os

    with open(path, "wb") as f:
        f.write(blob)
    logger.info("存档已写入 %s (%d bytes)", os.path.basename(path), len(blob))


def load_from_file(path: str) -> BattlefieldSave:
    """从文件读档。"""
    with open(path, "rb") as f:
        data = f.read()
    return deserialize_battlefield(data)
