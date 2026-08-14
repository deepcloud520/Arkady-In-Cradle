"""Phase 6c 步骤③ 测试：爆炸/毁伤提交池 + 弹药激活。

覆盖（docs/phase6-ai-pipeline.md §七/§十一）：
    - DamageSubmissionPool：submit/clear/active/扩容（非零默认 target_idx=-1 保留）/爆炸数据 drain
    - resolve_area_damage 内核：球形线性衰减 / 破片锥锥内判定 / 忽略失活装甲
    - FiringControl 集成：_sal_blast_damage 提交 SPHERE → tick 末结算
        _submit_hit_blast（HE_FRAG）提交 SPHERE + FRAG_CONE → tick 末区域毁伤
        _resolve_hit 命中 HE_FRAG → 提交爆炸
        CONE 事件经提交池 → _resolve_top_attack 攻顶毁伤
    - 弹药激活：AmmoType HE_FRAG/PELE/APCR/SAPHEI + AmmoConfig 爆炸字段（默认 0）+ _kind_of 映射
    - SimulationState.snapshot 含 damage 池

运行：uv run python -m pytest tests/test_damage_submission.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.combat.collision import CollisionHit
from Airis.lulilo.combat.damage import ArmorType
from Airis.lulilo.combat.damage_submission import (
    DamageKind,
    DamageSubmissionPool,
    resolve_area_damage,
)
from Airis.lulilo.combat.engagement import FiringControl
from Airis.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    WeaponConfig,
    WeaponRegistry,
    WeaponSystem,
)
from Airis.lulilo.entities.pools import ProjectileType
from Airis.lulilo.simulation.loop import EventBus
from Airis.lulilo.simulation.state import SimulationState
from tests.test_guidance_pipeline import _TestRNG, spawn_at
from tests.test_phase3_integration import make_battlefield

# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


def make_hefrag_registry() -> WeaponRegistry:
    """内置注册表 + 155mm 高爆破片弹（HE_FRAG，含爆炸/破片参数）。"""
    reg = WeaponRegistry.with_builtin()
    reg.register_ammo(
        AmmoConfig(
            ammo_id="he_frag_155",
            name="155mm 高爆破片（测试）",
            ammo_type=AmmoType.HE_FRAG,
            pen_mm_rha=20.0,
            muzzle_velocity=800.0,
            projectile_type=ProjectileType.MISSILE,
            blast_radius=15.0,
            blast_dmg=80.0,
            frag_dmg=10.0,
            frag_count=20,
        )
    )
    reg.register(
        WeaponConfig(
            weapon_id="test_gun",
            name="测试炮",
            caliber_mm=155.0,
            muzzle_velocity=800.0,
            fire_rate_rpm=10.0,
            max_range_m=10000.0,
            compatible_ammo=["he_frag_155"],
        )
    )
    return reg


def make_hefrag_fc(u: float = 0.0) -> FiringControl:
    """带 HE_FRAG 弹药注册表的 FiringControl。"""
    armor, boxes, projectiles, ammo, fuel, _ = make_battlefield(rng=_TestRNG(u=u))
    return FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        weapon_system=WeaponSystem(make_hefrag_registry()),
        rng=_TestRNG(u=u),
        events=EventBus(),
    )


def _spawn_proj(fc: FiringControl, x: float, y: float, z: float,
                vx: float, vy: float, vz: float, ammo_id: str) -> int:
    """手动 spawn 弹丸（带 ammo 索引，供 _resolve_hit / _submit_hit_blast 用）。"""
    p = fc.projectiles
    a_idx = fc.weapon_system.registry.ammo_index(ammo_id)
    return p.spawn(
        x, y, z, vx, vy, vz,
        radius=0.1, dmg=0.0, kind=int(ProjectileType.MISSILE),
        owner=-1, weapon=-1, ammo=a_idx,
    )


# ---------------------------------------------------------------------------
# DamageSubmissionPool：基础
# ---------------------------------------------------------------------------


def test_pool_submit_clear_reuse() -> None:
    pool = DamageSubmissionPool(capacity=8)
    assert pool.active_count == 0
    ev = pool.submit_damage(DamageKind.SPHERE, (1.0, 2.0, 3.0), radius=5.0, pen_mm=30.0)
    assert pool.active_count == 1
    assert pool.kind[ev] == int(DamageKind.SPHERE)
    assert pool.center[ev][0] == 1.0
    assert pool.radius[ev] == 5.0
    assert pool.pen_mm[ev] == 30.0
    assert pool.target_idx[ev] == -1
    assert pool.direction[ev][2] == 1.0  # 默认 +Z

    pool.clear()
    assert pool.active_count == 0
    # free_stack LIFO 复用索引
    ev2 = pool.submit_damage(DamageKind.CONE, (0.0, 0.0, 0.0), target_idx=3)
    assert ev2 == ev
    assert pool.target_idx[ev2] == 3
    assert pool.kind[ev2] == int(DamageKind.CONE)


def test_pool_expand_keeps_defaults() -> None:
    pool = DamageSubmissionPool(capacity=2)
    a = pool.submit_damage(DamageKind.SPHERE, (0.0, 0.0, 0.0))
    b = pool.submit_damage(DamageKind.CONE, (0.0, 0.0, 0.0), target_idx=5)
    c = pool.submit_damage(DamageKind.SPHERE, (0.0, 0.0, 0.0))  # 触发扩容
    assert pool.capacity == 4
    assert pool.active_count == 3
    assert pool.target_idx[b] == 5
    assert pool.target_idx[c] == -1  # 扩容新区默认 -1（非零默认补填）
    assert pool.target_idx[a] == -1


def test_pool_drain_explosions() -> None:
    pool = DamageSubmissionPool(capacity=4)
    pool.submit_damage(DamageKind.SPHERE, (1.0, 2.0, 3.0), radius=6.0)
    pool.submit_damage(DamageKind.FRAG_CONE, (4.0, 5.0, 6.0), radius=3.0)
    expl = pool.drain_explosions()
    assert len(expl) == 2
    assert expl[0]["kind"] == int(DamageKind.SPHERE)
    assert expl[0]["x"] == 1.0 and expl[0]["z"] == 3.0
    assert expl[1]["radius"] == 3.0
    assert pool.drain_explosions() == []  # 已清空


# ---------------------------------------------------------------------------
# resolve_area_damage 内核
# ---------------------------------------------------------------------------


def test_sphere_kernel_linear_falloff() -> None:
    pool = DamageSubmissionPool(capacity=4)
    pool.submit_damage(DamageKind.SPHERE, (0.0, 0.0, 0.0), radius=10.0, pen_mm=100.0)
    armor_pos = np.array(
        [[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 0, 0]], dtype=np.float32
    )
    health = np.full(4, 100.0, dtype=np.float32)
    active = np.ones(4, dtype=np.bool_)
    resolve_area_damage(
        pool.kind, pool.center, pool.radius, pool.pen_mm,
        pool.direction, pool.cone_angle, pool.frag_count, pool.active,
        armor_pos, health, active,
    )
    assert health[0] == pytest.approx(0.0)    # 中心满伤害 100 → 0
    assert health[1] == pytest.approx(50.0)   # 5/10 → 100*(1-0.5)=50
    assert health[2] == pytest.approx(100.0)  # 边界 10 → 衰减到 0
    assert health[3] == pytest.approx(100.0)  # 15 > 10 无伤


def test_sphere_kernel_ignores_inactive() -> None:
    pool = DamageSubmissionPool(capacity=4)
    pool.submit_damage(DamageKind.SPHERE, (0.0, 0.0, 0.0), radius=10.0, pen_mm=100.0)
    armor_pos = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
    health = np.full(2, 100.0, dtype=np.float32)
    active = np.array([True, False], dtype=np.bool_)
    resolve_area_damage(
        pool.kind, pool.center, pool.radius, pool.pen_mm,
        pool.direction, pool.cone_angle, pool.frag_count, pool.active,
        armor_pos, health, active,
    )
    assert health[0] == pytest.approx(0.0)    # 失活装甲不动
    assert health[1] == pytest.approx(100.0)  # 失活装甲不动


def test_frag_cone_kernel() -> None:
    pool = DamageSubmissionPool(capacity=4)
    pool.submit_damage(
        DamageKind.FRAG_CONE,
        (0.0, 0.0, 0.0),
        radius=10.0,
        pen_mm=20.0,
        direction=(0.0, 0.0, 1.0),
        cone_angle=float(np.radians(30.0)),
        frag_count=10,
    )
    # 装甲：正上方 5m（锥内）/ 侧向 5m（90° 锥外）/ 斜上 45°（锥外）/ 正上 15m（超射程）
    armor_pos = np.array(
        [[0, 0, 5], [5, 0, 0], [5, 0, 5], [0, 0, 15]], dtype=np.float32
    )
    health = np.full(4, 100.0, dtype=np.float32)
    active = np.ones(4, dtype=np.bool_)
    resolve_area_damage(
        pool.kind, pool.center, pool.radius, pool.pen_mm,
        pool.direction, pool.cone_angle, pool.frag_count, pool.active,
        armor_pos, health, active,
    )
    assert health[0] == pytest.approx(90.0)   # 锥内 5/10 → 20*(1-0.5)=10
    assert health[1] == pytest.approx(100.0)  # 侧向 90° > 30° 锥外
    assert health[2] == pytest.approx(100.0)  # 斜上 45° > 30° 锥外
    assert health[3] == pytest.approx(100.0)  # 超射程


# ---------------------------------------------------------------------------
# FiringControl 集成
# ---------------------------------------------------------------------------


def test_sal_blast_submits_to_pool_and_resolves() -> None:
    """SAL 爆破 → 提交 SPHERE → tick 末结算（§七.3）。"""
    fc = make_hefrag_fc(u=0.0)
    t = spawn_at(fc, 0.0, 0.0)
    fc._sal_blast_damage(0.0, 0.0, 0.0)  # 以目标为中心爆破
    assert fc.damage_pool.active_count == 1
    ev = int(fc.damage_pool.active_indices()[0])
    assert fc.damage_pool.kind[ev] == int(DamageKind.SPHERE)
    # 未结算：血量不变
    assert fc.armor.health[t] == 100.0
    fc.tick(0.1)  # tick 末结算
    # 中心满伤 SAL_BLAST_DMG=60 → 100-60=40
    assert fc.armor.health[t] == pytest.approx(40.0)


def test_submit_hit_blast_sphere_and_frag() -> None:
    """HE_FRAG 命中 → 提交 SPHERE + FRAG_CONE → tick 末区域毁伤。"""
    fc = make_hefrag_fc(u=0.0)
    victim = spawn_at(fc, 0.0, 0.0)
    bystander = spawn_at(fc, 12.0, 0.0)
    ammo = fc.weapon_system.registry.get_ammo("he_frag_155")
    proj = _spawn_proj(fc, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, "he_frag_155")
    hit = CollisionHit(proj, victim, 0.0, 0.0, 0.0, 0.0)
    fc._submit_hit_blast(ammo, hit, proj)

    assert fc.damage_pool.active_count == 2  # SPHERE + FRAG_CONE
    kinds = sorted(int(fc.damage_pool.kind[i]) for i in fc.damage_pool.active_indices())
    assert kinds == [int(DamageKind.SPHERE), int(DamageKind.FRAG_CONE)]
    # 未结算：血量不变
    assert fc.armor.health[victim] == 100.0

    fc.tick(0.1)  # tick 末结算 → 区域毁伤
    assert fc.armor.health[victim] < 100.0     # 中心满伤
    assert fc.armor.health[bystander] < 100.0  # 半径内受范围毁伤


def test_resolve_hit_submits_blast_for_he_frag() -> None:
    """_resolve_hit 命中 HE_FRAG → 自动提交爆炸 → tick 末范围毁伤。"""
    fc = make_hefrag_fc(u=0.0)
    target = spawn_at(fc, 0.0, 0.0)
    bystander = spawn_at(fc, 12.0, 0.0)
    proj = _spawn_proj(fc, 100.0, 0.0, 0.0, -100.0, 0.0, 0.0, "he_frag_155")
    hit = CollisionHit(proj, target, 0.0, 0.0, 0.0, 0.0)

    fc._resolve_hit(hit, 0.01)
    assert fc.damage_pool.active_count >= 1
    kinds = [int(fc.damage_pool.kind[i]) for i in fc.damage_pool.active_indices()]
    assert int(DamageKind.SPHERE) in kinds

    fc.tick(0.1)  # tick 末结算 → bystander 也受范围毁伤
    assert fc.armor.health[bystander] < 100.0


def test_cone_event_top_attack_via_pool() -> None:
    """CONE 事件经提交池 → _resolve_top_attack 攻顶毁伤（docs §五.3）。"""
    fc = make_hefrag_fc(u=0.0)
    t = spawn_at(fc, 100.0, 100.0, armor_mm=600.0, armor_type=int(ArmorType.RHA))
    fc.damage_pool.submit_damage(
        DamageKind.CONE,
        (100.0, 100.0, 0.0),
        pen_mm=400.0,
        target_idx=t,
        top_factor=0.5,
    )
    fc.tick(0.1)
    # 顶甲 = 600×0.5 = 300mm（RHA）< 穿深 400 → 击穿，血量显著下降
    assert fc.armor.health[t] < 100.0


def test_cone_event_skips_missing_target() -> None:
    """CONE 事件目标不存在/已死 → 跳过，不崩溃。"""
    fc = make_hefrag_fc(u=0.0)
    fc.damage_pool.submit_damage(
        DamageKind.CONE, (0.0, 0.0, 0.0), pen_mm=400.0, target_idx=99, top_factor=0.5
    )
    fc.tick(0.1)  # 目标越界 → 跳过，无异常
    assert fc.damage_pool.active_count == 0


# ---------------------------------------------------------------------------
# 弹药激活
# ---------------------------------------------------------------------------


def test_kind_of_new_ammo() -> None:
    """HE_FRAG/SAPHEI → MISSILE；APCR/PELE → BULLET（§十一.4）。"""
    fc = make_hefrag_fc(u=0.0)
    reg = fc.weapon_system.registry
    reg.register_ammo(AmmoConfig(ammo_id="apcr_t", ammo_type=AmmoType.APCR))
    reg.register_ammo(AmmoConfig(ammo_id="pele_t", ammo_type=AmmoType.PELE))
    reg.register_ammo(AmmoConfig(ammo_id="saphei_t", ammo_type=AmmoType.SAPHEI))
    assert fc._kind_of("he_frag_155") == int(ProjectileType.MISSILE)
    assert fc._kind_of("saphei_t") == int(ProjectileType.MISSILE)
    assert fc._kind_of("apcr_t") == int(ProjectileType.BULLET)
    assert fc._kind_of("pele_t") == int(ProjectileType.BULLET)


def test_ammo_config_new_fields() -> None:
    """AmmoConfig 新爆炸字段带默认值；旧构造不变（extra=forbid 已加齐）。"""
    a = AmmoConfig(ammo_id="x", ammo_type=AmmoType.HE_FRAG)
    assert a.blast_radius == 0.0
    assert a.blast_dmg == 0.0
    assert a.frag_dmg == 0.0
    assert a.frag_count == 0
    assert a.fuse_delay == 0.0
    b = AmmoConfig(
        ammo_id="y",
        blast_radius=10.0,
        blast_dmg=50.0,
        frag_dmg=5.0,
        frag_count=12,
        fuse_delay=0.5,
    )
    assert b.blast_radius == 10.0
    assert b.frag_count == 12
    assert b.fuse_delay == 0.5
    # extra=forbid：未知字段拒绝
    with pytest.raises(Exception):
        AmmoConfig(ammo_id="z", bogus_field=1)


# ---------------------------------------------------------------------------
# 状态挂载
# ---------------------------------------------------------------------------


def test_state_snapshot_includes_damage_pool() -> None:
    st = SimulationState()
    st.damage_pool = DamageSubmissionPool()
    snap = st.snapshot()
    assert snap.pools["damage"] is st.damage_pool
