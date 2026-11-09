"""Phase 6b 步骤 8/9 测试：末敏弹（SFM）扫描 + 激光制导炸弹（SAL）。

覆盖（docs/phase6-guidance-pipeline.md §九）：
    - SFM：scan_armor_detect 双模扫描（毫米波 rcs + 红外 cur_power 双达标抗诱饵）
        BALLISTIC → SCAN 阶段切换 / SCAN 悬停 / EFP 攻顶（顶部 = 正面 × top_armor_factor）
        扫描超时自毁 / mmw emitter 激活（可被 ESM/ARM 反辐射）
    - SAL：吊舱 activate + 漫反射光斑 designate（位置/normal/无父）
        光斑存活（吊舱 active 且 LOS 通）/ 导引头检测光斑（半球 + FOV）
        炸弹 fire(aim_at=光斑坐标) / 命中 → remove_span 炸桥 / 光斑丢失 → MEMORY

运行：uv run python -m pytest tests/test_sfm_sal.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.combat.damage import ArmorType
from Airia.lulilo.combat.engagement import FiringControl
from Airia.lulilo.combat.guidance import GuidanceStage
from Airia.lulilo.combat.sal import SALSystem
from Airia.lulilo.combat.sfm import SfmScanner, scan_armor_detect
from Airia.lulilo.combat.weapons import (
    AmmoConfig,
    AmmoType,
    GuidanceType,
    SeekerType,
    WeaponConfig,
    WeaponRegistry,
    WeaponSystem,
)
from Airia.lulilo.entities.pools import (
    EMITTER_KIND_ILLUMINATOR,
    EMITTER_KIND_IR_SOURCE,
    EMITTER_KIND_SEEKER,
    EMITTER_KIND_SPOT,
    EMITTER_PARENT_NONE,
    EmitterBand,
    ProjectileType,
)
from Airia.lulilo.terrain.heightfield import MAT_BRIDGE, NO_SURFACE, Heightfield
from Airia.lulilo.terrain.map import TerrainMap
from tests.test_guidance_pipeline import (
    _TestRNG,
    make_guided_fc,
    make_guided_registry,
    spawn_at,
)
from tests.test_phase3_integration import hull_health, make_battlefield

# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


def make_sal_registry() -> WeaponRegistry:
    """内置制导注册表 + 激光制导炸弹（gbu_sal）与专用发射武器。"""
    reg = make_guided_registry()
    reg.register_ammo(
        AmmoConfig(
            ammo_id="gbu_sal",
            name="激光制导炸弹（测试）",
            ammo_type=AmmoType.HEAT,
            pen_mm_rha=300.0,
            muzzle_velocity=300.0,
            projectile_type=ProjectileType.MISSILE,
            guidance=GuidanceType.SAL,
            control_law_id="boost_coast_pn",
            nav_gain=3.0,
            seeker=SeekerType.SEMI_ACTIVE_LASER,
            search_fov_deg=15.0,
            track_fov_deg=8.0,
            filter_quality=0.9,
            memory_mode=True,
            is_heat=True,
            drag_coefficient=0.0,
        )
    )
    reg.register(
        WeaponConfig(
            weapon_id="test_gun",
            name="测试发射器",
            caliber_mm=100.0,
            muzzle_velocity=800.0,
            fire_rate_rpm=60.0,
            max_range_m=10000.0,
            compatible_ammo=["gbu_sal"],
        )
    )
    return reg


def make_sal_fc(u: float = 0.0) -> FiringControl:
    """带 SAL 炸弹注册表的 FiringControl。"""
    armor, boxes, projectiles, ammo, fuel, _ = make_battlefield(rng=_TestRNG(u=u))
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        weapon_system=WeaponSystem(make_sal_registry()),
        rng=_TestRNG(u=u),
        events=None,
    )
    return fc


def _spawn_guided_proj(
    fc: FiringControl,
    x: float, y: float, z: float,
    tx: float, ty: float, tz: float,
    ammo_id: str,
    stage: GuidanceStage,
    *,
    vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
) -> int:
    """手动 spawn 制导弹丸（绕开 fire 的散布/冷却，直接设制导字段）。"""
    p = fc.projectiles
    reg = fc.weapon_system.registry
    a_idx = reg.ammo_index(ammo_id)
    proj = p.spawn(
        x, y, z, vx, vy, vz,
        radius=0.3, dmg=0.0, kind=int(ProjectileType.MISSILE),
        owner=-1, weapon=-1, ammo=a_idx,
    )
    ammo = reg.ammo_at(a_idx)
    p.guid_law[proj] = (
        fc.control.index_of(ammo.control_law_id) if ammo.control_law_id else -1
    )
    p.guid_target_pos[proj] = (tx, ty, tz)
    p.guid_stage[proj] = int(stage)
    p.guid_nav_gain[proj] = ammo.nav_gain
    p.guid_search_fov_cos[proj] = float(np.cos(np.radians(ammo.search_fov_deg)))
    p.guid_track_fov_cos[proj] = float(np.cos(np.radians(ammo.track_fov_deg)))
    p.guid_filter_quality[proj] = ammo.filter_quality
    return proj


def _bind_hot_tank(fc: FiringControl, armor_idx: int, rcs: float = 50.0) -> int:
    """给装甲绑定毫米波反射 + 红外辐射（双模达标）。"""
    fc.armor.rcs[armor_idx] = rcs
    e = fc.emitters.activate_ir(armor_idx, 0.8, decay_per_s=0.2)
    fc.emitters.sync_from_parent(fc.armor, fc.projectiles, 0.01)  # 充能到额定
    return e


def _make_bridge_terrain() -> tuple[Heightfield, TerrainMap]:
    """64×64 平地 + y=32 格一段桥（桥面 [6,7) → 顶世界 z=7.0）。"""
    hf = Heightfield(64, 64, cs=2.0, ch=1.0, z_min=-50.0, z_max=200.0)
    ground = np.zeros((64, 64), dtype=np.float32)
    hf.set_ground_layer(ground)
    for gx in range(20, 45):
        hf.add_span(gx, 32, 6.0, 7.0, mat=MAT_BRIDGE)
    return hf, TerrainMap(hf)


# ---------------------------------------------------------------------------
# SFM：双模扫描内核
# ---------------------------------------------------------------------------


def test_scan_armor_detect_both_modes() -> None:
    """双模达标才命中：RCS 高 + 红外热 → 命中；诱饵（低 RCS）/ 冷目标 → 不命中。"""
    armor_x = np.array([10.0, 10.0, 40.0], dtype=np.float32)
    armor_y = np.array([10.0, 10.0, 40.0], dtype=np.float32)
    armor_rcs = np.array([50.0, 1.0, 50.0], dtype=np.float32)  # 0 真目标 / 1 低 RCS 诱饵 / 2 冷目标
    armor_active = np.ones(3, dtype=np.bool_)
    # 红外：父 0 热（0.8）/ 父 2 冷（0.02 < 阈值 0.1）
    ir_cur = np.array([0.8, 0.02], dtype=np.float32)
    ir_parent = np.array([0, 2], dtype=np.int32)
    ir_active = np.ones(2, dtype=np.bool_)

    out = np.full(3, -1, dtype=np.int32)
    scan_armor_detect(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([100.0, 100.0, 100.0], dtype=np.float32),
        80.0,
        armor_x, armor_y, armor_rcs, armor_active,
        ir_cur, ir_parent, ir_active,
        5.0, 0.1,
        out,
    )
    assert out[0] == 0    # 装甲 0 双达标
    assert out[1] == 0    # 同扫描半径内仍命中装甲 0
    assert out[2] == 0
    # 单独验证 2 号弹丸扫描半径只覆盖装甲 2 但双模不达标 → -1
    out2 = np.full(1, -1, dtype=np.int32)
    scan_armor_detect(
        np.array([40.0], dtype=np.float32),
        np.array([40.0], dtype=np.float32),
        np.array([100.0], dtype=np.float32),
        5.0,  # 半径 5：只覆盖装甲 2（40,40）
        armor_x, armor_y, armor_rcs, armor_active,
        ir_cur, ir_parent, ir_active,
        5.0, 0.1,
        out2,
    )
    assert out2[0] == -1  # 冷目标（红外不达标）不命中


def test_sfm_scanner_hits_hot_armor() -> None:
    """SfmScanner：SCAN 弹丸双模扫描命中热装甲。"""
    fc = make_guided_fc(u=0.0)
    t = spawn_at(fc, 100.0, 100.0, armor_mm=600.0, armor_type=int(ArmorType.RHA))
    _bind_hot_tank(fc, t)

    proj = _spawn_guided_proj(
        fc, 100.0, 100.0, 90.0, 100.0, 100.0, 90.0, "sfm_155", GuidanceStage.SCAN
    )
    hits = fc.sfm.scan(fc.projectiles, fc._ammo_of)
    assert hits.get(proj) == t


def test_sfm_scanner_ignores_cold_armor() -> None:
    """SfmScanner：无红外 emitter（冷目标）不被末敏弹命中（抗诱饵）。"""
    fc = make_guided_fc(u=0.0)
    t = spawn_at(fc, 100.0, 100.0, armor_mm=600.0)
    fc.armor.rcs[t] = 50.0  # 毫米波反射达标，但无红外 → 不达标

    proj = _spawn_guided_proj(
        fc, 100.0, 100.0, 90.0, 100.0, 100.0, 90.0, "sfm_155", GuidanceStage.SCAN
    )
    hits = fc.sfm.scan(fc.projectiles, fc._ammo_of)
    assert proj not in hits


def test_sfm_ballistic_to_scan_transition() -> None:
    """BALLISTIC → SCAN：高度 ≤ scan_altitude 时切换 + 激活 mmw emitter + age 清零。"""
    fc = make_guided_fc(u=0.0)
    proj = _spawn_guided_proj(
        fc, 10.0, 10.0, 50.0, 10.0, 10.0, 0.0, "sfm_155", GuidanceStage.BALLISTIC
    )
    fc.projectiles.age[proj] = 3.0
    fc._update_guidance_stages()
    assert fc.projectiles.guid_stage[proj] == int(GuidanceStage.SCAN)
    assert fc.projectiles.age[proj] == 0.0  # 扫描窗口计时复用 age
    # mmw emitter 激活（父 = 弹丸，可被 ESM/ARM 反辐射，§7.4）
    assert fc.emitters.active_count == 1
    e = int(fc.emitters.active_indices()[0])
    assert fc.emitters.kind[e] == EMITTER_KIND_SEEKER
    assert fc.emitters.parent_kind[e] == 1  # 父 = 弹丸


def test_sfm_scan_hovers_and_detonates() -> None:
    """SCAN 弹丸悬停 + EFP 攻顶毁伤（顶部 = 正面 × top_armor_factor）。"""
    fc = make_guided_fc(u=0.0)
    t = spawn_at(fc, 100.0, 100.0, armor_mm=600.0, armor_type=int(ArmorType.RHA))
    _bind_hot_tank(fc, t)

    proj = _spawn_guided_proj(
        fc, 100.0, 100.0, 90.0, 100.0, 100.0, 90.0, "sfm_155", GuidanceStage.SCAN
    )
    z0 = float(fc.projectiles.pos[proj, 2])
    fc._sfm_scan()
    # 命中 → 弹丸销毁
    assert not fc.projectiles.active[proj]
    # 装甲受毁伤（顶甲 = 600×0.3 = 180mm < efp 200mm → 击穿，命中部位受损或摧毁）
    damaged = any(
        float(fc.boxes.hp[bi]) < float(fc.boxes.hp_max[bi])
        for bi in range(fc.boxes.capacity)
        if fc.boxes.active[bi] and fc.boxes.parent[bi] == t
    )
    assert damaged or not fc.armor.active[t]
    kinds = [e.kind for e in fc.events.drain()]
    assert "sfm_detonate" in kinds


def test_sfm_efp_needs_penetration() -> None:
    """EFP 穿深不足（顶甲 > efp_pen_mm）→ 未击穿，装甲基本无损。"""
    fc = make_guided_fc(u=0.0)
    # 顶甲 = 900×0.3 = 270mm > efp 200mm
    t = spawn_at(fc, 100.0, 100.0, armor_mm=900.0, armor_type=int(ArmorType.RHA))
    _bind_hot_tank(fc, t)

    proj = _spawn_guided_proj(
        fc, 100.0, 100.0, 90.0, 100.0, 100.0, 90.0, "sfm_155", GuidanceStage.SCAN
    )
    fc._sfm_scan()
    assert not fc.projectiles.active[proj]
    # 未击穿：health 只有微小结构损伤（未击穿封顶 min(15, ...)）
    assert hull_health(fc.armor, fc.boxes, t) >= 85.0


def test_sfm_scan_timeout_self_destruct() -> None:
    """SCAN 扫描超时（age > scan_duration_s）→ _recycle_stale 自毁。"""
    fc = make_guided_fc(u=0.0)
    proj = _spawn_guided_proj(
        fc, 10.0, 10.0, 50.0, 10.0, 10.0, 0.0, "sfm_155", GuidanceStage.SCAN
    )
    fc.projectiles.age[proj] = 4.9  # scan_duration_s=5.0
    fc.tick(0.2)
    assert not fc.projectiles.active[proj]
    kinds = [e.kind for e in fc.events.drain()]
    assert "miss" in kinds  # 复用超龄回收路径（脱靶）


# ---------------------------------------------------------------------------
# SAL：吊舱 + 光斑 + 导引头
# ---------------------------------------------------------------------------


def test_sal_activate_illuminator_and_designate() -> None:
    """吊舱激活 + LOS 通过后 designate 漫反射光斑（kind=4，无父，normal 记录）。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier, power=1000.0)
    assert fc.emitters.kind[illum] == EMITTER_KIND_ILLUMINATOR
    assert fc.emitters.band[illum] == int(EmitterBand.LASER)
    assert fc.emitters.parent_kind[illum] == 0  # 父 = 装甲

    spot = fc.sal.designate(illum, (100.0, 100.0, 5.0), 100.0, normal=(0.0, 0.0, 1.0))
    assert spot >= 0
    assert fc.emitters.kind[spot] == EMITTER_KIND_SPOT
    assert fc.emitters.parent_kind[spot] == EMITTER_PARENT_NONE  # 无父，不走 sync
    assert fc.emitters.parent_idx[spot] == illum  # 记录来源照射器
    assert np.allclose(fc.emitters.pos[spot], (100.0, 100.0, 5.0))
    assert fc.emitters.normal[spot, 2] == 1.0


def test_sal_designate_fails_without_illuminator() -> None:
    """吊舱失效（未激活/已关机）→ designate 失败返回 -1。"""
    fc = make_sal_fc(u=0.0)
    assert fc.sal.designate(999, (0.0, 0.0, 5.0), 100.0) == -1
    # 非吊舱 emitter（红外源）不能 designate
    fc2 = make_guided_fc(u=0.0)
    t = spawn_at(fc2, 0.0, 0.0)
    ir = fc2.emitters.activate_ir(t, 0.8)
    assert fc2.sal.designate(ir, (0.0, 0.0, 5.0), 100.0) == -1


def test_sal_spot_lifecycle() -> None:
    """光斑存活 = 吊舱 active 且 LOS 通；吊舱关机 → 光斑失效。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (100.0, 100.0, 5.0), 100.0)
    assert spot >= 0

    # 吊舱 active + LOS → 保留
    fc.sal.update(0.01)
    assert fc.emitters.active[spot]

    # 吊舱战术关机 → 光斑失效
    fc.emitters.deactivate(illum)
    fc.sal.update(0.01)
    assert not fc.emitters.active[spot]

    # 吊舱重新开机 → 重新 designate 新光斑（旧光斑失效后需重新照射）
    fc.emitters.reactivate(illum)
    spot2 = fc.sal.designate(illum, (100.0, 100.0, 5.0), 100.0)
    assert spot2 >= 0
    assert fc.emitters.active[spot2]


def test_sal_seeker_detects_spot() -> None:
    """SAL 导引头检测光斑（半球 + LOS + FOV）：炸弹在光斑上方 → 检测成功。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (0.0, 0.0, 50.0), 100.0, normal=(0.0, 0.0, 1.0))

    proj = _spawn_guided_proj(
        fc, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, "gbu_sal", GuidanceStage.TRACK,
        vx=0.0, vy=0.0, vz=-30.0,
    )
    detected = fc._seeker_detect()
    assert detected.get(proj, False) is True
    # 检测成功 → guid_target_pos 更新为光斑坐标（滤波噪声 σ∝(1-filter_quality)≈0.2）
    tpos = fc.projectiles.guid_target_pos[proj]
    assert np.allclose(tpos, (0.0, 0.0, 50.0), atol=1.0)


def test_sal_hemisphere_occlusion() -> None:
    """光斑半球漫反射：导引头在背面（normal 反方向）→ 检测不到 → MEMORY。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (0.0, 0.0, 50.0), 100.0, normal=(0.0, 0.0, 1.0))

    # 炸弹在光斑正下方（背面）：s - e = (0,0,-40)，normal·(s-e) < 0
    proj = _spawn_guided_proj(
        fc, 0.0, 0.0, 10.0, 0.0, 0.0, 50.0, "gbu_sal", GuidanceStage.TRACK
    )
    fc._update_guidance_stages()
    assert fc.projectiles.guid_stage[proj] == int(GuidanceStage.MEMORY)
    assert np.allclose(fc.projectiles.guid_memory_pos[proj], (0.0, 0.0, 50.0))


def test_sal_loses_spot_to_memory() -> None:
    """光斑消失（吊舱关机）→ 检测失败 → MEMORY（最后已知位置）。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (0.0, 0.0, 50.0), 100.0, normal=(0.0, 0.0, 1.0))

    proj = _spawn_guided_proj(
        fc, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, "gbu_sal", GuidanceStage.TRACK,
        vx=0.0, vy=0.0, vz=-30.0,
    )
    fc.emitters.deactivate(spot)
    fc._update_guidance_stages()
    assert fc.projectiles.guid_stage[proj] == int(GuidanceStage.MEMORY)
    assert np.allclose(fc.projectiles.guid_memory_pos[proj], (0.0, 0.0, 50.0))


# ---------------------------------------------------------------------------
# SAL：fire(aim_at=光斑坐标) + 炸桥
# ---------------------------------------------------------------------------


def test_sal_fire_aim_at_spot() -> None:
    """fire(aim_at=光斑坐标)：初始瞄准 = 光斑，stage=TRACK（docs §7.5）。"""
    fc = make_sal_fc(u=0.0)
    shooter = spawn_at(fc, 0.0, 0.0, weapon="test_gun")
    rec = fc.fire(shooter, -1, ammo_id="gbu_sal", aim_at=(100.0, 100.0, 50.0))
    assert rec is not None
    assert np.allclose(fc.projectiles.guid_target_pos[rec.proj_idx], (100.0, 100.0, 50.0))
    assert fc.projectiles.guid_stage[rec.proj_idx] == int(GuidanceStage.TRACK)
    assert fc.projectiles.guid_law[rec.proj_idx] == fc.control.index_of("boost_coast_pn")


def test_sal_strikes_bridge() -> None:
    """核心：SAL 炸弹到达光斑 → remove_span 破坏桥面（AGENTS.md §3.3 自由破坏）。"""
    hf, terrain = _make_bridge_terrain()
    fc = make_sal_fc(u=0.0)
    fc.terrain = terrain
    fc.sal = SALSystem(fc.emitters, terrain)

    carrier = spawn_at(fc, 30.0, 64.0)
    illum = fc.sal.activate_illuminator(carrier)
    # 桥面在 y=32 格（世界 y=64..66），光斑放桥面表面上方（z=8 > 桥面顶 7.0，
    # LOS 不被表面遮挡；激光光斑物理上漫反射于表面之上，docs §7.5）
    spot = fc.sal.designate(illum, (44.0, 65.0, 8.0), 100.0, normal=(0.0, 0.0, 1.0))
    assert spot >= 0

    # 炸桥前：桥面存在（surface_z layer=1 = 7.0）
    assert hf.surface_z(22, 32, 1) == pytest.approx(7.0)

    # 炸弹已到达光斑附近（pos = 目标坐标）→ _check_sal_impact 引爆
    proj = _spawn_guided_proj(
        fc, 44.0, 65.0, 8.0, 44.0, 65.0, 8.0, "gbu_sal", GuidanceStage.TRACK
    )
    fc._check_sal_impact()

    # 弹丸销毁
    assert not fc.projectiles.active[proj]
    # 桥面被炸掉（remove_span 截断）
    assert hf.surface_z(22, 32, 1) == NO_SURFACE
    # 地面仍保留（只挖桥面区间）
    assert hf.ground_z(22, 32) == pytest.approx(0.0)
    # 事件
    kinds = [e.kind for e in fc.events.drain()]
    assert "sal_detonate" in kinds
    # 导航缓存失效（下次访问重建）
    _ = terrain.nav_graph  # 触发生成
    terrain.invalidate_geometry()
    assert terrain._nav_graph is None


def test_sal_detonates_only_near_target() -> None:
    """炸弹未到达目标附近（距离 > 引爆半径）→ 不引爆。"""
    fc = make_sal_fc(u=0.0)
    carrier = spawn_at(fc, 0.0, 0.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (0.0, 0.0, 50.0), 100.0, normal=(0.0, 0.0, 1.0))

    proj = _spawn_guided_proj(
        fc, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, "gbu_sal", GuidanceStage.TRACK,
        vx=0.0, vy=0.0, vz=-30.0,
    )
    fc._check_sal_impact()
    assert fc.projectiles.active[proj]  # 远离目标，未引爆
    assert "sal_detonate" not in [e.kind for e in fc.events.drain()]


def test_sal_bomb_flies_to_spot() -> None:
    """端到端：SAL 炸弹从高空投放，制导飞向光斑（到达附近后引爆）。"""
    hf, terrain = _make_bridge_terrain()
    fc = make_sal_fc(u=0.0)
    fc.terrain = terrain
    fc.sal = SALSystem(fc.emitters, terrain)

    carrier = spawn_at(fc, 30.0, 64.0)
    illum = fc.sal.activate_illuminator(carrier)
    spot = fc.sal.designate(illum, (44.0, 65.0, 8.0), 100.0, normal=(0.0, 0.0, 1.0))
    assert spot >= 0

    # 炸弹在桥面上方 200m 投放，目标 = 光斑坐标
    proj = _spawn_guided_proj(
        fc, 44.0, 65.0, 200.0, 44.0, 65.0, 8.0, "gbu_sal", GuidanceStage.TRACK,
        vx=0.0, vy=0.0, vz=-50.0,
    )
    # 推进到接近目标（boost_coast_pn 5s 开机 + 滑翔）
    fc.tick(12.0)
    if fc.projectiles.active[proj]:
        pp = fc.projectiles.pos[proj]
        dist = float(np.sqrt(
            (pp[0] - 44.0) ** 2 + (pp[1] - 65.0) ** 2 + (pp[2] - 8.0) ** 2
        ))
        assert dist < 40.0  # 明显接近光斑
    else:
        # 已引爆（到达目标附近）→ 桥被破坏
        assert hf.surface_z(22, 32, 1) == NO_SURFACE
