"""开火链交战管理器（AGENTS.md Phase 3 正式迭代：combat/engagement.py）。

把 Phase 3 骨架各孤立模块串成完整开火链
（docs/phase3-combat-design.md §七 开火链）：

    1. 选择目标 + 瞄准点（AI/FireControl → FiringControl.fire）
    2. 计算 sigma（WeaponSystem 散布模型，统计层）
    3. 生成实际落点（RNGManager 高斯散布）
    4. 抛射物按落点方向发射（ProjectilePool.spawn，物理弹道）
    5. 空间哈希碰撞检测 → 命中部位（CollisionDetector + BoxPool 部位判定）
    6. 斜距/跳弹/穿透判定（damage.compute_armor_eff / resolve_penetration_single）
    7. 后效毁伤 → 更新 health / box.active / 摧毁（DamageResolver + 部位毁伤）
    8. 命中事件 → EventBus（叙事层消费，战斗可跳过）

设计要点：
    - 统计层只算 sigma（WeaponSystem），物理层决定落点/命中/毁伤——可解释、可追溯
    - 命中部位由 BoxPool 的 box 判定（PartType：HULL/TURRET/ENGINE/TRACK/AMMO_RACK）
    - 部位毁伤映射（design doc §5.3）：
        TURRET 毁 → box 失活 → 无法开火
        ENGINE / TRACK 毁 → box 失活 → 无法机动/减速
        AMMO_RACK 殉爆（cookoff）→ 秒杀
    - 弹药消耗由 AmmoManager 扣减（fire 每次 1 发），destroy 时联动 release
    - 子弹 Tick（TICK_BULLET=0.01s）由本类内部细分推进，外部只需 tick(dt)
    - 弹丸元数据（武器/弹药）暂存 Python dict（骨架期正确性优先；
      正式迭代可下沉为 SoA 数组 proj_weapon/proj_ammo）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Airis.lulilo.combat.collision import (
    CollisionDetector,
    CollisionHit,
    FormationResolver,
    segment_aabb_single,
    world_to_local,
)
from Airis.lulilo.combat.damage import (
    ArmorType,
    DamageResolver,
    compute_armor_eff,
    resolve_behind_armor_single,
    resolve_penetration_single,
)
from Airis.lulilo.combat.damage_submission import (
    DamageKind,
    DamageSubmissionPool,
    resolve_area_damage,
)
from Airis.lulilo.combat.guidance import (
    GuidanceStage,
    ManualOverride,
    MissileControlRegistry,
)
from Airis.lulilo.combat.sal import (
    SALSystem,
    SAL_BLAST_DMG,
    SAL_CRATER_ABOVE,
    SAL_CRATER_DEPTH,
    SAL_CRATER_RADIUS,
    SAL_DETONATE_RADIUS,
)
from Airis.lulilo.combat.sfm import SfmScanner
from Airis.lulilo.combat.weapons import (
    AmmoType,
    GuidanceType,
    SeekerType,
    WeaponRegistry,
    WeaponSystem,
)
from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.detection.los import line_of_sight
from Airis.lulilo.detection.sensors import (
    esm_detection_probability,
    esm_detection_range,
    ir_detection_probability,
    ir_detection_range,
    radar_detection_probability,
    radar_detection_range,
)
from Airis.lulilo.entities.pools import (
    EMITTER_KIND_IR_SOURCE,
    EMITTER_KIND_SEEKER,
    EMITTER_KIND_SPOT,
    EMITTER_PARENT_ARMOR,
    EMITTER_PARENT_PROJECTILE,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.movement.integrate import (
    integrate_guided_projectile_movement,
    integrate_projectile_movement,
)
from Airis.lulilo.simulation.loop import EventBus, TICK_BULLET

logger = get_logger(__name__)

# 弹丸最大飞行寿命（秒），超出回收（防止永不命中无限存在）
PROJECTILE_MAX_AGE = 20.0
# 弹丸回收边界（米），超出即视为脱靶/飞出战场
WORLD_BOUND = 5000.0
# 发射初始抬高（炮管高度，m）
MUZZLE_HEIGHT = 1.5

# Phase 6b 制导常量（docs/phase6-guidance-pipeline.md §八）
GUIDED_MAX_ACCEL = 40.0 * 9.81  # 总加速度饱和（推力+转向；40g 防截断轴向推力）
SEEK_RADIUS_M = 200.0           # 导引头在锁定坐标附近的目标搜索半径（防跳变）

# 部位 → 等效装甲厚度系数（简化：不同部位抗性不同）
PART_ARMOR_FACTOR: dict[int, float] = {
    int(PartType.HULL): 1.0,
    int(PartType.TURRET): 0.9,
    int(PartType.ENGINE): 0.6,
    int(PartType.TRACK): 0.5,
    int(PartType.AMMO_RACK): 0.5,
}


@dataclass
class UnitCombatConfig:
    """单辆装甲的交战属性（FiringControl 按 armor_idx 登记）。

    正式版可下沉为 SoA 数组（unit_weapon / unit_armor_mm / unit_armor_type …），
    骨架期用 dict 配置保证正确性与可测性。
    """

    weapon_id: str = "m256_120mm"
    ammo_id: str | None = None
    armor_mm: float = 300.0                 # 正面标称装甲厚度 (mm RHA)
    armor_type: int = ArmorType.RHA
    crew_skill: float = 0.7                 # 乘员技能 [0,1]
    visibility: float = 1.0                 # 能见度 [0,1]
    weapon_condition: float = 1.0           # 武器状况 [0,1]
    max_range_m: float = 4000.0             # 有效射程
    fire_rate_rpm: float | None = None      # 覆盖武器射速（None 用武器定义）


@dataclass(frozen=True)
class FireRecord:
    """一次开火的记录（供测试/叙事层消费）。"""

    shooter_idx: int
    target_idx: int
    proj_idx: int
    weapon_id: str
    ammo_id: str
    range_m: float
    sigma_m: float
    aim_x: float
    aim_y: float
    aim_z: float


class FiringControl:
    """开火链交战管理器：fire + step（多分辨率子弹 Tick）。

    用法:
        fc = FiringControl(armor, boxes, projectiles)
        fc.ammo = AmmoManager(...)
        fc.set_unit_config(0, UnitCombatConfig(weapon_id="m256_120mm", armor_mm=600))
        fc.set_unit_config(1, UnitCombatConfig(weapon_id="2a46m_125mm", armor_mm=800))
        rec = fc.fire(0, 1)            # 开火
        fc.tick(1.0)                   # 推进 1s（内部按 0.01s 子弹 Tick 细分）
        events = fc.events.drain()     # 命中/摧毁事件
    """

    def __init__(
        self,
        armor: ArmorPool,
        boxes: BoxPool,
        projectiles: ProjectilePool,
        *,
        weapon_system: WeaponSystem | None = None,
        damage_resolver: DamageResolver | None = None,
        detector: CollisionDetector | None = None,
        formation_resolver: FormationResolver | None = None,
        rng: RNGManager | None = None,
        events: EventBus | None = None,
        ammo: AmmoManager | None = None,
        fuel: FuelManager | None = None,
        gravity: float = 9.81,
        # ---- Phase 6b 制导管线（docs/phase6-guidance-pipeline.md）----
        control: MissileControlRegistry | None = None,
        emitters: EmitterPool | None = None,
        terrain: Any | None = None,
        # ---- Phase 6b 步骤 8/9：末敏弹扫描 / 激光制导炸弹 ----
        sfm: SfmScanner | None = None,
        sal: SALSystem | None = None,
        # ---- Phase 6c 步骤③：爆炸/毁伤提交池（docs/phase6-ai-pipeline.md §七）----
        damage_pool: DamageSubmissionPool | None = None,
    ) -> None:
        self.armor = armor
        self.boxes = boxes
        self.projectiles = projectiles
        self.weapon_system = weapon_system or WeaponSystem(WeaponRegistry.with_builtin())
        self.damage_resolver = damage_resolver or DamageResolver()
        self.detector = detector or CollisionDetector()
        self.formation = formation_resolver or FormationResolver()
        self.rng = rng or RNGManager(seed=0)
        self.events = events if events is not None else EventBus()
        self.ammo = ammo
        self.fuel = fuel
        self.gravity = float(gravity)

        # ---- Phase 6b 制导管线 ----
        self.control = control or MissileControlRegistry.with_builtin()
        self.emitters = emitters or EmitterPool()
        # 可选：地形（TerrainMap / Heightfield 容器）——导引头 LOS 检测用；
        # None 时默认通视（无地形单测可用）
        self.terrain = terrain
        # 末敏弹双模扫描（docs §七）与激光制导炸弹（docs §7.5）
        self.sfm = sfm or SfmScanner(self.armor, self.emitters)
        self.sal = sal or SALSystem(self.emitters, self.terrain)
        # 爆炸/毁伤提交池（Phase 6c 步骤③；SAL 爆破/HE_FRAG 等区域毁伤先提交、Tick 末统一结算）
        self.damage_pool = damage_pool or DamageSubmissionPool()

        # 单位交战配置（armor_idx → UnitCombatConfig）
        self._configs: dict[int, UnitCombatConfig] = {}

        # 开火冷却（armor_idx → 剩余秒数）
        self._cooldown: dict[int, float] = {}

        # 抛射物元数据已下沉为 ProjectilePool SoA 数组（2026-08-11）：
        #   projectiles.weapon / projectiles.ammo / projectiles.age
        # 取代骨架期 _proj_meta / _proj_age dict（见 docs/ammunition-expansion-design.md §八）

    # ------------------------------------------------------------------
    # 单位配置
    # ------------------------------------------------------------------

    def set_unit_config(self, armor_idx: int, cfg: UnitCombatConfig) -> None:
        """为装甲登记交战属性。"""
        self._configs[armor_idx] = cfg

    def get_unit_config(self, armor_idx: int) -> UnitCombatConfig | None:
        return self._configs.get(armor_idx)

    def _config_of(self, armor_idx: int) -> UnitCombatConfig:
        try:
            return self._configs[armor_idx]
        except KeyError:
            raise KeyError(
                f"装甲 {armor_idx} 未登记交战配置（set_unit_config）"
            ) from None

    def can_fire(self, armor_idx: int) -> bool:
        """该装甲是否具备开火能力（有配置 + 有活跃 TURRET box + 弹药 > 0）。"""
        if not self.armor.active[armor_idx]:
            return False
        if armor_idx not in self._configs:
            return False
        if not self._has_turret(armor_idx):
            return False
        if self.ammo is not None and self.ammo.state.reserve[armor_idx] <= 0:
            return False
        return bool(self._cooldown.get(armor_idx, 0.0) <= 0.0)

    def _has_turret(self, armor_idx: int) -> bool:
        """该装甲是否仍有活跃炮塔（部位毁伤判定：TURRET 毁 → 无法开火）。"""
        b = self.boxes
        mask = b.active & (b.parent == armor_idx) & (b.part == int(PartType.TURRET))
        return bool(np.any(mask))

    # ------------------------------------------------------------------
    # 开火（开火链第 1~4 步 + 弹药消耗）
    # ------------------------------------------------------------------

    def fire(
        self,
        shooter_idx: int,
        target_idx: int,
        *,
        ammo_id: str | None = None,
        mods: dict[str, Any] | None = None,
        aim_at: tuple[float, float, float] | np.ndarray | None = None,
    ) -> FireRecord | None:
        """发射一发（统计层解算 → 高斯散布 → 抛射物生成 → 弹药扣减）。

        Args:
            shooter_idx: 射手装甲索引。
            target_idx: 目标装甲索引（aim_at 提供时可为占位）。
            ammo_id: 覆盖单位配置的弹药；None 用配置默认。
            mods: 透传给 WeaponSystem.compute_sigma_m 的修正因子。
            aim_at: 可选覆盖瞄准坐标（世界坐标）。SAL 炸弹炸桥时传光斑坐标
                （docs §7.5）；为 None 时瞄准 target_idx 装甲车体中心。

        Returns:
            FireRecord；若冷却中 / 无炮塔 / 无弹药 / 超射程返回 None。
        """
        if not self.can_fire(shooter_idx):
            return None

        cfg = self._config_of(shooter_idx)
        shooter_pos = self.armor.pos[shooter_idx]

        if aim_at is not None:
            # SAL 炸弹/地形目标：直接指定瞄准坐标（光斑坐标，§7.5）
            aim_x = float(aim_at[0])
            aim_y = float(aim_at[1])
            aim_z = float(aim_at[2])
            dx = aim_x - shooter_pos[0]
            dy = aim_y - shooter_pos[1]
            dz = aim_z - shooter_pos[2]
            range_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if range_m <= 1e-6 or range_m > cfg.max_range_m:
                return None
        else:
            if not self.armor.active[target_idx]:
                return None
            target_pos = self.armor.pos[target_idx]
            # 射距
            dx = target_pos[0] - shooter_pos[0]
            dy = target_pos[1] - shooter_pos[1]
            dz = target_pos[2] - shooter_pos[2]
            range_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if range_m <= 1e-6:
                return None
            if range_m > cfg.max_range_m:
                return None
            # 瞄准点：目标车体中心（水平面瞄准，z 抬到目标半高）
            aim_x = float(target_pos[0])
            aim_y = float(target_pos[1])
            aim_z = float(target_pos[2] + self.armor.half[target_idx, 2])

        # 开火解算（统计层：sigma + 初速）
        sol = self.weapon_system.solve(
            cfg.weapon_id, ammo_id or cfg.ammo_id, range_m,
            crew_skill=cfg.crew_skill,
            visibility=cfg.visibility,
            weapon_condition=cfg.weapon_condition,
            **(mods or {}),
        )

        # 高斯散布（统计层 → 物理落点，design doc §2.3）
        disp_x, disp_y = self.rng.dispersion_offset(sol.sigma_m)
        hit_x = aim_x + disp_x
        hit_y = aim_y + disp_y

        # 发射方向：射手炮口 → 散布落点
        ox = float(shooter_pos[0])
        oy = float(shooter_pos[1])
        oz = float(shooter_pos[2] + MUZZLE_HEIGHT)
        dir_x = hit_x - ox
        dir_y = hit_y - oy
        dir_z = aim_z - oz  # 落点高度 = 目标中心高度
        norm = float(np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z))
        if norm <= 1e-9:
            return None

        speed = sol.muzzle_velocity
        ammo_def = self.weapon_system.registry.get_ammo(sol.ammo_id)
        proj_idx = self.projectiles.spawn(
            ox, oy, oz,
            dir_x / norm * speed,
            dir_y / norm * speed,
            dir_z / norm * speed,
            radius=0.08,
            dmg=float(ammo_def.pen_mm_rha),
            kind=int(self._kind_of(sol.ammo_id)),
            owner=shooter_idx,
        )

        # 弹药消耗（每次开火 1 发）
        if self.ammo is not None:
            self.ammo.consume(shooter_idx, 1.0)

        # 开火冷却（射速 → 每发间隔秒）
        rpm = cfg.fire_rate_rpm
        if rpm is None:
            definition = self.weapon_system.registry.get_definition(sol.weapon_id, sol.ammo_id)
            rpm = definition.weapon.fire_rate_rpm
        self._cooldown[shooter_idx] = 60.0 / max(rpm, 1e-6)

        # 抛射物元数据（下沉 SoA 数组，取代 _proj_meta/_proj_age dict）
        self.projectiles.weapon[proj_idx] = self.weapon_system.registry.weapon_index(sol.weapon_id)
        self.projectiles.ammo[proj_idx] = self.weapon_system.registry.ammo_index(sol.ammo_id)
        self.projectiles.age[proj_idx] = 0.0

        # ---- 制导字段（Phase 6b，docs/phase6-guidance-pipeline.md §2.1/§7.1）----
        # guid_law = MissileControlRegistry 函数索引；**末敏弹也有索引**（sfm_ballistic），
        # BALLISTIC 阶段该函数不制导（直线 + 重力由内核施加）。stage/override 是传给
        # 控制回调的状态机输入。
        law_id = ammo_def.control_law_id
        if law_id:
            self.projectiles.guid_law[proj_idx] = self.control.index_of(law_id)
            self.projectiles.guid_target_pos[proj_idx] = (aim_x, aim_y, aim_z)  # 初始瞄准坐标
            self.projectiles.guid_nav_gain[proj_idx] = float(ammo_def.nav_gain)
            self.projectiles.guid_search_fov_cos[proj_idx] = float(
                np.cos(np.radians(ammo_def.search_fov_deg))
            )
            self.projectiles.guid_track_fov_cos[proj_idx] = float(
                np.cos(np.radians(ammo_def.track_fov_deg))
            )
            self.projectiles.guid_filter_quality[proj_idx] = float(ammo_def.filter_quality)
            # 末敏弹：初始阶段 = BALLISTIC（弹道飞行，函数不制导）；其余 = TRACK
            if ammo_def.ammo_type == AmmoType.SENSOR_FUZED:
                self.projectiles.guid_stage[proj_idx] = int(GuidanceStage.BALLISTIC)
            else:
                self.projectiles.guid_stage[proj_idx] = int(GuidanceStage.TRACK)
            # 主动雷达导引头：开机 → 激活辐射源（可被 ESM 检测 / ARM 反辐射，§6.1）
            if ammo_def.seeker == SeekerType.ACTIVE_RADAR:
                self.emitters.activate_from_projectile(
                    proj_idx,
                    float(ammo_def.emitter_power_db),
                    EmitterBand.RF,
                    EMITTER_KIND_SEEKER,
                )

        rec = FireRecord(
            shooter_idx=shooter_idx,
            target_idx=target_idx,
            proj_idx=proj_idx,
            weapon_id=sol.weapon_id,
            ammo_id=sol.ammo_id,
            range_m=range_m,
            sigma_m=sol.sigma_m,
            aim_x=aim_x,
            aim_y=aim_y,
            aim_z=aim_z,
        )
        self.events.emit(
            "fire",
            {
                "shooter": shooter_idx,
                "target": target_idx,
                "proj": proj_idx,
                "weapon": sol.weapon_id,
                "ammo": sol.ammo_id,
                "range": range_m,
                "sigma": sol.sigma_m,
            },
        )
        return rec

    def _kind_of(self, ammo_id: str) -> int:
        """弹药 → 抛射物类型（APFSDS/AP/APCR/PELE 为子弹，HEAT/ATGM/HE_FRAG/SAPHEI 为导弹）。

        Phase 6c 步骤③（docs/phase6-ai-pipeline.md §十一.4）：HE_FRAG/SAPHEI 有爆炸 → MISSILE，
        APCR/PELE 为动能细锥 → BULLET。
        """
        from Airis.lulilo.entities.pools import ProjectileType
        from Airis.lulilo.combat.weapons import AmmoType

        ammo = self.weapon_system.registry.get_ammo(ammo_id)
        if ammo.ammo_type in (
            AmmoType.HEAT,
            AmmoType.ATGM,
            AmmoType.HESH,
            AmmoType.HE_FRAG,
            AmmoType.SAPHEI,
        ):
            return int(ProjectileType.MISSILE)
        return int(ProjectileType.BULLET)

    # ------------------------------------------------------------------
    # 主步进（子弹 Tick 细分）
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """推进 dt 秒的交战（内部按 TICK_BULLET=0.01s 细分，防穿透天然正确）。

        Args:
            dt: 本帧累计时长（秒），由 SimulationLoop 的战斗 hook 传入。
        """
        # 冷却倒计时（所有已登记单位）
        for k in list(self._cooldown):
            self._cooldown[k] = max(0.0, self._cooldown[k] - dt)

        # 编队推开：防装甲重叠（空间哈希，AGENTS.md 3.4）
        # 每地面 Tick 调用一次（模拟层已积分移动，此处修正重叠）
        self.formation.resolve(self.armor.pos, self.armor.half, self.armor.active)

        acc = 0.0
        while acc + TICK_BULLET <= dt + 1e-9:
            self._step_sub(TICK_BULLET)
            acc += TICK_BULLET
        # 残余不足一个子弹 Tick 的时间并入最后一子步（保证总时长 = dt）
        if acc < dt - 1e-9:
            self._step_sub(dt - acc)

        # Phase 6c 步骤③：Tick 末统一结算伤害提交池（§七.3）
        self._resolve_damage_pool()

    def _step_sub(self, sub_dt: float) -> None:
        """一个子弹子步：阶段切换 → 弹道/制导积分 → 碰撞 → 毁伤 → 回收。"""
        self.boxes.sync_all_from_parent(self.armor)

        # 0. 辐射源同步（位置 + 级联失效 + 红外残热；导引头检测读最新坐标）
        self.emitters.sync_from_parent(self.armor, self.projectiles, sub_dt)

        # 0.5 SAL 光斑存活管理（吊舱 active 且 LOS 通；导引头检测读最新光斑状态）
        self.sal.update(sub_dt)

        # 1. 阶段切换（导引头检测 + 状态机；docs §五）
        self._update_guidance_stages()

        # 2. 弹道/制导积分：按 guid_law 分组制导积分 + 无制导直线积分（docs §四）
        self._integrate_guided(sub_dt)

        # 2.5 末敏弹 SCAN 双模扫描 → EFP 攻顶（docs §七，步骤 8）
        self._sfm_scan()

        # 2.6 SAL 炸弹命中地形判定 → 炸桥（docs §7.5，步骤 9）
        self._check_sal_impact()

        # 3. 碰撞检测（空间哈希 broad + Numba narrow）
        hits = self.detector.detect(
            self.projectiles.pos,
            self.projectiles.vel,
            self.projectiles.radius,
            self.projectiles.active,
            self.armor.pos,
            self.armor.half,
            self.armor.active,
            sub_dt,
        )

        # 4. 逐命中毁伤解析
        destroyed_proj: set[int] = set()
        for hit in hits:
            if hit.proj_idx in destroyed_proj:
                continue  # 同发弹丸只解析一次
            destroyed_proj.add(hit.proj_idx)
            self._resolve_hit(hit, sub_dt)

        # 5. 销毁已命中弹丸
        for proj_idx in destroyed_proj:
            self._destroy_projectile(proj_idx)

        # 6. 回收超龄 / 出界弹丸（脱靶）
        self._recycle_stale()

    def _integrate_guided(self, sub_dt: float) -> None:
        """按 guid_law 分组制导积分 + 无制导弹直线积分（docs §4.1）。

        - guid_law >= 0 的弹丸（含末敏弹，BALLISTIC 阶段函数不制导）按控制律分组，
          每组调一次 `integrate_guided_projectile_movement`（Numba 多版本缓存）。
        - guid_law == -1（普通子弹/无控制律）仍走直线积分，零行为差异。
        - sorted() 保证可复现（确定性已废弃，但保排序防测试抖动）。
        """
        p = self.projectiles
        active = p.active
        law_ids = sorted(
            {int(x) for x in np.unique(p.guid_law[active]) if x >= 0}
        )
        for lid in law_ids:
            fn = self.control.law_at(lid)
            if fn is None:
                continue
            integrate_guided_projectile_movement(
                p.pos, p.vel, p.active, p.age,
                p.guid_target_pos, p.guid_memory_pos, p.guid_stage,
                p.guid_nav_gain, p.guid_override,
                p.guid_search_fov_cos, p.guid_track_fov_cos,
                GUIDED_MAX_ACCEL, sub_dt, self.gravity, fn,
            )
        # 无制导弹（guid_law == -1）直线积分
        integrate_projectile_movement(p.pos, p.vel, p.active, sub_dt, self.gravity)

    # ------------------------------------------------------------------
    # 末敏弹扫描（Phase 6b 步骤 8，docs/phase6-guidance-pipeline.md §七）
    # ------------------------------------------------------------------

    def _sfm_scan(self) -> None:
        """SCAN 阶段弹丸双模扫描 → EFP 攻顶 → 销毁弹丸（docs §7）。

        扫描探测在 combat/sfm.py（scan_armor_detect 内核 + SfmScanner，按 ammo
        分组传扫描参数）；毁伤在本方法（复用 DamageResolver，§7.3）。
        """
        hits = self.sfm.scan(self.projectiles, self._ammo_of)
        for proj_idx, armor_idx in hits.items():
            self._resolve_sfm_hit(proj_idx, armor_idx)

    def _resolve_sfm_hit(self, proj_idx: int, armor_idx: int) -> None:
        """末敏弹 EFP 攻顶毁伤（docs §7.3）——保留直接命中路径。

        顶部 = 正面 × top_armor_factor，垂直攻顶（impact_angle_deg=0），
        EFP 固定穿深（is_heat 化学能语义、不随距离衰减）。命中部位 = 最近 box。
        毁伤核心已抽 `_resolve_top_attack`（docs §五.3），供攻顶 ATGM 共用。
        """
        if not self.armor.active[armor_idx]:
            self._destroy_projectile(proj_idx)
            return
        ammo = self._ammo_of(proj_idx)
        if ammo is None:
            return
        cfg = self._configs.get(armor_idx)
        if cfg is None:
            return

        top_factor = float(ammo.top_armor_factor)
        result = self._resolve_top_attack(
            armor_idx, float(ammo.efp_pen_mm), top_factor, cause="sfm"
        )
        self.events.emit(
            "sfm_detonate",
            {
                "proj": proj_idx,
                "target": armor_idx,
                "top_mm": cfg.armor_mm * top_factor,
                "efp_pen": ammo.efp_pen_mm,
                "margin": result["margin"],
                "penetrated": result["margin"] > 0 and not result["ricochet"],
            },
        )
        self._destroy_projectile(proj_idx)

    def _resolve_top_attack(
        self, armor_idx: int, pen_mm: float, top_factor: float, cause: str
    ) -> dict[str, float | bool]:
        """攻顶毁伤公共函数（docs §五.3 / §七.3）。

        顶部 = 正面 × top_factor，垂直攻顶（impact_angle_deg=0），固定穿深
        （is_heat 化学能语义、不随距离衰减）。SFM 直接路径与提交池 CONE 事件
        共用（步骤④ 攻顶 ATGM 也走此函数）。返回 DamageResolver.resolve 结果。
        """
        cfg = self._configs.get(armor_idx)
        if cfg is None:
            return {
                "armor_eff": 0.0,
                "margin": -1e9,
                "ricochet": True,
                "health_dmg": 0.0,
                "fire": False,
                "cookoff": False,
            }
        top_mm = cfg.armor_mm * top_factor
        result = self.damage_resolver.resolve(
            armor_idx=armor_idx,
            armor_mm=top_mm,
            impact_angle_deg=0.0,               # 垂直攻顶
            range_m=0.0,
            pen_mm_rha=pen_mm,                  # EFP/攻顶 固定穿深
            is_heat=True,                       # 爆炸成型，化学能语义
            drag_coefficient=0.0,               # 穿深不随距离衰减
            muzzle_velocity=0.0,
            rng_u=self.rng.uniform(),
        )
        if not result["ricochet"]:
            health = float(self.armor.health[armor_idx]) - float(result["health_dmg"])
            self.armor.health[armor_idx] = max(0.0, health)
            if result["margin"] > 0:
                part = self._resolve_top_part(armor_idx)
                self._apply_part_damage(armor_idx, part, result["cookoff"])
            if result["cookoff"] or self.armor.health[armor_idx] <= 0.0:
                self._destroy_armor(
                    armor_idx, cause=cause if result["cookoff"] else "health"
                )
        return result

    def _resolve_top_part(self, armor_idx: int) -> int:
        """攻顶命中部位：顶部投影 → 简化取最近 box（docs §7.3 fallback 分支）。"""
        b = self.boxes
        ax = float(self.armor.pos[armor_idx, 0])
        ay = float(self.armor.pos[armor_idx, 1])
        az = float(self.armor.pos[armor_idx, 2])
        best_d = 1e18
        best = int(PartType.HULL)
        for i in range(b.capacity):
            if not b.active[i] or b.parent[i] != armor_idx:
                continue
            cx = float(b.pos[i, 0])
            cy = float(b.pos[i, 1])
            cz = float(b.pos[i, 2])
            hbx = float(b.half[i, 0])
            hby = float(b.half[i, 1])
            hbz = float(b.half[i, 2])
            dx = max(0.0, abs(ax - cx) - hbx)
            dy = max(0.0, abs(ay - cy) - hby)
            dz = max(0.0, abs(az - cz) - hbz)
            d = dx * dx + dy * dy + dz * dz
            if d < best_d:
                best_d = d
                best = int(b.part[i])
        return best

    # ------------------------------------------------------------------
    # 激光制导炸弹（Phase 6b 步骤 9，docs/phase6-guidance-pipeline.md §7.5）
    # ------------------------------------------------------------------

    def _check_sal_impact(self) -> None:
        """SAL 炸弹命中判定（弹丸 vs 地形，docs §7.5）。

        炸弹到达 guid_target_pos 附近（< SAL_DETONATE_RADIUS）→ 引爆 + 破坏地形。
        目标坐标恒为光斑坐标（桥不动，无目标机动）；碰撞检测对地形不命中，
        故此处单独判引爆（弹丸 vs 装甲仍走既有 CollisionDetector）。
        """
        p = self.projectiles
        for proj in p.active_indices():
            ammo = self._ammo_of(proj)
            if ammo is None or ammo.seeker != SeekerType.SEMI_ACTIVE_LASER:
                continue
            tpos = p.guid_target_pos[proj]
            pp = p.pos[proj]
            dx = pp[0] - tpos[0]
            dy = pp[1] - tpos[1]
            dz = pp[2] - tpos[2]
            if dx * dx + dy * dy + dz * dz <= SAL_DETONATE_RADIUS ** 2:
                self._detonate_sal(
                    proj, float(tpos[0]), float(tpos[1]), float(tpos[2])
                )

    def _detonate_sal(
        self, proj_idx: int, tx: float, ty: float, tz: float
    ) -> None:
        """SAL 炸弹引爆（docs §7.5）：破坏地形 + 近距毁伤 + 销毁弹丸。"""
        # 1. 破坏地形：以目标坐标为中心挖掉柱体（桥面 remove_span，自由破坏路径）
        if self.terrain is not None:
            self._remove_terrain_crater(tx, ty, tz)
        # 2. 近距毁伤：爆炸半径内装甲（线性衰减）
        self._sal_blast_damage(tx, ty, tz)
        # 3. 事件 + 销毁弹丸（级联关闭 emitter）
        self.events.emit(
            "sal_detonate", {"proj": proj_idx, "x": tx, "y": ty, "z": tz}
        )
        self._destroy_projectile(proj_idx)

    def _remove_terrain_crater(
        self, x: float, y: float, tz: float
    ) -> None:
        """以世界坐标 (x,y) 为中心挖掉半径 SAL_CRATER_RADIUS 的柱体（§7.5）。

        挖 [tz - SAL_CRATER_DEPTH, tz + SAL_CRATER_ABOVE) 区间——目标在桥面时
        桥面 span 被截断（桥断），目标在地面时地面被挖坑。随后失效几何/导航缓存。
        """
        hf = self.terrain.heightfield
        cs = hf.cs
        r = SAL_CRATER_RADIUS
        gx0 = int(np.floor((x - r) / cs))
        gx1 = int(np.floor((x + r) / cs))
        gy0 = int(np.floor((y - r) / cs))
        gy1 = int(np.floor((y + r) / cs))
        z0 = tz - SAL_CRATER_DEPTH
        z1 = tz + SAL_CRATER_ABOVE
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                if not (0 <= gx < hf.width and 0 <= gy < hf.height):
                    continue
                hf.remove_span(gx, gy, z0, z1)
        # 地形破坏 → 导航/几何缓存失效（下次访问重建，AGENTS.md §3.3）
        invalidate = getattr(self.terrain, "invalidate_geometry", None)
        if callable(invalidate):
            invalidate()

    def _sal_blast_damage(self, tx: float, ty: float, tz: float) -> None:
        """SAL 爆炸近距毁伤：提交球形爆破事件进提交池（Tick 末统一结算，§七.3）。

        原实现即时遍历装甲线性衰减；现改为提交 `DamageKind.SPHERE` 事件，
        由 `_resolve_damage_pool` 统一批量结算（Numba 内核原地扣血）+ 补摧毁。
        """
        self.damage_pool.submit_damage(
            DamageKind.SPHERE,
            (tx, ty, tz),
            radius=SAL_CRATER_RADIUS * 3.0,
            pen_mm=SAL_BLAST_DMG,
        )

    # ------------------------------------------------------------------
    # 伤害提交池结算（Phase 6c 步骤③，docs/phase6-ai-pipeline.md §七.3）
    # ------------------------------------------------------------------

    def _resolve_damage_pool(self) -> None:
        """Tick 末统一结算伤害提交池（§七.3）。

        1. 区域毁伤（球形/破片锥）：Numba 批量内核原地扣 armor.health（O(E×N)）
        2. 区域毁伤导致的摧毁：Python 补 `_destroy_armor`（内核只扣血）
        3. 锥形/细锥（直接目标）：逐事件顶级攻顶毁伤（复用 `_resolve_top_attack`）
        结算后清空事件池（`clear` 归还索引）。
        """
        pool = self.damage_pool
        if pool is None or pool.active_count == 0:
            return

        # 1. 区域毁伤（球形/破片锥）：Numba 批量结算
        if pool.active_count > 0:
            resolve_area_damage(
                pool.kind,
                pool.center,
                pool.radius,
                pool.pen_mm,
                pool.direction,
                pool.cone_angle,
                pool.frag_count,
                pool.active,
                self.armor.pos,
                self.armor.health,
                self.armor.active,
            )

        # 2. 区域毁伤导致的摧毁（Numba 只扣血，Python 补 destroy）
        self._finalize_area_kills()

        # 3. 锥形/细锥（直接目标）：逐事件顶级攻顶毁伤
        for ev in pool.active_indices():
            ev = int(ev)
            k = int(pool.kind[ev])
            if k not in (int(DamageKind.CONE), int(DamageKind.FINE_CONE)):
                continue
            tid = int(pool.target_idx[ev])
            if not (0 <= tid < self.armor.capacity) or not self.armor.active[tid]:
                continue
            tf = float(pool.top_factor[ev])
            self._resolve_top_attack(
                tid,
                float(pool.pen_mm[ev]),
                tf if tf > 0.0 else 1.0,
                cause="cone",
            )

        pool.clear()

    def _finalize_area_kills(self) -> None:
        """区域毁伤后清理：health<=0 的装甲走既有销毁路径（事件/级联/后勤）。"""
        a = self.armor
        low = a.active & (a.health <= 0.0)
        for idx in np.flatnonzero(low):
            self._destroy_armor(int(idx), cause="blast")

    # ------------------------------------------------------------------
    # 阶段状态机（Phase 6b：guid_stage / guid_override 传给控制回调的输入）
    # ------------------------------------------------------------------
    # docs/phase6-guidance-pipeline.md §五：
    #   TRACK → MEMORY（检测丢失：ECM/遮挡/辐射源关机）
    #   TRACK → TERMINAL（末端距离且双模导引头锁定本体）
    #   MEMORY → TERMINAL（记忆飞行中重新检测到本体）
    #   BALLISTIC → SCAN（末敏弹，高度 ≤ scan_altitude_m）
    # 制导「丢失」只由 detection 层决定（§6.5）；guid_confidence 与制导无关（Phase 8）。

    def _update_guidance_stages(self) -> None:
        """每子步：导引头检测（写 guid_target_pos）+ 阶段切换（docs §5.2）。

        导弹数量少，逐弹丸 O(k) Python 扫描（不进 Numba 内核，陷阱 §八.12）。
        """
        detected = self._seeker_detect()
        p = self.projectiles
        for proj in p.active_indices():
            stage = int(p.guid_stage[proj])
            ammo = self._ammo_of(proj)
            if ammo is None:
                continue

            if stage in (GuidanceStage.TRACK, GuidanceStage.TERMINAL):
                if detected.get(proj, False):
                    # 检测成功 → guid_target_pos 已是新坐标；末端距离内锁定本体
                    if stage == GuidanceStage.TRACK and self._terminal_ready(proj, ammo):
                        self._relock_body(proj)
                else:
                    # 检测丢失（ECM/遮挡/辐射源关机）→ 记忆模式或直接失的
                    if ammo.memory_mode:
                        p.guid_memory_pos[proj] = p.guid_target_pos[proj]
                        p.guid_stage[proj] = int(GuidanceStage.MEMORY)
                    else:
                        self._stop_guidance(proj)

            elif stage == GuidanceStage.MEMORY:
                if detected.get(proj, False) and self._terminal_ready(proj, ammo):
                    self._relock_body(proj)  # 记忆飞行中重新检测到本体 → 末端引导
                elif self._memory_expired(proj):
                    self._stop_guidance(proj)  # 已到记忆点未命中 → 停导（脱靶回收）

            # 末敏弹：BALLISTIC → SCAN（弹道阶段函数不制导，开伞后悬停扫描）
            if stage == GuidanceStage.BALLISTIC and p.pos[proj, 2] <= ammo.scan_altitude_m:
                p.guid_stage[proj] = int(GuidanceStage.SCAN)
                self.emitters.activate_from_projectile(
                    proj, float(ammo.mmw_emitter_power_db), EmitterBand.RF, EMITTER_KIND_SEEKER
                )
                p.age[proj] = 0.0  # 扫描窗口计时复用 age（_recycle_stale 按 scan_duration_s 判超时）

    def _seeker_detect(self) -> dict[int, bool]:
        """导引头检测（docs §6.5）：对 TRACK/MEMORY/TERMINAL 弹丸做逐 tick 检测。

        返回 {proj_idx: detected}；检测成功时 guid_target_pos 已更新为新坐标。
        检测链：目标选择（锁定坐标附近最近合格目标，防跳变）→ LOS → 检测概率
        （× ECM 干扰）→ RNG 采样 → 写坐标（滤波加噪声）。
        """
        p = self.projectiles
        detected: dict[int, bool] = {}
        for proj in p.active_indices():
            ammo = self._ammo_of(proj)
            if ammo is None or ammo.seeker == SeekerType.NONE:
                continue
            stage = int(p.guid_stage[proj])
            if stage not in (
                GuidanceStage.TRACK,
                GuidanceStage.MEMORY,
                GuidanceStage.TERMINAL,
            ):
                continue
            if self._seeker_probe(proj, ammo):
                detected[proj] = True
        return detected

    def _seeker_probe(self, proj_idx: int, ammo: Any) -> bool:
        """对单枚弹丸做一次检测尝试（按 seeker 类型分派目标源，§6.5）。"""
        p = self.projectiles
        sx = float(p.pos[proj_idx, 0])
        sy = float(p.pos[proj_idx, 1])
        sz = float(p.pos[proj_idx, 2])
        ref = p.guid_target_pos[proj_idx]
        fq = float(p.guid_filter_quality[proj_idx])
        seeker = ammo.seeker

        # --- 主动发射-接收：主动雷达导引头（目标 = 装甲，RCS 反射）---
        if seeker == SeekerType.ACTIVE_RADAR:
            t = self._find_nearest_armor(ref, SEEK_RADIUS_M)
            if t < 0:
                return False
            tx = float(self.armor.pos[t, 0])
            ty = float(self.armor.pos[t, 1])
            tz = float(self.armor.pos[t, 2])
            if not self._los(sx, sy, sz, tx, ty, tz):
                return False
            dist = self._dist3(sx, sy, sz, tx, ty, tz)
            # 干扰闭环：目标 ecm 压制检测距离（陷阱 §八.6）
            max_r = radar_detection_range(float(self.armor.rcs[t])) * (
                1.0 - float(self.armor.ecm[t])
            )
            prob = radar_detection_probability(dist, max_r)
            if self.rng.uniform() >= prob:
                return False
            self._write_detected_pos(proj_idx, tx, ty, tz, fq)
            return True

        # --- 被动接收：红外成像（目标 = 红外 emitter，cur_power 含残热）---
        if seeker in (SeekerType.IR_IMAGING, SeekerType.TV_EO):
            e = self._find_emitter(ref, EmitterBand.IR, EMITTER_KIND_IR_SOURCE, SEEK_RADIUS_M)
            if e < 0:
                return False
            ex = float(self.emitters.pos[e, 0])
            ey = float(self.emitters.pos[e, 1])
            ez = float(self.emitters.pos[e, 2])
            if not self._los(sx, sy, sz, ex, ey, ez):
                return False
            dist = self._dist3(sx, sy, sz, ex, ey, ez)
            max_r = ir_detection_range(float(self.emitters.cur_power[e]))
            prob = ir_detection_probability(dist, max_r)
            if self.rng.uniform() >= prob:
                return False
            self._write_detected_pos(proj_idx, ex, ey, ez, fq)
            return True

        # --- 被动接收：反辐射（ARM，目标 = RF emitter）---
        if seeker == SeekerType.PASSIVE_RF:
            e = self._find_emitter(ref, EmitterBand.RF, -1, SEEK_RADIUS_M)
            if e < 0:
                return False
            ex = float(self.emitters.pos[e, 0])
            ey = float(self.emitters.pos[e, 1])
            ez = float(self.emitters.pos[e, 2])
            if not self._los(sx, sy, sz, ex, ey, ez):
                return False
            dist = self._dist3(sx, sy, sz, ex, ey, ez)
            max_r = esm_detection_range(float(self.emitters.power_db[e]))
            prob = esm_detection_probability(dist, max_r)
            if self.rng.uniform() >= prob:
                return False
            self._write_detected_pos(proj_idx, ex, ey, ez, fq)
            return True

        # --- 被动接收：半主动激光（目标 = 漫反射光斑，半球 normal 限制）---
        if seeker == SeekerType.SEMI_ACTIVE_LASER:
            e = self._find_emitter(ref, EmitterBand.LASER, EMITTER_KIND_SPOT, SEEK_RADIUS_M)
            if e < 0:
                return False
            ex = float(self.emitters.pos[e, 0])
            ey = float(self.emitters.pos[e, 1])
            ez = float(self.emitters.pos[e, 2])
            # 半球漫反射（Lambert）：导引头须在光斑法线半球内（点积 > 0），
            # 在目标背面/被遮挡则检测不到（docs §7.5）
            nx = float(self.emitters.normal[e, 0])
            ny = float(self.emitters.normal[e, 1])
            nz = float(self.emitters.normal[e, 2])
            if nx * (sx - ex) + ny * (sy - ey) + nz * (sz - ez) <= 0.0:
                return False
            if not self._los(sx, sy, sz, ex, ey, ez):
                return False
            self._write_detected_pos(proj_idx, ex, ey, ez, fq)
            return True

        return False  # 未知 seeker 类型：不检测

    # ------------------------------------------------------------------
    # 阶段状态机辅助
    # ------------------------------------------------------------------

    def _ammo_of(self, proj_idx: int) -> Any:
        """弹丸 → 弹药配置（由 projectiles.ammo 注册表索引反查）。"""
        ai = int(self.projectiles.ammo[proj_idx])
        return self.weapon_system.registry.ammo_at(ai) if ai >= 0 else None

    def _find_nearest_armor(self, ref: np.ndarray, seek_radius: float) -> int:
        """在 ref（锁定坐标）附近 seek_radius 内搜最近装甲（目标机动时坐标自然跟随）。"""
        a = self.armor
        best = -1
        best_d = seek_radius * seek_radius
        for idx in a.active_indices():
            dx = a.pos[idx, 0] - ref[0]
            dy = a.pos[idx, 1] - ref[1]
            dz = a.pos[idx, 2] - ref[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= best_d:
                best_d = d2
                best = int(idx)
        return best

    def _find_emitter(
        self, ref: np.ndarray, band: int, kind_filter: int, seek_radius: float
    ) -> int:
        """在 ref 附近 seek_radius 内搜最近合格 emitter（band + 可选 kind 过滤）。"""
        em = self.emitters
        best = -1
        best_d = seek_radius * seek_radius
        for e in em.active_indices():
            if em.band[e] != int(band):
                continue
            if kind_filter >= 0 and em.kind[e] != kind_filter:
                continue
            dx = em.pos[e, 0] - ref[0]
            dy = em.pos[e, 1] - ref[1]
            dz = em.pos[e, 2] - ref[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= best_d:
                best_d = d2
                best = int(e)
        return best

    def _write_detected_pos(
        self, proj_idx: int, x: float, y: float, z: float, filter_quality: float
    ) -> None:
        """写检测坐标（滤波加噪声：σ ∝ (1-filter_quality)，体现导弹优劣）。"""
        sigma = max(0.0, 1.0 - filter_quality) * 2.0
        if sigma > 1e-6:
            x += self.rng.normal(0.0, sigma)
            y += self.rng.normal(0.0, sigma)
        self.projectiles.guid_target_pos[proj_idx] = (x, y, z)

    def _terminal_ready(self, proj_idx: int, ammo: Any) -> bool:
        """末端引导触发：terminal_seeker 有效且弹丸到锁定坐标 < terminal_range_m。"""
        if ammo.terminal_seeker == SeekerType.NONE or ammo.terminal_range_m <= 0.0:
            return False
        p = self.projectiles.pos[proj_idx]
        t = self.projectiles.guid_target_pos[proj_idx]
        dx = t[0] - p[0]
        dy = t[1] - p[1]
        dz = t[2] - p[2]
        return dx * dx + dy * dy + dz * dz <= ammo.terminal_range_m ** 2

    def _relock_body(self, proj_idx: int) -> None:
        """末端锁定本体：检测成功时 guid_target_pos 已是最新本体坐标 → 仅切阶段。"""
        self.projectiles.guid_stage[proj_idx] = int(GuidanceStage.TERMINAL)

    def _memory_expired(self, proj_idx: int) -> bool:
        """记忆模式耗尽：已到达记忆点附近未命中，或记忆飞行超时 → 停导（脱靶回收）。"""
        p = self.projectiles
        mp = p.guid_memory_pos[proj_idx]
        pp = p.pos[proj_idx]
        dx = pp[0] - mp[0]
        dy = pp[1] - mp[1]
        dz = pp[2] - mp[2]
        if dx * dx + dy * dy + dz * dz < 400.0:  # 20m 内
            return True
        return p.age[proj_idx] > 15.0

    def _stop_guidance(self, proj_idx: int) -> None:
        """停导：guid_law=-1 → 走直线积分（脱靶由 _recycle_stale 自然回收）。"""
        self.projectiles.guid_law[proj_idx] = -1

    def _set_manual_override(self, proj_idx: int, override: ManualOverride) -> None:
        """人在回路推进指令（docs §3.5）：先判定 guid_comm 通信门控。

        门控失败（guid_comm < ammo.comm_threshold）→ 丢弃，导弹继续自主飞行；
        门控通过 → 写 guid_override（控制回调消费 HOLD/REIGNITE）。
        改目标 / 自毁不占位——由叙事层直接改 guid_target_pos / destroy。
        """
        ammo = self._ammo_of(proj_idx)
        if ammo is None:
            return
        if float(self.projectiles.guid_comm[proj_idx]) >= ammo.comm_threshold:
            self.projectiles.guid_override[proj_idx] = int(override)

    def _los(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> bool:
        """导引头 ↔ 目标坐标视线（无 terrain 时默认通视，供无地形单测）。"""
        if self.terrain is None:
            return True
        hf = self.terrain.heightfield
        return line_of_sight(
            x0, y0, z0, x1, y1, z1,
            hf.spans, hf.count, hf.width, hf.cs, hf.ch, hf.z_min,
        )

    @staticmethod
    def _dist3(
        x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
    ) -> float:
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # ------------------------------------------------------------------
    # 命中解析（开火链第 5~8 步）
    # ------------------------------------------------------------------

    def _resolve_hit(self, hit: CollisionHit, sub_dt: float) -> None:
        """解析一次命中：部位判定 → 斜距/穿透 → 后效毁伤 → SoA 更新。"""
        proj_idx = hit.proj_idx
        armor_idx = hit.armor_idx

        # 忽略命中发射者自身（近距自伤；防穿透下一般不会发生，但双保险）
        owner = int(self.projectiles.owner[proj_idx])
        if owner == armor_idx:
            return

        p1 = self.projectiles.pos[proj_idx]
        vel = self.projectiles.vel[proj_idx]
        p0 = p1 - vel * sub_dt

        # 命中部位（BoxPool 精确判定，fallback 到最近 box）
        part = self._resolve_hit_part(hit, p0, p1)

        # 弹药参数（下沉 SoA 数组反查，取代 _proj_meta dict）
        ammo_idx = int(self.projectiles.ammo[proj_idx])
        ammo = self.weapon_system.registry.ammo_at(ammo_idx) if ammo_idx >= 0 else None
        # 事件/日志用的 id 字符串（-1 → 空串）
        weapon_id = ""
        ammo_id = ""
        if ammo_idx >= 0:
            weapon_idx = int(self.projectiles.weapon[proj_idx])
            wdef = self.weapon_system.registry.weapon_at(weapon_idx) if weapon_idx >= 0 else None
            if wdef is not None:
                weapon_id = wdef.weapon_id
            ammo_id = ammo.ammo_id if ammo is not None else ""

        cfg = self._config_of(armor_idx) if armor_idx in self._configs else None
        if cfg is None:
            return

        # 部位等效装甲厚度
        armor_mm = cfg.armor_mm * PART_ARMOR_FACTOR.get(part, 1.0)

        # 入射角（速度方向 vs 命中点所在表面法线，正击=0°，擦过=90°）
        impact_angle = self._impact_angle(
            (hit.hit_x, hit.hit_y, hit.hit_z), vel, armor_idx
        )

        if ammo is None:
            pen_rha = float(self.projectiles.dmg[proj_idx])
            is_heat = False
            drag_coeff = 0.15
            muzzle_vel = 1500.0
        else:
            pen_rha = ammo.pen_mm_rha
            is_heat = ammo.is_heat
            drag_coeff = ammo.drag_coefficient
            muzzle_vel = ammo.muzzle_velocity

        range_m = self._range_of(proj_idx, armor_idx)

        # 毁伤解析（斜距 + 穿透 + 后效）
        result = self.damage_resolver.resolve(
            armor_idx=armor_idx,
            armor_mm=armor_mm,
            impact_angle_deg=impact_angle,
            range_m=range_m,
            pen_mm_rha=pen_rha,
            is_heat=is_heat,
            drag_coefficient=drag_coeff,
            muzzle_velocity=muzzle_vel,
            rng_u=self.rng.uniform(),
        )

        self.events.emit(
            "hit",
            {
                "proj": proj_idx,
                "target": armor_idx,
                "shooter": owner,
                "part": part,
                "weapon": weapon_id,
                "ammo": ammo_id,
                "armor_eff": result["armor_eff"],
                "margin": result["margin"],
                "penetrated": result["margin"] > 0 and not result["ricochet"],
                "health_dmg": result["health_dmg"],
                "fire": result["fire"],
            },
        )

        if result["ricochet"]:
            self.events.emit(
                "ricochet",
                {"proj": proj_idx, "target": armor_idx, "part": part},
            )
            return  # 跳弹：弹丸继续飞行（保留动能），本次不造成毁伤

        # 结构损伤累积
        health = float(self.armor.health[armor_idx]) - float(result["health_dmg"])
        self.armor.health[armor_idx] = max(0.0, health)

        penetrated = result["margin"] > 0
        if penetrated:
            self._apply_part_damage(armor_idx, part, result["cookoff"])

        if result["cookoff"] or self.armor.health[armor_idx] <= 0.0:
            self._destroy_armor(armor_idx, cause="cookoff" if result["cookoff"] else "health")

        # Phase 6c 步骤③：爆炸/破片弹药（HE_FRAG/SAPHEI 等）→ 提交池（§七.3）
        # 即使目标被击毁也爆炸（HE 杀伤链：穿透 + 爆破并存）
        if ammo is not None and ammo.blast_radius > 0.0:
            self._submit_hit_blast(ammo, hit, proj_idx)

    def _submit_hit_blast(self, ammo: Any, hit: CollisionHit, proj_idx: int) -> None:
        """命中点爆炸/破片 → 提交池（docs §七.3 / §十一.3）。

        以命中点为球心提交球形爆破事件（线性衰减）；破片（frag_count>0）加提交
        破片锥事件（方向 = 弹丸飞行方向，半角 30°，射程 = blast_radius）。
        由 `_resolve_damage_pool` 于 Tick 末统一结算。
        """
        cx = float(hit.hit_x)
        cy = float(hit.hit_y)
        cz = float(hit.hit_z)
        if ammo.blast_radius > 0.0 and ammo.blast_dmg > 0.0:
            self.damage_pool.submit_damage(
                DamageKind.SPHERE,
                (cx, cy, cz),
                radius=float(ammo.blast_radius),
                pen_mm=float(ammo.blast_dmg),
            )
        if ammo.frag_count > 0 and ammo.frag_dmg > 0.0:
            vel = self.projectiles.vel[proj_idx]
            vx = float(vel[0])
            vy = float(vel[1])
            vz = float(vel[2])
            vn = float(np.sqrt(vx * vx + vy * vy + vz * vz))
            if vn > 1e-9:
                self.damage_pool.submit_damage(
                    DamageKind.FRAG_CONE,
                    (cx, cy, cz),
                    radius=float(ammo.blast_radius),
                    pen_mm=float(ammo.frag_dmg),
                    direction=(vx / vn, vy / vn, vz / vn),
                    cone_angle=float(np.radians(30.0)),
                    frag_count=int(ammo.frag_count),
                )

    def _apply_part_damage(self, armor_idx: int, part: int, cookoff: bool) -> None:
        """击穿后的部位毁伤（box 失活映射，design doc §5.3）。

        AMMO_RACK 被击穿：直接殉爆秒杀（cookoff 已由 resolver 判定）。
        TURRET / ENGINE / TRACK：失活对应 box（失去开火 / 机动 / 减速）。
        """
        b = self.boxes
        mask = b.active & (b.parent == armor_idx) & (b.part == part)
        idxs = np.flatnonzero(mask)
        if idxs.size:
            b.active[idxs] = False
            for bi in idxs:
                self.boxes._free_stack.append(int(bi))
            # 失活后 parent 清空（防止误用）
            b.parent[idxs] = -1

    def _destroy_armor(self, armor_idx: int, cause: str = "destroyed") -> None:
        """摧毁装甲：health 清零 + 失活所有 box + 弹药/燃料 release + 事件。"""
        if not self.armor.active[armor_idx]:
            return
        self.armor.health[armor_idx] = 0.0
        self.armor.destroy(armor_idx)

        # Phase 6b：级联关闭该装甲的主动 emitter（红外除外，走残热）
        self._close_armor_emitters(armor_idx)

        b = self.boxes
        mask = b.active & (b.parent == armor_idx)
        idxs = np.flatnonzero(mask)
        if idxs.size:
            b.active[idxs] = False
            b.parent[idxs] = -1
            self.boxes._free_stack.extend(idxs.tolist())

        if self.ammo is not None:
            self.ammo.state.release(armor_idx)
        if self.fuel is not None:
            self.fuel.state.release(armor_idx)

        self._configs.pop(armor_idx, None)
        self._cooldown.pop(armor_idx, None)

        self.events.emit("destroyed", {"armor_idx": armor_idx, "cause": cause})

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _resolve_hit_part(self, hit: CollisionHit, p0: np.ndarray, p1: np.ndarray) -> int:
        """命中部位：对命中装甲的所有活跃 box 做胶囊体检测，取 t 最小者。

        若精确 box 均未命中（近似 AABB 偏差），fallback 到离命中点最近的 box。
        """
        armor_idx = hit.armor_idx
        boxes = self.boxes
        radius = float(self.projectiles.radius[hit.proj_idx])

        best_t = 1e9
        best_part = int(PartType.HULL)
        found = False
        for b in range(boxes.capacity):
            if not boxes.active[b] or boxes.parent[b] != armor_idx:
                continue
            hb, t = segment_aabb_single(
                p0[0], p0[1], p0[2],
                p1[0], p1[1], p1[2],
                boxes.pos[b, 0], boxes.pos[b, 1], boxes.pos[b, 2],
                boxes.half[b, 0], boxes.half[b, 1], boxes.half[b, 2],
                radius,
            )
            if hb and t < best_t:
                best_t = t
                best_part = int(boxes.part[b])
                found = True

        if found:
            return best_part

        # Fallback：最近 box
        hx = hit.hit_x
        hy = hit.hit_y
        hz = hit.hit_z
        best_d = 1e18
        best = int(PartType.HULL)
        for b in range(boxes.capacity):
            if not boxes.active[b] or boxes.parent[b] != armor_idx:
                continue
            cx, cy, cz = boxes.pos[b]
            hbx, hby, hbz = boxes.half[b]
            # 命中点到 box 表面的近似距离（AABB 距离）
            dx = max(0.0, abs(hx - cx) - hbx)
            dy = max(0.0, abs(hy - cy) - hby)
            dz = max(0.0, abs(hz - cz) - hbz)
            d = dx * dx + dy * dy + dz * dz
            if d < best_d:
                best_d = d
                best = int(boxes.part[b])
        return best

    def _impact_angle(
        self, hit_pos: tuple[float, float, float], vel: np.ndarray, armor_idx: int
    ) -> float:
        """入射角（度）：弹丸方向 vs 命中点所在表面法线，正击≈0°，擦过≈90°。

        命中点转入装甲局部坐标（world_to_local），按占满度判断打的是
        正面/背面（法线沿局部 x）还是侧面（法线沿局部 y），再算夹角。
        —— 正面射击迎面即正击；侧面射击侧面即正击（物理正确）。
        """
        ax = float(self.armor.pos[armor_idx, 0])
        ay = float(self.armor.pos[armor_idx, 1])
        yaw = float(self.armor.yaw[armor_idx])

        hx, hy = world_to_local(hit_pos[0], hit_pos[1], ax, ay, yaw)
        vx, vy = world_to_local(vel[0], vel[1], ax, ay, yaw)
        vn = float(np.sqrt(vx * vx + vy * vy))
        if vn <= 1e-9:
            return 0.0

        hxf = max(float(self.armor.half[armor_idx, 0]), 1e-6)
        hyf = max(float(self.armor.half[armor_idx, 1]), 1e-6)
        if abs(hx) / hxf >= abs(hy) / hyf:
            cos_a = abs(vx) / vn          # 正面/背面：法线沿局部 x
        else:
            cos_a = abs(vy) / vn          # 侧面：法线沿局部 y
        cos_a = max(0.0, min(1.0, cos_a))
        return float(np.degrees(np.arccos(cos_a)))

    def _range_of(self, proj_idx: int, armor_idx: int) -> float:
        """射距（弹丸当前到目标距离，用于穿深衰减估算）。"""
        p = self.projectiles.pos[proj_idx]
        a = self.armor.pos[armor_idx]
        dx = p[0] - a[0]
        dy = p[1] - a[1]
        dz = p[2] - a[2]
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _destroy_projectile(self, proj_idx: int) -> None:
        """销毁弹丸（元数据存于 SoA 数组，destroy 时由 pool 一并清零）。"""
        if not self.projectiles.active[proj_idx]:
            return
        # 级联关闭父实体=该弹丸的 emitter（即时性，双保险之一；sync 也会兜底）
        self._close_projectile_emitters(proj_idx)
        self.projectiles.destroy(proj_idx)

    def _close_projectile_emitters(self, proj_idx: int) -> None:
        """关闭父实体=弹丸的 emitter（主动导引头/毫米波扫描器随载体毁灭）。"""
        em = self.emitters
        mask = (
            em.active
            & (em.parent_kind == EMITTER_PARENT_PROJECTILE)
            & (em.parent_idx == proj_idx)
        )
        idxs = np.flatnonzero(mask)
        if idxs.size:
            em.active[idxs] = False
            for i in idxs:
                i = int(i)
                em._deactivated.discard(i)
                if i not in em._free_set:
                    em._free_stack.append(i)
                    em._free_set.add(i)

    def _close_armor_emitters(self, armor_idx: int) -> None:
        """关闭父实体=装甲的**主动** emitter（红外除外，走残热衰减，§2.2 陷阱 §八.16）。"""
        em = self.emitters
        mask = (
            em.active
            & (em.parent_kind == EMITTER_PARENT_ARMOR)
            & (em.parent_idx == armor_idx)
            & (em.band != int(EmitterBand.IR))
        )
        idxs = np.flatnonzero(mask)
        if idxs.size:
            em.active[idxs] = False
            for i in idxs:
                i = int(i)
                em._deactivated.discard(i)
                if i not in em._free_set:
                    em._free_stack.append(i)
                    em._free_set.add(i)

    def _recycle_stale(self) -> None:
        """回收超龄 / 出界 / 无目标弹丸（脱靶）。

        SCAN 阶段（末敏弹）：按 scan_duration_s 超时自毁（age 在 SCAN 切换时清零，
        复用超龄路径，零新数组——docs §5.2）。
        """
        p = self.projectiles
        stale: list[int] = []
        # 只遍历活跃弹丸（flatnonzero 向量化；全容量 16384 的 Python 循环是每帧开销）
        for i in p.active_indices():
            i = int(i)
            if p.guid_stage[i] == int(GuidanceStage.SCAN):
                ammo = self._ammo_of(i)
                max_scan = float(ammo.scan_duration_s) if ammo is not None else 5.0
                if p.age[i] > max_scan:
                    stale.append(i)
                else:
                    p.age[i] += TICK_BULLET
                continue
            if p.age[i] > PROJECTILE_MAX_AGE:
                stale.append(i)
                continue
            p.age[i] += TICK_BULLET
            px, py, _ = p.pos[i]
            if abs(px) > WORLD_BOUND or abs(py) > WORLD_BOUND:
                stale.append(i)
        for i in stale:
            owner = int(p.owner[i])
            self.events.emit("miss", {"proj": i, "shooter": owner})
            self._destroy_projectile(i)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def pending_projectiles(self) -> int:
        """当前飞行中弹丸数。"""
        return int(np.sum(self.projectiles.active))

    def cooldown_remaining(self, armor_idx: int) -> float:
        return float(self._cooldown.get(armor_idx, 0.0))

    def active_armor_count(self) -> int:
        return int(np.sum(self.armor.active))
