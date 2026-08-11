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
from Airis.lulilo.combat.weapons import WeaponRegistry, WeaponSystem
from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import ArmorPool, BoxPool, PartType, ProjectilePool
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.movement.integrate import integrate_projectile_movement
from Airis.lulilo.simulation.loop import EventBus, TICK_BULLET

logger = get_logger(__name__)

# 弹丸最大飞行寿命（秒），超出回收（防止永不命中无限存在）
PROJECTILE_MAX_AGE = 20.0
# 弹丸回收边界（米），超出即视为脱靶/飞出战场
WORLD_BOUND = 5000.0
# 发射初始抬高（炮管高度，m）
MUZZLE_HEIGHT = 1.5

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
    ) -> FireRecord | None:
        """发射一发（统计层解算 → 高斯散布 → 抛射物生成 → 弹药扣减）。

        Args:
            shooter_idx: 射手装甲索引。
            target_idx: 目标装甲索引。
            ammo_id: 覆盖单位配置的弹药；None 用配置默认。
            mods: 透传给 WeaponSystem.compute_sigma_m 的修正因子。

        Returns:
            FireRecord；若冷却中 / 无炮塔 / 无弹药 / 超射程返回 None。
        """
        if not self.can_fire(shooter_idx):
            return None
        if not self.armor.active[target_idx]:
            return None

        cfg = self._config_of(shooter_idx)
        shooter_pos = self.armor.pos[shooter_idx]
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

        # 开火解算（统计层：sigma + 初速）
        sol = self.weapon_system.solve(
            cfg.weapon_id, ammo_id or cfg.ammo_id, range_m,
            crew_skill=cfg.crew_skill,
            visibility=cfg.visibility,
            weapon_condition=cfg.weapon_condition,
            **(mods or {}),
        )

        # 瞄准点：目标车体中心（水平面瞄准，z 抬到目标半高）
        aim_x = float(target_pos[0])
        aim_y = float(target_pos[1])
        aim_z = float(target_pos[2] + self.armor.half[target_idx, 2])

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
        """弹药 → 抛射物类型（APFSDS/AP 为子弹，HEAT/ATGM 为导弹）。"""
        from Airis.lulilo.entities.pools import ProjectileType
        from Airis.lulilo.combat.weapons import AmmoType

        ammo = self.weapon_system.registry.get_ammo(ammo_id)
        if ammo.ammo_type in (AmmoType.HEAT, AmmoType.ATGM, AmmoType.HESH):
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

    def _step_sub(self, sub_dt: float) -> None:
        """一个子弹子步：移动弹丸 → 碰撞 → 毁伤 → 回收。"""
        self.boxes.sync_all_from_parent(self.armor)

        # 1. 弹道积分（含重力）
        integrate_projectile_movement(
            self.projectiles.pos,
            self.projectiles.vel,
            self.projectiles.active,
            sub_dt,
            self.gravity,
        )

        # 2. 碰撞检测（空间哈希 broad + Numba narrow）
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

        # 3. 逐命中毁伤解析
        destroyed_proj: set[int] = set()
        for hit in hits:
            if hit.proj_idx in destroyed_proj:
                continue  # 同发弹丸只解析一次
            destroyed_proj.add(hit.proj_idx)
            self._resolve_hit(hit, sub_dt)

        # 4. 销毁已命中弹丸
        for proj_idx in destroyed_proj:
            self._destroy_projectile(proj_idx)

        # 5. 回收超龄 / 出界弹丸（脱靶）
        self._recycle_stale()

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
        self.projectiles.destroy(proj_idx)

    def _recycle_stale(self) -> None:
        """回收超龄 / 出界 / 无目标弹丸（脱靶）。"""
        p = self.projectiles
        stale: list[int] = []
        for i in range(p.capacity):
            if not p.active[i]:
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
