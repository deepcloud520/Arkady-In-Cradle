import math
from pathlib import Path
from typing import Any
import numpy as np

import sys
#sys.path.append(str(Path(__file__).parent.parent.parent))



from Airia.lulilo.combat.collision import CollisionDetector
from Airia.lulilo.combat.damage import ArmorType
from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
from Airia.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.coordinates.grid import world_to_grid
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.entities.pools import AirFormationPool, AircraftPool, AircraftRole, ArmorPool, BoxPool, EmitterPool, FormationAIPool, PartType, ProjectilePool
from Airia.lulilo.entities.side import Side, SideModelMapper, allegiance
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.logistics.fuel import FuelManager
from Airia.lulilo.movement.aerodynamics import AircraftAero
from Airia.lulilo.movement.flight_control import FlightControl, build_air_movement_hook
from Airia.lulilo.movement.integrate import build_armor_movement_hook
from Airia.lulilo.movement.pathfinding import GraphPathfinder
from Airia.lulilo.simulation.ai import ArtilleryAI, FormationAI, FormationIntent, MuzzleFlashLog, UnitRole
from Airia.lulilo.simulation.air_ai import AirFormationAI, AirUnitAI
from Airia.lulilo.simulation.command_router import GroundCommandRouter
from Airia.lulilo.simulation.state import SimulationState
from Airia.lulilo.terrain.dual_contour import DualContourBuilder


from Airia.lulilo.combat.weapons import AmmoConfig, AmmoType, GuidanceType, ProjectileType, SeekerType, WeaponConfig, WeaponRegistry, WeaponSystem
from Airia.lulilo.simulation.loop import SimulationLoop
from Airia import render, narrative, ui, lulilo

logger = get_logger(__name__)

def _air_registry() -> WeaponRegistry:
    """内置 + 航空弹药（AIM-120 雷达 AAM / 1000kg 滑翔炸弹）。

    与 tests/test_phase7_aam.py::_air_registry 等价但独立实现（UI 层不 import tests）。
    空战演示（Phase 7）需要：`fire_air` 发射 AAM 必须有制导弹药定义。
    """
    reg = WeaponRegistry.with_builtin()
    reg.register_definition(
        WeaponConfig(
            weapon_id="aim120_amraam",
            name="AIM-120 AMRAAM",
            caliber_mm=178.0,
            muzzle_velocity=900.0,
            base_accuracy_mrad=0.5,
            fire_rate_rpm=1.0,
            max_range_m=30000.0,
            fire_control_stabilization=0.9,
            compatible_ammo=["aim120_amraam"],
        ),
        [
            AmmoConfig(
                ammo_id="aim120_amraam",
                name="AIM-120 AMRAAM",
                ammo_type=AmmoType.ATGM,
                pen_mm_rha=90.0,            # 破片战斗部等效
                muzzle_velocity=900.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="dual_pulse_pn",
                seeker=SeekerType.ACTIVE_RADAR,
                emitter_power_db=70.0,
                nav_gain=4.0,
                search_fov_deg=60.0,
                track_fov_deg=10.0,
                filter_quality=0.85,
                prox_fuse_radius=25.0,      # 近炸（AIM-120 破片散布 ~25m）
                blast_radius=15.0,
                blast_dmg=90.0,
            ),
        ],
    )
    reg.register_definition(
        WeaponConfig(
            weapon_id="gb_1000kg",
            name="1000kg 滑翔炸弹",
            caliber_mm=457.0,
            muzzle_velocity=120.0,
            base_accuracy_mrad=2.0,
            fire_rate_rpm=1.0,
            max_range_m=20000.0,
            fire_control_stabilization=0.5,
            compatible_ammo=["gb_1000kg"],
        ),
        [
            AmmoConfig(
                ammo_id="gb_1000kg",
                name="1000kg 滑翔炸弹",
                ammo_type=AmmoType.HE_FRAG,
                pen_mm_rha=400.0,
                muzzle_velocity=120.0,
                projectile_type=ProjectileType.MISSILE,
                guidance=GuidanceType.FIRE_AND_FORGET,
                control_law_id="glide_bomb",
                seeker=SeekerType.SEMI_ACTIVE_LASER,
                nav_gain=2.0,
                search_fov_deg=40.0,
                track_fov_deg=8.0,
                filter_quality=0.8,
                blast_radius=30.0,
                blast_dmg=300.0,
                prox_fuse_radius=0.0,
            ),
        ],
    )
    return reg
def create_simulation(gl_widget: Any) -> tuple[SimulationLoop, dict[int, str], dict[int, str]]:
    """创建模拟层 + 演示编队（坦克×2 + 炮兵×2 + 空军×2）+ 后勤 + 全链路。

    注意：ArmorPool.side 为绝对阵营 SoA 字段（entities/side.py Side 枚举）；
    演示默认玩家=北约(0)，敌友中立由 allegiance() 派生。

    🔥 2026-08-13 接入移动+悬挂（docs/suspension-slope-design.md §2）：
    - 逻辑 WorldMesh = DualContourBuilder（连续场平滑 → 地面双线性斜面，
        悬挂 pitch/roll 在 fBm 起伏上生效；build_adjacency=False 省扫描线）
    - 编队设路标点（NavGraph 寻路，失败降级直线）→ movement hook 每帧
        推进：integrate_3d → resolve_mesh → solve_suspension → resolve_contact

    2026-08-17 演示升级（配合 320×320 大地图，世界 640×640m）：
    - 地面：北约/华约坦克编队相向推进到峡谷长桥两端接敌；
    - 炮兵：双方 4 门 M109 后置炮位，初始 STRIKE 互轰 → 炮口闪光 →
        反炮闭环（ArtilleryAI×2 共享 MuzzleFlashLog）→ 疏散转移；
    - 空军：双方双机编队迎头空战（自由开火令 → AAM → 坠毁链）；
    - 渲染：WorldAdapter 加 aircraft 视图，HUD 阵营色小圆点标记飞机。
    """
    state = SimulationState()
    armor = ArmorPool(capacity=96)
    box = BoxPool(capacity=384)
    proj = ProjectilePool(capacity=256)
    ammo = AmmoManager(capacity=96)
    fuel = FuelManager(capacity=96)
    emitters = EmitterPool(capacity=128)   # 电磁辐射源（导引头/诱饵）
    damage_pool = DamageSubmissionPool(capacity=256)  # 爆炸/区域毁伤
    formations = FormationAIPool(capacity=12)          # 地面编队 AI 池
    aircraft = AircraftPool(capacity=16)               # 空中单位池（Phase 7）
    air_formations = AirFormationPool(capacity=8)      # 空中编队 AI 池
    state.armor_pool = armor
    state.box_pool = box
    state.projectile_pool = proj
    state.emitter_pool = emitters
    state.damage_pool = damage_pool
    state.formation_pool = formations
    state.aircraft_pool = aircraft
    state.air_formation_pool = air_formations
    state.ammo = ammo
    state.fuel = fuel

    # 主循环（先建：FiringControl / FormationAI 共享 loop.events 事件流）
    loop = SimulationLoop(state=state)
    # 每车速度（m/s，沿路标点；履带毁减速由 FormationAI 钳制）
    speed = np.zeros(armor.capacity, dtype=np.float32)
    # ---- 交战链（Phase 3 开火链 + 6c 编队 AI + 7 空战：三军共用）----
    # 航空弹药注册表 = 内置地面武器 + AIM-120/滑翔弹（fire_air 必需）
    # detector：空间哈希网格覆盖 15km 地图（默认 world_half=2500 只覆盖
    # [-2500,2500]²，战场 7500 处装甲登记越界 → 弹丸 0 命中；按地图放大）
    fc = FiringControl(
        armor, box, proj,
        weapon_system=WeaponSystem(_air_registry()),
        control=MissileControlRegistry.with_builtin(),
        emitters=emitters,
        rng=RNGManager(seed=7),
        events=loop.events,                    # 命中/毁伤事件 → 日志/弹道线
        ammo=ammo,
        damage_pool=damage_pool,
        aircraft=aircraft,                     # 空战（fire_air/飞机毁伤）
        detector=CollisionDetector(world_half=8000.0),
    )
    ai = FormationAI(armor, formations, fc=fc, events=loop.events, speed=speed)

    terrain = gl_widget.terrain
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
        count: int = 16, target: tuple[float, float] = (128.0, 128.0),
        role: int = int(UnitRole.TANK), speed_v: float = 5.0,
    ) -> tuple[int, int]:
        """生成一个地面编队（坦克/炮兵），返回 (首车索引, 编队索引)。

        每车登记：编队成员、主炮配置、装甲值、碰撞箱、后勤基数、
        路标点（寻路 → 悬挂生效；炮兵静止无路标点）。

        演示兵力设计（让「部位毁伤」可见，而非 APFSDS 互秒）：
            - 北约 M1A2（M256 120mm APFSDS 650mm 穿深，复合装甲 250mm）：
                APFSDS 打华约 T-14 需 2~3 发（正面 350 反应装甲），
                命中 TRACK/ENGINE 一发毁薄部位。
            - 华约 T-14 机炮车（2A90 57mm 自动炮 200mm 穿深，反应装甲 350mm）：
                机炮打不穿北约正面（复合 375 等效）→ 震荡磨 HULL；
                打北约 TRACK/ENGINE（~190 等效）→ 穿透 → 部位逐步毁伤
                （履带断 → 失机动、炮塔毁 → 停火），高射速持续输出。
            - 炮兵（role=ARTILLERY）：M109 155mm 曲射炮位，开火由
                ArtilleryAI 驱动（FormationAI 跳过炮兵成员直射）。
        """
        fidx = formations.spawn(side=side, role=role)
        formations.comm[fidx] = 1.0
        formations.target_zone[fidx] = (target[0], target[1], 0.0)
        nato = int(side) == int(Side.NATO)
        is_arty = int(role) == int(UnitRole.ARTILLERY)
        first = -1
        for i in range(count):
            wx = x0 + i * 6
            wy = y0 + i * 6
            aidx = armor.spawn(
                x=wx, y=wy, z=_ground_z(wx, wy),
                half_x=1.5, half_y=3.0, half_z=0.8,
                yaw=yaw, role=role,
            )
            armor.side[aidx] = side
            armor.formation_id[aidx] = fidx
            # 主炮 + 装甲配置（Phase 3 开火链 + 6c 编队 AI 消费）
            if is_arty:
                fc.set_unit_config(aidx, UnitCombatConfig(
                    weapon_id="m109_155mm",
                    ammo_id="m107_155mm_he_frag",
                    armor_mm=100.0, armor_type=int(ArmorType.RHA),
                    crew_skill=0.7, visibility=1.0, weapon_condition=1.0,
                    max_range_m=15000.0, fire_rate_rpm=4.0,
                ))
            elif nato:
                fc.set_unit_config(aidx, UnitCombatConfig(
                    weapon_id="m256_120mm",
                    ammo_id="m829a3_apfsds",
                    armor_mm=2500.0, armor_type=int(ArmorType.COMPOSITE),
                    crew_skill=0.9, visibility=1.0, weapon_condition=1.0,
                    max_range_m=4000.0, fire_rate_rpm=10.0,
                ))
            else:
                fc.set_unit_config(aidx, UnitCombatConfig(
                    weapon_id="2a90_57mm",
                    ammo_id="57mm_apfsds",
                    armor_mm=350.0, armor_type=int(ArmorType.REACTIVE),
                    crew_skill=0.8, visibility=1.0, weapon_condition=1.0,
                    max_range_m=6000.0, fire_rate_rpm=120.0,
                ))
            # 后勤演示数据（Phase 5 DetailPanel 级别 0/2 显示）
            ammo.state.set_unit(aidx, cap=40.0, initial=40.0 * (0.8 + 0.2 * i / count))
            fuel.state.set_unit(aidx, cap=100.0, initial=100.0 * (0.85 + 0.15 * i / count))
            # 路标点（寻路 → 悬挂在 fBm 起伏/桥上生效；炮兵静止待命）
            if not is_arty:
                armor.set_waypoints(aidx, _route((wx, wy), target))
                speed[aidx] = speed_v
            if first < 0:
                first = aidx
            # 碰撞箱（Phase 3 部位毁伤演示：多层判定，见 docs 演示注释）。
            # 每部位**自身装甲厚度**（BoxPool.armor_mm；0=兜底整车值）+
            # 未击穿震荡系数（BoxPool.shock_factor；T-14 无人炮塔 = 0 →
            # 机炮打不穿炮塔也不会把震荡传导进乘员舱扣 HULL 血）
            if is_arty:
                part_armor = {
                    int(PartType.HULL): 80.0,      # 自行火炮轻甲
                    int(PartType.TURRET): 60.0,
                }
            elif nato:
                part_armor = {
                    int(PartType.HULL): 350.0,     # 车体正面（复合）
                    int(PartType.TURRET): 400.0,   # 炮塔重甲
                    int(PartType.ENGINE): 150.0,   # 后部引擎舱
                    int(PartType.TRACK): 120.0,    # 侧裙/履带薄甲
                }
            else:
                part_armor = {
                    int(PartType.HULL): 350.0,     # 乘员舱（有人 → 震荡全传导）
                    int(PartType.TURRET): 300.0,   # 无人炮塔（装甲厚但无人）
                    int(PartType.ENGINE): 180.0,
                    int(PartType.TRACK): 130.0,
                }
            hull_b = box.spawn(aidx, int(PartType.HULL), 1.5, 3.0, 0.8,
                                local_x=0.0, local_y=0.0, local_z=0.0,
                                armor_mm=part_armor[int(PartType.HULL)],
                                shock_factor=1.0)
            armor.bind_hull(aidx, hull_b)  # 绑定 HULL box（part-health-design.md §2.2）
            box.spawn(aidx, int(PartType.TURRET), 1.2, 1.2, 0.8,
                        local_x=0.0, local_y=0.0, local_z=1.8,   # 抬高露出车体
                        armor_mm=part_armor[int(PartType.TURRET)],
                        shock_factor=(0.2 if not nato else 0.7) if not is_arty else 1.0)
            if not is_arty:
                box.spawn(aidx, int(PartType.ENGINE), 1.0, 1.2, 0.8,
                            local_x=0.0, local_y=-1.0, local_z=1.0,  # HULL 内部（多次判定）
                            armor_mm=part_armor[int(PartType.ENGINE)])
                box.spawn(aidx, int(PartType.TRACK), 0.4, 3.4, 0.3,
                            local_x=0.0, local_y=0.0, local_z=0.0,    # 两侧外伸
                            armor_mm=part_armor[int(PartType.TRACK)])
        formations.member_count[fidx] = count
        # 意图：坦克进攻（接敌即分配目标开火）；炮兵待命（曲射由 ArtilleryAI 驱动）
        formations.intent[fidx] = int(
            FormationIntent.ADVANCE if not is_arty else FormationIntent.HOLD
        )
        return first, fidx

    # 兵力布局（世界坐标；15km 扁平地图，战场收拢在地图中央 (7500,7500) ±400m）
    # 说明：直射武器无重力补偿（fire aim_z=目标中心），坦克间距须 <~500m
    # 弹道下坠才在装甲 Z 范围内（57mm 700m 下坠 ~3m 会穿透地面 → miss）。
    # 地面：双方坦克编队相向推进到中央接敌（间距 155~410m）
    nato_first, nato_tank_fidx = _squad(
        9350, 9350, int(Side.NATO), 0.0, count=16, target=(7500.0, 7500.0)
    )
    enemy_first, _ = _squad(7550, 7550, int(Side.PACT), math.pi, count=16,
                            target=(7500.0, 7500.0))
    # 炮兵：双方 4 门后置炮位（接敌区两侧，曲射对轰 + 反炮闭环）
    _squad(7150, 5900, int(Side.NATO), math.pi * 0.5, count=4,
            role=int(UnitRole.ARTILLERY))
    _squad(7850, 7100, int(Side.PACT), -math.pi * 0.5, count=4,
            role=int(UnitRole.ARTILLERY))
    box.sync_all_from_parent(armor)

    # 演示损坏部位标记：敌军首车炮塔毁伤（HUD ⚠ 标记 Phase 5 即可生效）
    for bi in range(box.capacity):
        if box.parent[bi] == enemy_first and box.part[bi] == int(PartType.TURRET):
            box.active[bi] = False
            break
    # 部位毁伤 = 单位状态（渲染层读 armor.parts_destroyed，与 box 槽位解耦）
    armor.parts_destroyed[enemy_first, int(PartType.TURRET)] = True

    # ---- 炮兵编队 AI（Phase 6c 步骤⑤⑥）：双方各一实例 + 共享炮口闪光日志
    # 反炮闭环（docs §六.3「开火→暴露→反炮→转移」）：A 开炮 → B 检测闪光 →
    # 反炮 STRIKE → A 检测来袭 → 疏散转移；闪光时钟由 _ai_hook 单点驱动。
    flashes = MuzzleFlashLog()
    art_nato = ArtilleryAI(
        armor, formations, fc=fc, events=loop.events,
        side=int(Side.NATO), muzzle_flashes=flashes,
    )
    art_pact = ArtilleryAI(
        armor, formations, fc=fc, events=loop.events,
        side=int(Side.PACT), muzzle_flashes=flashes,
    )
    # 初始演示火力任务：双方炮兵互轰对方坦克编队区域（此后反炮闭环自动接管）
    art_nato.request_strike((7600.0, 7600.0, 0.0), rounds=3)
    art_pact.request_strike((7400.0, 7400.0, 0.0), rounds=3)

    def _ai_hook(_loop: SimulationLoop, dt: float) -> None:
        """ai（1s）组合钩：坦克编队 AI + 炮兵 AI×2（共享闪光时钟）。"""
        ai.tick(_loop, dt)          # 坦克编队 AI（含支援请求链）
        flashes.tick(dt)            # 炮口闪光共享时钟（单点驱动，防双重老化）
        art_nato.tick(_loop, dt)    # 北约炮兵（接单/反炮/疏散）
        art_pact.tick(_loop, dt)    # 华约炮兵

    def _strike_handler(cmd: Any) -> bool:
        """玩家 STRIKE → 本阵营炮兵 AI（cmd.params 可带 side，默认北约）。"""
        side = int(cmd.params.get("side", int(Side.NATO)))
        art = art_nato if side == int(Side.NATO) else art_pact
        return art.handle_command(cmd)

    # 注：完整 command_handler 在 unit_names 建好后再装配（见函数末），
    # 因为地面指令路由需要单位名表来解析叙事层 target_id。

    # ---- 空军（Phase 7）：双方双机编队迎头空战（自由开火令 → AAM）----
    aero = AircraftAero(aircraft.capacity)
    flight = FlightControl(aircraft.capacity, aero)  # 供单位 AI（movement hook 内部另建）
    air_fai = AirFormationAI(
        aircraft, air_formations, aero, armor=armor,
        rng=RNGManager(seed=8), flight=flight, emitters=emitters,
    )
    air_uai = AirUnitAI(aircraft, aero, flight, fc, armor=armor)

    def _air_squadron(
        x: float, y: float, side: int, vel_x: float, n: int = 2,
    ) -> int:
        """生成一个空中编队（巡航高度 3000m），返回编队索引。"""
        fidx = air_formations.spawn(side=side, target_zone=(7500.0, 7500.0, 3000.0))
        members: list[int] = []
        for k in range(n):
            m = aircraft.spawn(
                x, y + k * 80.0 - 40.0, 3000.0, side=side,
                role=int(AircraftRole.FIGHTER), formation_id=fidx,
                cruise_alt=3000.0, cruise_spd=250.0,
            )
            aircraft.vel[m] = (vel_x, 0.0, 0.0)
            aircraft.heading[m] = 0.0 if vel_x > 0 else math.pi
            aircraft.aam_left[m] = 4
            members.append(m)
        air_formations.member_count[fidx] = n
        air_formations.leader_idx[fidx] = members[0]
        return fidx

    blue_f = _air_squadron(7000.0, 7500.0, int(Side.NATO), 250.0)
    red_f = _air_squadron(8000.0, 7500.0, int(Side.PACT), -250.0)
    air_fai.weapons_free[blue_f] = True   # 自由开火令（CAP 接敌即可交战）
    air_fai.weapons_free[red_f] = True

    def _air_ai_hook(_loop: SimulationLoop, dt: float) -> None:
        air_fai.tick(dt)   # 编队决策（FSM + 目标分配）
        air_uai.tick(dt)   # 单位执行（机动 + fire_air）

    # 单位名称 + 阵营映射（型号统一走 entities.side SideModelMapper，
    # 与 GUI HUD/DetailPanel 同一数据源；叙事层 recorder 消费 unit_names）
    side_models = SideModelMapper()
    unit_names: dict[int, str] = {}
    side_map: dict[int, str] = {}
    for idx in range(armor.capacity):
        if armor.active[idx]:
            side_int = int(armor.side[idx])
            role_int = int(armor.role[idx])  # 炮兵（ARTILLERY）→ M109/2S19，非 MBT
            model = side_models(side_int, role_int)
            unit_names[idx] = f"{model} #{idx:02d}"
            # 相对敌友派生：演示场景玩家=北约，中立 → "neutral"
            side_map[idx] = allegiance(int(Side.NATO), -1, side_int, -1)

    # ---- 玩家指令路由（2026-09-11：让叙事层选择真正改变战局）----
    # 背景：loop._apply_command 对未被 handler 消费的指令只 emit 一个 command 事件，
    # 而 ArtilleryAI 只认 STRIKE → 此前 HOLD/REDEPLOY 零实际效果
    # （docs/story-architecture-review.md §三.5）。
    # GroundCommandRouter 消费 HOLD/REDEPLOY/RESUPPLY/DISENGAGE，STRIKE 仍走炮兵。
    router = GroundCommandRouter(
        armor, formations, speed,
        pathfinder=pathfinder, ammo=ammo, fuel=fuel,
        events=loop.events, player_side=int(Side.NATO),
    )
    router.set_unit_names(unit_names)
    # 叙事层别名：story/dialogues/*.yaml 的 target_id 用叙事名（如 "7th_armor"），
    # 装配层把叙事名绑到真实编队索引 —— 引擎不认识剧情专名。
    router.register_alias("7th_armor", nato_tank_fidx)
    router.register_alias("7th_armor_company", nato_tank_fidx)

    def _command_handler(cmd: Any) -> bool:
        """玩家指令总入口：地面编队指令优先，STRIKE 回落炮兵 AI。"""
        if router.handle_command(cmd):
            return True
        return _strike_handler(cmd)

    loop.command_handler = _command_handler

    # 移动+悬挂管线（每帧推进；弹簧 dt=1/60 稳定域内）
    loop.attach("movement", build_armor_movement_hook(armor, world_mesh, speed))
    # 空军管线（Phase 7）：飞控 → 气动积分 → 失速坠毁 → 箱/辐射源同步
    loop.attach("air_movement", build_air_movement_hook(
        aircraft, aero, box, emitters, armor, proj))
    loop.attach("air_ai", _air_ai_hook)
    # 交战链（三军共用）：
    #   - combat：子弹 Tick（0.01s）——弹道/碰撞/毁伤（Phase 3）
    #   - ai：地面 Tick（1s）——坦克编队 + 炮兵×2（Phase 6c）
    #   - air_movement：空中 Tick（0.1s）——气动积分（Phase 7）
    #   - air_ai：地面 Tick（1s）——空中编队/单位 AI（Phase 7）
    loop.attach("combat", lambda _loop, dt: fc.tick(dt))
    loop.attach("ai", _ai_hook)
    loop.set_rate("combat", 0.01)
    loop.set_rate("ai", 1.0)
    loop.set_rate("air_movement", 0.1)
    loop.set_rate("air_ai", 1.0)
    return loop, unit_names, side_map