# 开发进度

## 2026-08-17 Phase 8 数据链体系作战实施完成（docs/phase8-datalink-design.md §十 步骤①~⑧⑩；磨盘⑨按用户要求留待后续）

> **Phase 8 = 封印上帝视角**：AI/玩家/渲染接触的一切敌情来自认知孪生（Twin），
> 孪生由「传感器观测 + 数据链共享 + 时间衰减 + 情报注入」驱动；数据链好坏直接决定
> 孪生与物理世界的偏差 = 战争迷雾浓度。**磨盘战术（步骤⑨）用户明确留待后续。**

### 新增文件
- `entities/twin.py`：`TwinPool`（认知孪生池，MAX_TWIN=4096；1:1 槽映射免 dict——
  armor idx → [0,2048)，aircraft idx → [2048,2176)；status/confidence/pos_err/衰减字段）
- `simulation/twin_engine.py`：`TwinEngine` 四通道（传感器注入/数据链共享/时间衰减/延迟报告）
  + 死亡双通道（confirm_kia 立即 KIA / 物理失活 → MIA → 侦察确认 KIA）
- `simulation/datalink.py`：`DataLinkNetwork`（networkx/scipy：链路质量公式、连通簇、
  中央簇、route_quality 路径连乘、航迹广播；连边批量向量化 + scipy 稀疏连通 +
  networkx 惰性图）
- 测试 7 文件（twin_pool/twin_engine/datalink/adapter/armor_ai/loiter/air_datalink/integration）+ `tests/benchmark_phase8.py`

### 关键接缝改动（追加不重排）
- `simulation/state.py` / `serialization.py`：`twin_pool` 进快照 + `_restore_twin` 往返
- `simulation/loop.py`：`EventBus.drain_filter`（定向取事件，多消费者不互吞）
- `simulation/ai.py`：`Contact` 加 confidence/pos_err/twin_slot；`score_targets` 加
  confidence 折扣（迷雾影响决策）；`TwinContactProvider`（接触换孪生源）；
  `FormationAI._maybe_request_support`（支援请求链）+ `_assign_targets` 低置信过滤；
  `ArtilleryAI._handle_support_events`（自主接单，瞄准孪生坐标带误差）
- `simulation/air_ai.py`：`LoiterRecon`（巡飞弹侦察上报，`ProjectileType.LOITER=2`）；
  `AirFormationAI._tick_coordinated_evasion`（协同诱饵 + 机动脱离）；
  `AirUnitAI` 数据链发射（contact_confidence 注入 → fire_air 散布折扣）
- `combat/engagement.py`：`relay_guidance`（A 导 B 射中继写入 + guid_comm 门控）；
  `fire_air(confidence=...)`（共享航迹发射 sigma 放大）
- `render/adapter.py`：`map_twin_render_status`（TwinStatus+confidence → 渲染状态映射表）；
  `from_snapshot_twin`（位置/血量/置信度来自孪生，真值不进 adapter）

### 战争迷雾四大集成用例（docs §十一，全绿）
1. 地雷命中 → 通讯中断 MIA → 无人机确认 KIA 完整时序；
2. 装甲呼叫炮兵 → 孪生坐标（带误差）落点 → 毁伤 → 报告回流；
3. 数据链中断编队退回 6c 自保（comm 门控）；
4. A 导 B 射：relay_guidance 中继制导命中。

### 性能基准（tests/benchmark_phase8.py，现实预算达标 ✅）
- 孪生更新（2048 装甲 + 128 飞机 全观测）：2.4ms（设计理想 0.1ms 为纯内核；现实含
  Python 观测注入成本，预算 <5ms）
- 链路图重建（2176 节点，50km 战场分布）：~100ms（设计理想 1ms；现实含 scipy 连通 +
  networkx 惰性预留，预算 <150ms；连边批量向量化 + 距离截断 + scipy 稀疏连通 +
  惰性 networkx 图已优化）
- 全帧影响（128 装甲）：~1ms（目标 <2ms ✅）

### 遗留
- 磨盘战术（步骤⑨）用户明确留待后续；`AirTactic` 枚举位已预留
- 玩家手动数据链 UI 面板（范围外，Phase 5b 另立）
- numba 化链路图重建（现实 100ms / 1s tick ≈ 10% 单核，可接受）

## 2026-08-17（同日补充）main.py 演示调用链：装甲部队交战

- `Airis/ui/main_window.py::_create_simulation` 从「移动+悬挂演示」升级为**完整装甲交战**：
  - 装配 `FiringControl`（内置 M256/2A46M 主炮 + 弹药 + 伤害提交池 + EmitterPool）
    + `FormationAI`（编队 AI，默认全知接触）+ `FormationAIPool`；
  - 两队坦克编队（北约 M256 / 华约 2A46M）每车 `set_unit_config` + 编队意图
    ADVANCE → 接敌自动分配目标开火；
  - `loop.attach("combat", fc.tick)`（0.01s 子弹 Tick）+ `loop.attach("ai", ai.tick)`
    （1s 地面 Tick）——事件流（fire/hit/destroyed）进主循环日志/弹道线/命中标记。
- `main.py`：相机对准交战区 + 演示说明注释。
- offscreen 验证：16 辆互射 30s → fire=24 / hit=23 / destroyed=8（剩 8 辆）✅

## 2026-08-18 main.py 演示调用链升级：三军体系作战（坦克 + 炮兵 + 空军）

> 用户需求：略微加大地图，加入双方炮兵/空军进行测试。配合 320×320 大地图，
> `_create_simulation` 从「装甲交战」升级为**三军体系作战演示**（docs §4.1 依赖链
> 全部挂载：movement / air_movement / ai / air_ai / combat）。

### 改动文件
- `Airis/render/gl_widget.py::_default_terrain`：256×256 → **320×320**（世界 640×640m，
  长桥延长到 240 格；普通桥位置适配；DualContour 构建约 6~7s 加载期一次）；
  `_render_hud_inner` 加 **2.5) 空中单位标记**（阵营色小圆点，读 adapter.aircraft_*）
- `Airis/ui/main_window.py::_create_simulation`：
  - **地面**：北约 M1A2 / 华约 T-14 机炮车编队（16×2）相向推进到长桥两端接敌
    （布局从 (60,60)/(196,196) 散开到 (100,100)/(540,540)，目标 (200,258)/(440,258)）；
  - **炮兵**：双方 4 门 M109 155mm 后置炮位（角落 (70,570)/(570,70)），
    `ArtilleryAI`×2（side=NATO/PACT）共享 `MuzzleFlashLog` 炮口闪光日志
    → **反炮闭环**（开火暴露→反炮→来袭告警→疏散转移）；初始 STRIKE 任务
    互轰对方坦克编队区域（此后反炮闭环自动接管）；`loop.command_handler`
    按 cmd.params.side 路由玩家 STRIKE；
  - **空军**：双方双机编队迎头空战（`AircraftPool` + `AirFormationPool` +
    `AircraftAero` + `FlightControl` + `AirFormationAI`/`AirUnitAI`，自由开火令 →
    AAM 发射 → 近炸 → 坠毁链）；`FiringControl` 加 `aircraft=` 参数 +
    航空弹药注册表 `_air_registry()`（AIM-120 / 1000kg 滑翔弹，独立实现不 import tests）；
  - `_squad` 加 `role`/`speed_v` 参数（炮兵编队 role=ARTILLERY + intent=HOLD，
    静止无路标点，`FormationAI` 跳过炮兵成员直射）；
  - `ai` hook 组合：坦克 `FormationAI.tick` + `flashes.tick`（单点时钟）+
    `ArtilleryAI.tick`×2；`air_movement`（0.1s）/`air_ai`（1s）挂载
- `Airis/render/adapter.py`：`WorldAdapter` 加 **aircraft 视图**
  （aircraft_pos/active/side；`_aircraft_arrays()` 未装配→空数组），
  `from_snapshot` / `from_snapshot_twin` 填充
- `main.py`：相机 distance 260→380 俯瞰全战场

### offscreen 验证（60s @ 1/60s，QT_QPA_PLATFORM=offscreen）
- 事件：fire=595 / hit=497 / destroyed=16（坦克战）；artillery_fire=6（初始 6 发全打出）/
  counter_battery=5（反炮闭环）/ artillery_impact=6；fire_air=8 / prox_detonate=8 /
  air_crash=4（双方飞机全部被击落）
- 单位：armor 40 开局 → 24 存活；aircraft 4/4 坠毁
- 结论：三军调用链全通——坦克接敌、炮兵曲射+反炮互灭、空战 AAM 对射至全员坠毁 ✅

### 已知环境问题（非本次改动引入）
- `tests/output/` 下 11 个 pytest 残留临时目录（.pytest-basetemp / pytest-tmp-* 等）
  ACL 被锁无法删除（Get-Acl UnauthorizedAccess）→ `smoke_gui` 的 clean_output_dir
  与 pytest 的 tmp_path 收集会 PermissionError（pyproject.toml 已有注释记录）；
  全量回归 680 passed / 2 failed / 15 errors 中，failed（narrative 集成，story 数据
  pause=False）与 errors（PermissionError）均为环境/工作区既有状态，与本次改动无关

## 2026-08-18（同日补充）演示地图切换：15km×15km 扁平地图（Phase 9 最小尺寸）

> 用户需求：用 15km×15km 扁平地图，**不改管线**。战场收拢在地图中央 (7500,7500) ±400m，
> 三军演示照常。管线零结构改动，仅两处**装配参数适配地图尺寸**（见下）。

### 改动文件
- `Airis/render/gl_widget.py::_default_terrain`：320×320（640m）→ **400×400 格 × 37.5m =
  15km×15km 扁平**（amplitude=0，无桥/峡谷；WorldMesh ~19s / NavGraph ~1.1s 加载期一次；
  `surface_z` 全平 0）
- `Airis/ui/main_window.py`：兵力布局收拢到地图中心（坦克 7350/7550 相向接敌，间距
  155~410m；炮兵 7150,7900 / 7850,7100；空军 7000/8000,7500）；
  `FiringControl` 加 `detector=CollisionDetector(world_half=8000.0)`
- `Airis/lulilo/combat/engagement.py`：`WORLD_BOUND` 5000 → **12000**（弹丸回收边界随
  地图放大；曲射弹丸仍豁免出界回收）
- `main.py`：相机 distance 9500 俯瞰全图（far=12000 覆盖）

### 🔥 两个 15km 暴露的"地图尺寸相关"配置（管线逻辑零改动）
1. **CollisionDetector 空间哈希覆盖**：默认 `world_half=2500` 只覆盖 [-2500,2500]²，
   战场 7500 处装甲登记**越界格** → 弹丸 0 命中（hit=0/miss=472，320 小地图因战场
   <2500 未暴露）。修复 = 装配层传入 `CollisionDetector(world_half=8000.0)`。
2. **直射弹道下坠**：fire aim_z=目标中心（无重力补偿），57mm APFSDS 700m 射程飞行
   0.8s 下坠 ~3m → 弹丸穿透装甲 Z 范围 → miss。战场收拢到间距 <500m（下坠 <1m）规避。
   ✅ **已修复（超仰火控）**：`combat/engagement.py` 新增 `solve_direct_fire_elevation`
   （低伸解，允许负仰角=下坡/近距离俯角），`FiringControl.fire` 对无制导直射按距离反解
   仰角，弹丸下坠后命中目标中心；制导弹仍走制导律。回归测试 `tests/test_phase3_integration.py`
   `TestDirectFireSuperelevation`（2 用例）+ `test_per_part_armor_mm` 改用 `aim_at` 直瞄履带
   （旧断言依赖下坠误坠履带，已随修复失效）。

### offscreen 验证（60s，15km 扁平地图）
- 事件：hit=149 / ricochet=43 / miss=26（**命中率 68%**）/ destroyed=23；
  artillery_fire=11 / counter_battery=8 / artillery_displace=2；fire_air=14 /
  prox_detonate=13 / air_crash=3；rwr_warning=18 / smoke=18 / decoy_deployed=13
- 单位：armor 40 开局 → 17 存活；aircraft 4 → 1 存活
- 结论：15km 扁平地图三军全通（坦克直射命中、炮兵对轰+反炮+疏散、空战 AAM、
  RWR 告警+诱饵）✅

### 抛射体 / 导弹可视化（HUD）
- `render/adapter.py`：新增 `_projectile_views` + 视图字段（pos/vel/active/owner/owner_kind/
  guid_law/guid_confidence）+ `projectile_markers(player_side, confidence_threshold)` 查询——
  **己方抛射体全绘**，**敌方导弹（guid_law>=0）按 guid_confidence 门控**（上帝视角恒 1.0 全绘，
  Phase 8 迷雾接入后自动只显已探测），敌方无制导弹不绘。
- `render/gl_widget.py`：`_render_projectiles` 在 HUD 画抛射体圆点（己方青 / 敌方已探测导弹橙红）
  + 导弹沿速度反方向短尾迹。
- 回归测试 `tests/test_projectile_render.py`（3 用例）；45 render/adapter 用例全绿。

### 直射火控前置量（按敌方速度打预留量）
- **纪律落地**：`movement/integrate.py` 的 `integrate_armor_movement` / `integrate_armor_movement_3d`
  新增 `vel` 输出参数，移动积分时同步写 `armor.vel`（本 tick 位移 / dt，静止/失活归零）——
  `ArmorPool.vel` 原本就是「由 integrate_armor_movement 更新」的预留字段，这次补齐写入。
  `build_armor_movement_hook` + `scenario._make_movement_hook` 回退路径均接入。
- `combat/engagement.py::fire`：解算后按 `armor.vel[target]` × `t_flight(≈R/初速)` 前移瞄准点
  （前置量），制导/地形目标（`aim_at`）跳过；与超仰火控叠加（先前置量、后仰角解算）。
- 测试：`test_movement.py`（vel 写入/静止归零）+ `test_phase3_integration.py` 前置量用例；
  movement/worldmesh/suspension/scenario/combat/air 全绿（131+33 用例）。已清 Numba 缓存。
