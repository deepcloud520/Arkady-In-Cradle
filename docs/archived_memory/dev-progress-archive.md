

---

## 归档批次（2026-08-14 之前的记录，由 archive_dev_progress.py 移出）

## 2026-08-13 参考项目 stochastic-warfare 源码研究（docs/phase6-guidance-pipeline.md §十）
- 结论：它的导弹是**解析式战役层模型**（FlightProfile 预计算 + 时间线性插值 + 末端 CEP 命中判定），**无逐帧制导、无推力剖面、无二次点火**——`propulsion` 只是字符串枚举（rocket/turbojet/ramjet）全程恒速 max_speed_mps
- 关键文件：combat/missiles.py（解析式飞行）/ ballistics.py（RK4 积分，optional_jit 可选 Numba）/ ammunition.py（AmmoDefinition 的 propulsion/guidance/seeker_fov_deg 字段）/ ew/emitters.py（EmitterRegistry = dict Python 对象 + get_active_emitters 按 type/freq/side 过滤）
- 可借鉴（已写入文档 §10.2）：GuidanceType 枚举扩展（INERTIAL/GPS/TERCOM/COMMAND）、seeker_fov_deg/seeker_range_m 印证 FOV 门限、propulsion→推进剖面命名、terminal_maneuver→TERMINAL 阶段、EmitterRegistry 查询接口
- 不借鉴：解析式飞行、枚举分派（正是脚本化要解决的痛点）、optional_jit
- 我们的「控制回调 + age 时序 + 通信/置信度」是它之上的更细粒度原创（2026-08-14 起弃数据链系统，见文件顶部）

## 2026-08-13 Phase 6b 设计更新：推力剖面 × 制导律合并为「导弹控制回调」（docs/phase6-guidance-pipeline.md §三）
- 用户拍板：中远程导弹「开机→滑翔→二次点火」的推力剖面也要写成回调，且与制导律**合并**——制导律（PN/PP）降级为**公共转向求解器**（模块级 njit，不注册、不进回调参数，被控制回调 import 直调）
- 理由：推力函数需要 age（时序），二次点火触发依赖导引头锁定（目标位置）+ 数据链（人在回路），与制导律输入高度重叠
- 统一签名：制导律 7 参数 + `age` + `datalink_cmd`，out 输出**总加速度 = 转向 + 轴向推力**
- 新增 `DatalinkCmd` 枚举（NONE/HOLD/REIGNITE/RETARGET/SELF_DESTRUCT）：HOLD/REIGNITE 由控制回调消费（影响推进），RETARGET/SELF_DESTRUCT 由 Python 层处理
- 新增 SoA `guid_datalink_cmd`（int32）；`age` 复用已有 ProjectilePool.age（_recycle_stale 累加，滞后一子步 0.01s 无影响）
- 注册表改名 `MissileControlRegistry`（注册 boost_coast_pn/dual_pulse_pn 等组合），制导律求解器不注册
- 内核签名：`integrate_guided_projectile_movement` 加 age/guid_datalink_cmd，调 control_fn 替代 guidance_fn
- 陷阱：制导律若也作参数传入控制回调→缓存版本膨胀为「控制律×制导律」笛卡尔积；max_accel 语义升级为总加速度上限（40g 防截断推力）
- 推进参数（开机时长/加速度/二次点火门限）写死在控制函数内，per-ammo SoA 参数化留后续

## 2026-08-13 Phase 6b 制导管线实施级设计定稿（docs/phase6-guidance-pipeline.md）
- 定位：phase6-guidance-design.md（定稿）→ 实施级细化（管线+算法+数据接缝），范围=制导律+电子战闭环+末敏弹，不含 AI/地雷/炮兵/CAS
- 关键补充（对原设计的落地修正/补漏）：
  1. **新增 `guid_nav_gain` SoA 数组**：nav_gain 是 per-ammo 的，同 law 不同 ammo 可不同 gain → 不能按 law 分组传标量；max_accel/fov 属平台参数按 law 分组传标量
  2. **分层原则**：制导律=纯数学（只输出理想加速度）；执行器饱和(max_accel clamp)/导引头 FOV 门限/重力 全放积分内核，不进制导律
  3. **ArmorPool 缺 RCS/IR 字段**（design §13.5 扫描需要）→ 补 `rcs`/`ir` SoA 数组
  4. **`armor.vel_or_zeros` 不存在** → 内核 target_vel 传模块级缓存 `_zero_vel`（地面恒 0，Phase 7 换真实速度）
  5. **SCAN 窗口计时**：不加新数组，SCAN 切换时清零 age + _recycle_stale 按 scan_duration_s 判超时
  6. 内核签名：stage 分支 BALLISTIC→直线 / SCAN→悬停不追踪 / TRACK/MEMORY/TERMINAL→制导律（目标来源按 stage×kind 表解析）
- 接缝现状（已核实源码）：ProjectilePool 无制导字段、无 EmitterPool、integrate_projectile_movement 直线+重力、evaluate_sensor_detection ESM 分支用默认功率（需独立 evaluate_esm_emitter_detection）、serialization _restore_projectiles name tuple 需追加
- 文件归属：combat/guidance.py（注册表+PN/PP+GuidanceStage）+ combat/sfm.py（末敏弹扫描内核+阶段切换）+ combat/weapons.py（SeekerType+AmmoConfig 字段）+ entities/pools.py + movement/integrate.py + detection/sensors.py + simulation/serialization.py
- 未实施（仅设计文档）

## 2026-08-13 🔥 修复：CPU 骤升 90%（prange 短内核线程忙等）+ 单位 z 跳动
- **CPU 90% 根因（用户报告：恢复 tick 后骤升并保持）**：
  - 每帧 2 个 `prange` 内核（integrate_armor_movement_3d + solve_suspension）唤醒 16 线程（omp threading_layer）忙等同步
  - 实测：hook 300 次 wall=71.8ms 但 **CPU=1125ms（CPU/wall=15.7x）**，60fps 烧 ~22.5% CPU（+渲染等 = 90%）；FPS 无影响（主线程仅等 0.24ms）
  - **修复 = 顺序化**：prange→range、parallel=True→False。顺序版 2048 单位 0.137ms/帧、CPU/wall=1.17x、60fps CPU≈0.94%——短内核并行是纯负优化（线程同步 > 并行收益），移动/悬挂非碰撞热点
  - ⚠️ 教训：**prange 短内核高频调用（每帧）是多核忙等烧 CPU 的元凶，wall 时间短不代表 CPU 占用低**；判断用 CPU/wall 比（time.process_time vs perf_counter）；Numba 缓存改动后需删 `.nbc`
- **z 跳动根因 + 修复（38 跳变帧→2，max|dz| 0.25m→0.02m=理论值）**：
  1. 触点跨面片时 `_sample_contact_z` 返回钳制近似（本帧错）→ resolve 修复（下帧对）→ 帧间跳变。修复：越出时返回 `(valid=False, 旧 contact_z, -1)`，solve_suspension 无效触点用旧 z 平滑延续 + **contact_mesh_id 置 -1**（待 resolve，不能保留旧面片否则永远 invalid）
  2. hook 顺序调整：resolve_contact_mesh_ids 提前到 solve_suspension 之前（触点在过渡帧也能用最新面片）
  3. 🔥 **resolve_contact hint 回归**：顺序调整后触点 hint 用 contact_z（初始 0），「桥面/地面共存」时把触点错误定位到地面。修复 = hint 改用 `pos[i,2]`（单位当前 z，触点在单位周围同层表面最可靠）
- 新增 `tests/diag_cpu.py`（GUI CPU 诊断：计时 sim.step vs paintGL）
- 全量 313 passed；smoke_gui EXIT=0
- 遗留：`_sync_boxes_kernel` 仍 parallel=True（战斗 tick 调用，非每帧；若未来战斗跑起 CPU 高需同样顺序化/降频）；diag_cpu 首次 step 780ms = Numba 编译（一次性）

## 2026-08-13 main_window 接入移动+悬挂演示 ✅
- `main_window._create_simulation`：逻辑 WorldMesh = `DualContourBuilder(hf, continuous_ground, build_adjacency=False)`（65536 地面 patch，构建 ~4s 加载期一次，加日志提示）
- 编队 16 单位（友 60,60 → (150,128) / 敌 196,196 → (110,128)）设路标点 = `GraphPathfinder(terrain.nav_graph)` 寻路（≤8 点，失败降级直线），speed=5m/s
- `loop.attach("movement", build_armor_movement_hook(armor, world_mesh, speed))`：每帧（1/60s）推进 integrate_3d → resolve_mesh → solve_suspension → resolve_contact
- 无头验证：车从 (60,60)→(150,128) 45 tick，max|pitch|=5.8° max|roll|=9.7°（fBm 起伏生效），触点全解析
- smoke_gui：逻辑 65567 面片 + render 47 面片双构建正常，EXITCODE=0（首次偶发 exit 1 = 新内核 Numba 首次编译）
- 已知：演示车到 (150,128) 停在峡谷边缘斜坡（z=-3.31, roll≈9.7°）——峡谷长桥挖深区，可接受（演示效果）
- 全量 313 passed

## 2026-08-13 悬挂模拟 + E1 采样斜面实施 ✅（docs/suspension-slope-design.md）
- **E1 采样斜面**：
  - `WorldPatch.z_at/normal_at`（FACE_TOP 轴对齐矩形 → 双线性插值 + 梯度法线；非矩形 → ray_plane_z 兜底）；模块级 `_is_bilinear_rect/_bilinear_rect/_gradient_normal`
  - `_z_bilinear_flat`（Numba）→ 后升级为 `_bilinear_rect_sample_flat`（返回 (inside, z)，bbox 含边界判定）
  - `integrate_armor_movement_3d` z 同步 + `resolve_mesh_ids` 改双线性（消除面片内移动 z 阶梯）
- **悬挂模拟（Step 2 + Step 3 触点独立面片一次到位）**：
  - ArmorPool 加 `contact_z (N,4)` / `contact_mesh_id (N,4)` / `pitch_vel` / `roll_vel`（spawn/_expand 同步）
  - `solve_suspension` Numba 内核（四触点投影 + 三点平面拟合 c0/c1/c3 + 临界阻尼弹簧 + pos.z 覆盖）
  - `resolve_contact_mesh_ids` Python fallback（触点越出面片 → STRtree hint_z 定位）
  - `build_armor_movement_hook(armor, world_mesh, speed)` 编排（integrate_3d → resolve_mesh → solve_suspension → resolve_contact），可 attach 到 SimulationLoop
  - `_sync_boxes_kernel` + `sync_from_parent` 改 Rz(yaw)·Rx(roll)·Ry(pitch) 连乘（pitch 绕车身横轴 ly、roll 绕纵轴 lx）
  - `WorldAdapter` 透传 armor_pitch/armor_roll（getattr 防御）
- 🔥 **两个关键坑（已修）**：
  1. **弹簧 dt 稳定域**：半隐式欧拉 k·dt²<1 才稳定，地面 Tick=1s + k=8 → K=8 发散 → solve_suspension 内部拆子步 dt_sub≤0.1s（n_sub=ceil(dt/0.1)）
  2. **边界点 miss**：车长 6m=3×cs 对称放置时触点全落格线上，Numba 射线法 `_point_in_poly_flat` 对边界点误判 miss → 触点一直钳制近似 → pitch 偏小（atan(1/6) 而非 atan(0.5)）。修复 = `_bilinear_rect_sample_flat` bbox 判定（u,v∈[0,1] 含边界，与 shapely covers 一致），integrate 局部性检查同步受益
- 测试：`tests/test_suspension.py`（8 用例：斜坡端到端/桥上水平/空中跳过/弹簧无过冲/触点 resolve/碰撞箱解析对比/端到端爬坡）+ `TestPatchSamplingE1`（6 用例）
- **全量 313 passed**（原 299 + E1 6 + 悬挂 8）零回归
- 遗留：E2 结构斜面（坡道）/ E3 破坏斜面（弹坑 QEF）未做；movement hook 未接入 main_window 演示（当前单位静态无路标点）；`_point_in_poly_flat` 仍被旧路径使用（未删）

## 2026-08-13 悬挂模拟 + 斜面设计定稿（docs/suspension-slope-design.md）
- 用户拍板：开始做 §5.6 悬挂 + 斜面，先写管线与算法设计（文档已建，未实施）
- 关键设计决策：
  1. **E1 采样斜面先行（悬挂前置）**：地面面片顶点 z 已平滑但 normal 恒 (0,0,1)、ray_plane_z 退化 v0.z（非插值）→ 新增 WorldPatch.z_at/normal_at（FACE_TOP 轴对齐矩形统一双线性插值 + 梯度法线）；integrate 的 z 同步改 _z_bilinear_flat
  2. **触点 z 采样 = 双线性插值**（非 ray_plane_z）：结构面片 4 顶点同 z → 双线性自然退化常量，无需类型分派
  3. **姿态解算 = 三点平面拟合**（c0/c1/c3 车身局部坐标）：b=(z1-z0)/2W→roll，a=(z3-z1)/2L→pitch，c=z0+aL+bW→底盘 z；四点最小二乘留可选增强
  4. **弹簧阻尼 = 方案 B 带速度项临界阻尼**（k, c=2√k，半隐式欧拉），需加 pitch_vel/roll_vel 字段；方案 A 无速度项 dt² 留作简化
  5. **碰撞箱/渲染共用 R_total = Rz(yaw)·Rx(roll)·Ry(pitch)**（pitch 绕车身横轴 ly、roll 绕纵轴 lx），_sync_boxes_kernel 加 pitch/roll 连乘（当前 TODO）
  6. **ArmorPool 需新增**：contact_z (N,4) / contact_mesh_id (N,4) / pitch_vel / roll_vel（spawn/_expand 同步）
  7. E2 结构斜面（坡道）= Heightfield 台阶碰撞 + WorldMesh 斜面采样解耦；E3 破坏斜面 = 完整 QEF + Hermite 法线，最后做
- 实施顺序：Step1 E1 采样斜面 → Step2 悬挂 → Step3 触点独立面片 → Step4 E2 → Step5 E3
- 性能预算：2048×4 触点全 Numba 并行 < 0.05ms

## 2026-08-13 Phase 6b 制导脚本化设计定稿（docs/phase6-guidance-design.md）
- 用户拍板：导弹制导律脚本化用 **Numba 回调函数**（不引入 Lua / 不手写 LLVM IR），零新依赖
- 核心：制导律 = 用户写 `@njit` 纯函数 → `GuidanceLawRegistry` 注册 → 作为回调传入 `integrate_guided_projectile_movement` 内核（Numba first-class function，不同回调各自多版本缓存）
- 签名（out-param 纯函数）：`(proj_pos, proj_vel, target_pos, target_vel, nav_gain, dt, out_accel)`
- `ProjectilePool` 加 `guid_target`/`guid_law` 两 SoA 数组；`AmmoConfig` 加 `guidance_law_id`/`nav_gain`；`FiringControl` 按 `guid_law` 分组调内核
- 内置 PN 比例导引 + PP 纯追踪；`guid_law=-1` 直射退化（目标丢失停导直射，脱靶由现有 `_recycle_stale` 回收）
- 电子战闭环（用户追加）：导弹/制导也是辐射源 → 新增 `SeekerType`（ACTIVE_RADAR/SEMI_ACTIVE_LASER/IR/PASSIVE_RF/TV_EO）+ `EmitterPool`（SoA）+ `guid_target_kind`（0=装甲 1=辐射源）；主动雷达导引头可被 ESM 检测、被 ARM（PASSIVE_RF 追辐射源）跟踪；辐射源关机→**记忆模式（惯性制导飞向最后已知位置）+ 双模末端引导（terminal_seeker 锁定本体）**，`GuidanceStage`（TRACK/MEMORY/TERMINAL）+ `guid_stage`/`guid_memory_pos` 两数组；ESM 检测辐射源走独立内核 `evaluate_esm_emitter_detection`（现有 `evaluate_sensor_detection` ESM 分支用默认功率且目标是装甲，需单独打通）
- 末敏弹（SFM）综合范例（§十三，用户考题）：`AmmoType.SENSOR_FUZED=35` + `SeekerType.SENSOR_FUZED=6` + `GuidanceStage.BALLISTIC=3/SCAN=4`；曲射→开伞→双模扫描（RCS+IR 双阈值都达标才引爆，抗诱饵）→ EFP 攻顶（`top_armor_factor` 顶部=正面30%，`is_heat` 语义固定穿深）；SCAN 毫米波 emitter 激活可被 ESM 检测/ARM 反辐射（暴露窗口=scan_duration_s，设计权衡）；扫描参数存 AmmoConfig 按 ammo 分组传入内核，不新增 SoA 数组；扫描内核 O(M×N) 预留空间哈希优化
- emitter 清理机制（用户追问补全，§12.2）：`activate`（free_stack 弹索引）/`deactivate`（战术关机**保留索引**可重新开机）/**级联失效**（父实体装甲/弹丸已死 → inactive + 索引归还 free_stack 防池耗尽）；`_sync_emitters_kernel` 每帧级联兜底（Python 层漏关也会被清），双保险：`_destroy_projectile/_destroy_armor` 显式关 + sync 兜底
- Lua 留档为「未来高层行为层（mod）」，绝不进热路径

## 2026-08-13 Phase 5d anime 主题 ✅（docs/anime-theme-design.md）
- **新增 `assets/shaders/anime/`**（terrain.vert 复制 default + terrain.frag + theme.toml）：cel 3 级色阶（smoothstep 双阈值）+ 暗部天光补色 + 粉彩调色（saturation/brightness 保留 v_color 色相）+ 法线描边 + 大气透视
- **关键算法决策（实测驱动）**：
  1. 🔥 **深度通道 early-z 失效**：`fwidth(gl_FragCoord.z)` 在 Qt FBO + GL_LESS 下恒 ≈0——背景像素（clear depth=1.0）被早期深度测试提前丢弃 → 2×2 quad 不完整 → 导数退化。设计稿「NDC 深度常数性」推导对剪影不成立（剪影需背景像素参与导数）。**u_depth_edge_scale 默认 0，代码保留供 G-Buffer 方案**
  2. 🔥 **法线边缘平方锐化（消除地面网格线）**：平滑法线中心差分跨格连续，但相邻三角形插值后屏幕导数仍有 ~0.02~0.04 微变 → 线性阈值描出网格线。`n_edge = n_edge_raw²` 把弱边缘（0.029²≈0.0008）压到阈值下、强棱线（0.065²≈0.0042+）保留 → 地面干净、桥棱线清晰
  3. 🔥 **描边统计必须用开关对照法**：cel 暗部天光补色接近描边色 → 「颜色接近 outline_color」统计有 ~24% 假阳性。正确方法 = 正常渲染 vs u_outline_threshold=10（描边关）两帧差异像素
- **最终参数**：u_outline_threshold=0.008 / u_depth_edge_scale=0 / u_normal_edge_scale=1.5 / 大气透视 u_fog_near=300/far=3000；相机 near=3/far=12000 固定常量写死 theme.toml 即可（零代码改动）
- **新增测试脚本**：`tests/smoke_theme_anime.py`（主题 5 轮热切换 + 描边开关对照 diff_px≈10536→0.89% + cel lum_peaks=2/lum_mean=74.5）+ `tests/diag_anime_edge.py`（u_debug_edge 可视化边缘值分位数标定）+ `tests/diag_anime_outline.py`（阈值/权重扫描 + 消融截图）
- **frag 保留 u_debug_edge uniform**（默认 0）：1=d_edge 2=n_edge_raw 3=edge 可视化，未来调参用
- 全量 **299 passed** 无回归；热切换 5 轮稳定；效果：粉彩亮调 + 桥黑描边清晰 + 地面干净无网格线 + 远景粉紫大气透视
- 遗留：深度剪影描边需 G-Buffer 方案 B；HUD 主题化留 Phase 8

## 2026-08-13 Phase 5b + 5c 实施 ✅（面板布局 + HUD/拾取）
- **WorldAdapter 扩展**：加 ammo/fuel 视图（from_snapshot 读 snapshot.logistics）+ unit_name/side_name（渲染层命名规则，不碰引擎）+ speed_at(占位 0) + get_stats()（级别 0 聚合：兵力/平均弹药燃料/时间）
- **`Airis/render/picking.py`**（新增）：project_world（world→屏幕像素，O(N) 向量化）+ pick_nearest/Picker（O(N) 屏幕距离最近，max_px=30）
- **`Airis/render/hud/overlay.py` + `log_widget.py`**（新增）：HUDOverlay（moderngl 屏幕空间圆点：NDC 实例化 quad，fragment 抗锯齿圆；选中高亮大圆黄色）+ LogWidget（Qt 透明滚动日志，QListWidget，200 行上限，layout_overlay 定位左下角）
- **GLWidget**：移除旧 3D 实例化 quad 单位标记（Phase 5a 内联）→ HUD pass（_render_hud：投影→圆点，**HUD 期间 ctx.disable(DEPTH_TEST)** 保证永远最上层）；拾取信号 unit_hovered/unit_selected(int)（-1=空白）；鼠标：press 记录→移动超 4px 判拖拽（orbit）/否则单击选中；Esc 取消；setMouseTracking+StrongFocus
- **`Airis/ui/` 四面板**（新增）：narrative_overlay（对话 Tab1：speaker/text/选项按钮→option_selected(dialogue_id,index)/历史）、detail_panel（详情 Tab4 三级信息 0 全局→1 悬浮→2 选中，QTextBrowser HTML）、archive_panel（档案 Tab2 骨架：时间线列表+刷新）、briefing_panel（简报 Tab3 骨架：生成简报按钮→to_text）
- **main_window 装配**：AmmoManager/FuelManager 演示数据（set_unit，Snapshot.logistics 生效）+ CombatEventRecorder(:memory:, unit_registry/side_map) + set_unit_context + QTabWidget SidePanel + LogWidget 叠加 + 面板数据源回调（_recent_events_for/_timeline_for/_generate_briefing，getattr 延迟防御 _build_ui 阶段）
- **验证**：299 passed（含 2 个投影回归测试）；smoke_gui 通过；verify_hud 像素统计 ally=1585/enemy=2596 ✓
- 🔥 **glm.mat4.to_list() 转置陷阱（2026-08-13，HUD 圆重合/飞出根因）**：
  - 症状：**所有单位 HUD 圆点投影到同一屏幕点**（友军/敌军完全重合）；特殊角度 clip_w 达 1e4~1e5（实际 ~450）→ NDC 爆炸 → 圆点剧烈位移飞出屏幕
  - 根因：`glm.mat4.to_list()` 返回 **4 个列向量**（非扁平 16 元素）！`np.array(to_list()).reshape(4,4, order='F')` 是 **no-op**（已 (4,4)，reshape 同形状直接返回原数组，行=数学列）→ 矩阵被转置 → clip = h @ m_col.T 错位
  - 修复：改用 `np.frombuffer(mvp.to_bytes(), dtype=np.float32).reshape(4,4, order='F')`（与 GPU uniform 完全一致的 column-major 扁平，已验证与 glm 乘法逐元素一致）
  - 教训：**glm 矩阵转 numpy 必须用 to_bytes()（或显式 to_list 转置），不要假设 to_list() 是扁平数组**；identity 矩阵转置不变 → 单测测不出，必须加非 identity 相机投影对比测试
  - 测试：test_project_matches_glm（真实相机 MVP 逐点对比 glm）+ test_project_distinct_positions（不同世界位置→不同屏幕点，回归）
- 🔥 **HUD 深度测试陷阱（2026-08-13）**：HUD 圆点 NDC z=0（深度值 ~0.5），GL_LESS 下近处地形（深度<0.5）会把标记剔除 → 低角度观察圆点消失。修复 = _render_hud 期间 `ctx.disable(DEPTH_TEST)`（HUD 语义=永远最上层），finally 恢复
- 遗留：Level2 速度占位 0（ArmorPool 无 velocity）；HUD 文字标签/弹道线/命中标记（设计文档 §4.3）留待后续；anime 主题 Phase 5d 未做；verify_hud/diag_hud_projection 为诊断脚本

## 2026-08-13 Phase 5a 管线骨架 ✅（docs/phase5-ui-pipeline-design.md）
- **新增 `Airis/render/adapter.py`**：WorldAdapter（渲染数据源抽象，Phase 5 透传 Real → Phase 8 切 Twin）+ UnitRenderStatus + `_build_damaged_parts`（box.parent≥0 & ~active & armor_active[parent] 聚合 5 部位损坏掩码）+ PART_NAMES
- **新增 `Airis/render/theme.py`**：ThemeManager（discover/load/switch 热切换：先通知回调重绑 VAO 再 release 旧 program）+ Theme；`SHADER_ROOT = Path(__file__).parents[2]/assets/shaders`
- **着色器外提**：`assets/shaders/default/` + `constructivism/`（terrain.vert/frag + theme.toml [uniforms]）；GLWidget 内联着色器删除
- **GLWidget 重构**：构造接受 `theme_manager`（默认自建加载 default）；buffer 缓存（_ground_vbo 等）支持主题切换重建 VAO；`set_adapter()` + 单位标记渲染（实例化 quad，读 WorldAdapter.armor_pos/active/side）
- **MainWindow 骨架**：QSplitter(GLWidget:SidePanel 占位) + 主题菜单（ThemeManager.switch 热切换）+ 叙事「跳过对话」菜单 + 主循环 `_on_frame`（Simulation.step → WorldAdapter.from_snapshot → NarrativeEngine.tick → set_adapter → update）+ 状态栏（暂停/模拟时间/FPS/单位/主题/相机）；演示编队：友军 8 + 敌军 8（ArmorPool 动态附加 side 属性演示敌我，Phase 8 正式接入）
- **main.py**：相机对准战场中心，distance=200
- **验证**：288 passed；smoke_gui 通过（开场对话 prologue_001 激活暂停 ✓）；单位标记红绿像素统计确认渲染正确
- 🔥 **moderngl 实例化必须加 `/i` 后缀（重要坑，2026-08-13 排查定位）**：
  - 症状：实例化 quad 只渲染前几个实例（2 个"正常"、16 个只画前 2-3 个），standalone 与 Qt 环境都复现
  - 根因（Reference/moderngl/src/moderngl.cpp 源码级）：`FormatIterator::info()` 里 divisor 由 format 字符串后缀决定——`/i`→divisor=1（实例化）、`/r`→0x7fffffff（整组）、`/v`→0（per-vertex 默认）。**moderngl 不会自动把第二个 buffer 设为 instanced！**
  - 修复：`vertex_array` 的实例 buffer format `"3f 1f 3f"` → `"3f 1f 3f /i"`
  - ✅ **官方文档确认**（Reference/moderngl/docs/topics/buffer_format.rst）：`/usage` 三值——`/v` per vertex（省略默认）、`/i` per instance（实例内所有顶点共享同值）、`/r` per render（似 uniform）；「多 VBO 时第一个必须 /v」。我方实现（quad 默认 /v + 实例 /i）完全合规；替代方案 `VertexArray.bind(..., divisor=...)`（不推荐）
  - 教训：2 实例实验的「成功」是误判——divisor=0 时实例数据被当 per-vertex 连续读取，画出一个被拉长的三角形，红绿各占一部分，看起来像"都渲染了"
- 🔥 **主题切换两坑（2026-08-13 修复，缺一不可）**：
  1. **InvalidObject**：`ThemeManager.switch` 释放旧 program 后**必须移出缓存**（`self._themes.pop(old.id, None)`）——否则切回该主题时 `ensure()` 复用已释放的 program，`ctx.vertex_array()` 抛 `TypeError: argument 1 must be mgl.Program, not InvalidObject`。切回会重新编译（glo 复用 id 但对象有效）
  2. **地形变黑（context 非 current）**：菜单槽触发时 GL context **不保证 current**，此时编译 program / 重建 VAO / 写 uniform 会静默失败 → 地形变黑。修复 = `MainWindow._switch_theme` 整个 `tm.switch()` 包进 `gl_widget.makeCurrent()/doneCurrent()`（覆盖 program 编译）+ `gl_widget._on_theme_switched` 里 GL 操作（rebuild VAO + apply_uniforms）再包一层 makeCurrent（防御其他调用路径）
  - 验证：GUI 自动切换 6 轮（default↔constructivism），各主题像素稳定（default 暗绿 42.5 亮度 / constructivism 红黑 35.5），单位标记红绿像素每次一致不受影响
- 📌 用户建议（未实施）：QOpenGLWidget 里应 `moderngl.get_context()` 重用 Qt 上下文，而非 `create_context()`（避免上下文累积触上限）；当前 GLWidget 仍用 create_context(require=460)（AGENTS.md 沿用，地形正常），待评估
- 遗留：单位标记 Phase 5c HUD 化（世界坐标→屏幕投影正交绘制）；adapter.armor_side 依赖动态附加（Phase 8 正式字段）

## 2026-08-12 Phase 4a 叙事层骨架 ✅（docs/phase4-narrative-design.md）
## 2026-08-12 Phase 5 管线设计定稿（docs/phase5-ui-pipeline-design.md）
- 整合「数据链与战争迷雾设计」：WorldAdapter 抽象层（Phase 5 透传 Real → Phase 8 切换 Twin）
- 渲染管线永不直读 Real 数据——GLWidget/HUD/InfoPanel 全通过 WorldAdapter
- 布局: QSplitter(GLWidget 3: SidePanel 1) + QTabWidget(对话/档案/简报)
- 着色器主题系统: default/constructivism/anime 三主题，热切换<100ms
- 主循环: UI线程持有SimulationLoop+NarrativeEngine，_on_frame()串行
- 实施顺序: 5a(WorldAdapter+主题) → 5b(面板) → 5c(HUD) → 5d(着色器)


- **`Airis/narrative/` 5 模块全部搭建**（设计文档 §3）：
  - `dialogue.py`：DialogueTreeConfig/DialogueNodeConfig/ChoiceOptionConfig（Pydantic YAML 加载）+ DialogueEngine 状态机（text/choice/branch/action 四类型 + set_flags + 条件选项 + pending_actions）
  - `triggers.py`：TriggerRule/TriggerCondition/TriggerAction（Pydantic）+ TriggerSystem（event_match/state_check/on_first_tick/flag_set/composite 五模式 + priority 降序 + once + cooldown）+ **ConditionEvaluator（ast 白名单安全求值）**
  - `commands.py`：CommandFactory 纯函数（action dict → Command，字符串类型→CommandType 映射）
  - `recorder.py`：**CombatEventRecorder（环形缓冲区 + SQLite 三表 combat_events/mission_summary/unit_stats）+ BriefingGenerator（战果统计/时间线/单位战报/关键事件高亮/友军误击/结局判定）+ CombatBriefing.to_text()**
  - `engine.py`：NarrativeEngine 中央调度器（tick → recorder + trigger → dialogue；on_player_choice → CommandFactory → CommandQueue.push；generate_briefing；首帧开场对话）
- 🔥 **三坑已修（重要教训）**：
  1. **Python3 set 没有 `.contains()` 方法**（Python2 遗留）→ `flags.contains(x)` 表达式全部失败。修复 = ConditionEvaluator 加 `_ContainsTransformer`（ast.NodeTransformer 把 `flags.contains(x)` → `x in flags`）
  2. `_match_state` 在 state=None 时直接 return False → `sim_time_greater_than` 这类不依赖快照的检查也失败。修复 = 不依赖 state 的检查移到 state 判空之前
  3. **destroyed 事件不含 shooter**（模拟层 `engagement.py` 只 emit `{armor_idx, cause}`）→ 击杀归属丢失（kills=0）。修复 = recorder 维护 `_last_hitter`（target→最近命中者，DCS 最后一击者语义），hit 事件更新、destroyed 时回填 source_id
- **新增 `tests/test_narrative.py` 34 用例**：对话加载/推进/分支/条件选项/非法引用/重复 id + ConditionEvaluator 安全求值（拒任意名称/属性链/非法语法）+ 触发器匹配/优先级/once/cooldown/state_check + CommandFactory + recorder（环形缓冲区容量/事件查询/简报生成/友军误击/迷雾过滤）+ engine（首帧开场/触发器激活对话/玩家选择指令入队/结束恢复/重置）
- **全量 284 passed**（原 250 + 新增 34）；端到端冒烟（SimulationLoop + NarrativeEngine + recorder）：fire/hit/destroyed 记录、击杀归属正确、简报 victory + 时间线 + 关键事件
- 遗留：`detection` 事件埋点（用户拍板**不加**）；ConditionEvaluator 的 `flags.add` 转换未实现（仅 contains）；renderer.py（NarrativeWidget Qt 面板）为 Phase 4b（用户拍板**暂缓**）

## 2026-08-12 Phase 4c 集成 + 示例数据 ✅（docs/phase4-narrative-design.md §6）
- **示例数据（story/）**：
  - `story/dialogues/prologue.yaml`（序章 5 节点：text→choice(3 选项含条件选项)→HOLD/REDEPLOY action + set_flags）+ `first_blood.yaml`（首次交战 7 节点：text→choice→branch 演示 + set_flags）
  - `story/triggers/chapter1.yaml`（3 条：on_first_tick 开场 / event_match destroyed filter side=enemy 首次击毁 / state_check sim_time_greater_than 300）
- **集成测试 `tests/test_narrative_integration.py`（4 用例）**：完整 Tick 循环（SimulationLoop + NarrativeEngine + recorder）：首帧开场对话→玩家选择→HOLD 指令入队→模拟层消费（command 事件）→战斗事件→首次击毁激活 first_blood→重组指令→SQLite 事件断言（击杀归属回填）→简报 stalemate→跳过对话→任务重置清空
- 🔥 **集成踩坑（4 处，均已修）**：
  1. **set_flags 应用错节点**：advance 里只应用「当前节点」set_flags → 玩家选择后的目标节点 flag 不设置。修复 = 离开时应用当前节点 + 进入时应用目标节点 set_flags（双语义）
  2. **缺 engine→loop.sim_paused 接线**：NarrativeEngine 加可选 `loop` 参数 + `_sync_pause()`（docs §零「叙事层可设置暂停」）；对话激活/结束/跳过/重置时同步。另加统一推进接口 `advance_dialogue(option_index)`（直接调 dialogue_engine 绕过缓存是坑）
  3. **触发器 filter side 不匹配**：模拟层 destroyed 事件无 side 字段 → `filter: {side: enemy}` 失效。修复 = TriggerSystem.evaluate 加 `side_map` 参数，`_filter_matches` 对 side/source_side/target_side 特殊键用 side_map 推断事件涉及方阵营（destroyed 用 armor_idx、fire/hit/miss 用 shooter、detection 用 source）
  4. **recorder 查询不 flush**：query/timeline/all_events 读 SQLite 但事件还在环形缓冲区 → 空结果。修复 = 查询前自动 flush()。另 BriefingGenerator.generate 加 side_map 参数（未出现事件的单位靠 side_map 推断阵营，否则 enemies_initial 少算→结局误判 victory）
- 全量 288 passed（原 284 + 集成 4）；用户拍板：detection 埋点不加、Phase 4b（Qt UI）暂缓
- 遗留：简报面板 UI（Phase 4b 一起做）；`advance_dialogue` 已替代直接 dialogue_engine.advance

## 2026-08-12 渲染去 continuous + 桥墩空心/底面修复 ✅
- **用户决策：地面渲染直接操作 Heightfield span，不要依赖 continuous_ground**
  - `WorldMeshRenderer` 去掉 `continuous_ground` 参数；`_build_ground_map` 直接遍历每列 span 取地面顶面层 z（`spans[col,0,_ZMAX]` → `_to_z`）
  - map.py to_render_data / test_worldmesh_render.py 同步去 continuous_ground 传参
  - 本质：skip dual 地面（include_ground=False 加快生成）+ 地面 z 直接从 span 来
- 🔥 **桥墩空心修复**：桥面板底面被桥墩紧贴支撑(gap=0) → 不暴露 → 桥墩列底面缺失、桥墩无顶 → 空心柱
  - 修复：底面判定加 `diff_on_struct` =（下方不同材质 & 下方非 TERRAIN）→ 桥面板底在桥墩列生成，桥墩实心
- 🔥 **桥墩底面误生成修复（用户反馈「几个侧面不贴合」）**：`diff_on_struct` 第一版只查「不同材质」→ 桥墩底（下方 TERRAIN 地面）也生成水平底面（量化层 z=-9 vs 峡谷斜坡 -8.6）→ 悬空面不贴合
  - 修复：加 `prev_mat != MAT_TERRAIN` 条件——长在地面的结构（桥墩/贴地建筑）不生成底面（被地面吸收）；被结构支撑的面（桥面板被桥墩托）生成
  - 最终：桥墩 = 4 面侧墙 + 顶（桥面板底盖）+ 底开放（从地面长出）；桥面板底 1 大面片覆盖全区
- 全量 256 passed

## 2026-08-12 峡谷长桥（山体切开 + 桥墩）✅（factory.py CanyonBridgeConfig）
- 用户反馈：默认地形的桥「浮空」（桥面锚定桥区最高点+lift，fBm 落差 9~15m → 低洼处悬空十几米）+ 无桥墩 → 要求现实地形
- **`CanyonBridgeConfig(BaseModel)`**（factory.py）：x0/y0/x1/y1 桥区 + canyon_frac（峡谷占长比例，默认 0.6）+ canyon_depth（挖深 14m）+ pier_interval（桥墩间隔 16 格）+ pier_width（2 格）
- **`_carve_canyon`**：桥区中间 canyon_frac 挖深成峡谷（山体被切开，原地改 ground），两端保留作桥台；桥面锚定 = 桥台最高点 + lift
- **`_add_canyon_bridge_spans`**：桥面板横跨全区（MAT_BRIDGE）+ 峡谷段每隔 pier_interval 立 pier_width 宽桥墩
- 🔥 **桥墩必须用 MAT_BUILDING（混凝土）**：若用 MAT_BRIDGE 会与桥面板 [z_bot,z_top] **相邻同 mat 合并成实心堤坝**（add_span 合并逻辑）——桥变坝！实测 count 从 3 变 2
- **gl_widget._default_terrain()**：fBm 256×256 + 峡谷长桥（40..215, y=128..131, 352m 长）+ 2 座普通桥；DualContour 无地面构建 0.02s，to_render_data 0.41s
- 效果：峡谷挖深后谷底 -21.5m，桥墩从谷底支撑到桥面 12m（混凝土色），桥面板 132 三角形
- 测试 `tests/test_canyon_bridge.py`（6 用例）：峡谷挖深/桥面横跨/桥墩 3 段 span/桥墩不合并/桥墩侧墙提取/渲染含混凝土桥墩
- 全量 256 passed（250+6）

## 2026-08-12 默认演示地形 + render_world_mesh 性能优化 ✅
- **`gl_widget._default_terrain()`**：纯平坦 → fBm(256×256) + 3 座界内演示桥（用户要看 WorldMesh Phase B/D 效果：平滑地面 + 桥完整 3D）
- 🔥 **性能优化（关键）**：fBm 256×256 的 `to_render_data` 实测 **8.12s → 0.44s**（18 倍）
  - 根因：DualContourBuilder 逐格地面产生 65536 个 patch（渲染地面根本不需它们——走 continuous_ground）+ build_adjacency 扫描线 O(n·k²) 对 65536 patch 爆炸
  - 修复：`DualContourBuilder` 加 `include_ground` / `build_adjacency` 参数；`TerrainMap.render_world_mesh` 用 `include_ground=False, build_adjacency=False`（渲染专用，只需结构 patch）
  - `WorldMeshRenderer._build_ground_map`：无地面 patch 且无 continuous_ground → 回退 Heightfield 量化层
- 全量 250 passed
- 教训：**渲染用 WorldMesh 与逻辑用 WorldMesh 参数不同**——逻辑要地面 patch（z 采样）+ 邻接（寻路），渲染只要结构；逐格地面在 256×256 即 6.5 万 patch，是大坑

## 2026-08-12 WorldMesh Phase D 实施 ✅（渲染直接从 WorldMesh，docs/worldmesh-3d-surface-design.md §7）
- 用户拍板：Phase C（NavGraph 从 WorldMesh 重烘焙）缓一缓——NavGraph 依赖 spans，与 WorldMesh 结果类似，收益低
- **surface.py 新增 `WorldMeshRenderer`**（数据源统一为 WorldMesh，不再从 Heightfield 独立提取）：
  - 地面：continuous_ground（若提供，与旧 SurfaceExtractor 完全一致）或 WorldMesh 地面 patch 导出格中心 z_map → SUBDIV 细分平滑网格（保留连续法线 + 高度色，无斜纹回归）
  - 结构（顶/底/侧）：WorldMesh patch 直接三角化（扇形），法线 = patch.normal，颜色 = mat 色——**Phase B 完整 3D（桥侧墙/桥底）首次进入渲染**
  - struct_offset 记录结构起始（to_render_data 按此分离 ground/结构，接口不变）
- **TerrainMap**：新增 `render_world_mesh` property（DualContourBuilder 完整 3D，懒加载）；`to_render_data()` 改用 WorldMeshRenderer（替代 SurfaceExtractor）
- **dual_contour.py 修正（🔥 重要）**：`_build_sides` **跳过 MAT_TERRAIN 侧面**——地面 span 从世界底(z_min=-50)延伸到地表，地图边界会产出 52m 巨墙（400×400 fBm 的 bridge 顶点几乎全是地面边界墙！）；地面由平滑网格渲染 + Heightfield 碰撞兜底。只给结构生成侧面
- 测试 `tests/test_worldmesh_render.py`（7 用例）：渲染数据格式/地面法线单位化/桥完整 3D（顶+底+侧）/ground-bridge 分离/纯平地无桥/render_world_mesh 完整 3D
- 全量 250 passed（243+7）
- 遗留：SurfaceExtractor 仍保留（diag/smoke 旧脚本用，未废弃）；Phase C 未做（用户缓办）；fBm 默认 400×400 DualContour to_render_data 较慢（逐格地面 patch，已知风险）

## 2026-08-12 WorldMesh Phase B 实施 ✅（Dual Contouring 简化版，docs/worldmesh-3d-surface-design.md §4）
- **新建 `terrain/dual_contour.py`**：`DualContourBuilder`（column-span 轴对齐盒 → 完整 3D WorldMesh）
  - 暴露面判定 numpy 向量化：顶面（最顶/上方有隙）+ 底面（si>0 下方有隙）+ 侧面（邻居列同 z 区间无覆盖）
  - **地面顶面逐格四边形**（🔥 关键决策）：GreedyMesh 合并会丢内部顶点 → 平滑失效（非线性起伏中心 z 丢失）。结构/底面才合并（平面不丢信息）
  - **结构/底面 GreedyMesh 合并** + **侧面一维合并**（按固定坐标+z 层带分组，沿墙轴 run-length；桥 3×3 每面墙合成 1 片）
  - B2 平滑：MAT_TERRAIN 顶面顶点 z ← continuous_ground 双线性插值（`_bilinear_z`，格中心场，坐标 -0.5 偏移同 surface.py）
  - B4 法线：FACE_TOP/BOTTOM(±z)、FACE_PX/NX/PY/NY(±x/±y)
  - walkable：仅顶面（暴露表面天然排除被覆盖层）
- **world_mesh.py 扩展**：WorldPatch 加 `face` 字段（默认 FACE_TOP）；FACE_* 枚举 + `_FACE_NORMALS`；`_build_adjacency_3d` 提取为模块级 `build_adjacency(mesh, max_climb)`（WorldMeshBuilder/DualContourBuilder 共用）
- **TerrainMap 支持构建器切换**：`world_mesh_builder=DualContourBuilder` 参数（默认 WorldMeshBuilder）；两个 builder 统一接口（continuous_ground 参数）
- 🔥 **STRtree contains→covers（重要 bug）**：逐格地面面片共享边/角点，shapely `contains` 对边界点返回 False → 定位漏掉站在格线上的单位。改 `covers`（含边界）
- **测试 `tests/test_dual_contour.py`**（11 用例）：平坦地形/桥 6 面/侧面合并/桥下地面保留/双线性平滑/平台起伏逐格/无平滑回退/STRtree hint_z/邻接对称/TerrainMap 集成
- 全量 243 passed（232+11）
- 遗留：侧面法线 `_build_sides` 的材质反查用 col_ref 匹配 [z0,z1)；世界边缘侧墙会生成（几何正确）；生产规模逐格地面面片数大（已知风险，设计文档风险表承认，QEF 斜面/合并留未来）；Phase C（NavGraph 从 WorldMesh 重烘焙）/ Phase D（渲染）未开始

## 2026-08-12 WorldMesh Phase A 实施 ✅（docs/worldmesh-3d-surface-design.md）
- **新建 `terrain/world_mesh.py`**：WorldPatch（3D 顶点+法线+Newell 面积/形心）/ WorldMesh（STRtree 懒索引+adjacency+portal_edges）/ WorldMeshBuilder（复用 GreedyMesh 分组，输出 3D 面片）/ flatten_world_mesh（Numba 展平）
- **关键设计决策（与用户讨论，重要）**：
  1. 移动热路径 mesh_id 追踪**不用邻接表**——利用「单位移动连续」物理事实：Numba 局部性检查（point_in_poly 当前面片）99% 命中；miss 置 -1 → Python 侧 STRtree `locate_patch(x,y,hint_z=z)` fallback（`movement/integrate.py::resolve_mesh_ids`）
  2. 邻接表仅供 A*/SSFA 寻路（纯 Python），不进热路径
  3. 生产规模（4000×4000~8000×10000）**不建全量 Grid Hash**（cell_start 12.8MB 浪费）——STRtree 内存 O(面片数) 与格数无关
  4. Pathfinding 短期保留 NavGraph（SurfaceMesh/WorldPatch GreedyMesh 地面每格一面片 → 节点爆炸）；Phase B（Dual Contouring 出大面片）后再评估融合
- **ArmorPool 扩展**：mesh_id(int32, -1=空中)/pitch/roll/suspension（悬挂预留，Phase A 恒 0）；spawn 接受 mesh_id 参数；_expand 同步扩容
- **integrate.py 新增**：`integrate_armor_movement_3d`（Numba 并行：移动+面片局部性+z 同步）、`_point_in_poly_flat`/`_ray_plane_z_flat`（展平数组版）、`resolve_mesh_ids`（Python fallback）
- **TerrainMap 新增 `world_mesh` property**（懒构建，与 surface_mesh/nav_graph 并列）；terrain/__init__.py 导出 WorldMesh/WorldMeshBuilder/WorldPatch
- 🔥 **STRtree 首个候选 bug**：locate_patch 的 best_dist 初始 inf 但无 hint_z 分支用 `z > best_dist`（取最高面）→ `z > inf` 恒 False → 首候选必丢 → 全 -1。修复=无 hint 用 -inf
- **测试 `tests/test_world_mesh.py`**（15 用例）：WorldPatch 3D/斜面积、Builder（平坦/带桥/fBm）、STRtree 定位（mat 过滤/hint_z 桥下）、ray_plane_z、展平 roundtrip、integrate_3d z 同步/穿越、TerrainMap 集成
- 全量 232 passed（217+15）
- 遗留：gl_widget 无单位渲染（Model 矩阵 pitch/roll 预留 N/A）；BoxPool._sync_boxes_kernel 加 TODO（pitch/roll 连乘）；Phase B Dual Contouring 未开始

## 2026-08-12 导航网格重构设计
- 废弃 GreedyMesh（`surface_mesh.py`），原因：按精确 (layer_z, mat) 分组导致斜坡每格独立 → 面片爆炸
- 新方案：可行走格提取 → BFS 区域生长（逐对 z 比较 + mat 硬约束）→ shapely 轮廓 → Ear Clipping 三角剖分 → 三角形 = 导航节点
- 详见 `docs/navmesh-region-growth-design.md`
- 旧方案归档于 `docs/surface-mesh-unification.md`（^ 标记）（注意：本文件编号与 AGENTS.md 有错位）

## 2026-08-12 导航网格重构骨架 ✅（docs/navmesh-region-growth-design.md 实施起步）
- **新建 `terrain/navmesh_builder.py`**（新管线骨架，不动现有 surface_mesh.py，渐进式重构）：
  - 数据模型：`WalkableConfig` / `WalkableCell` / `NavTriangle` / `NavGraph`（含 vertical_portals）
  - 已实现：可行走判定（`is_top_surface`/`slope_at`/`is_hanging`/`find_neighbor_z`）、`extract_walkable_cells`、`grow_regions`+`build_neighbor_map`（BFS+mat 硬约束+逐对 z）、`region_to_polygon`/`polygon_to_list`（shapely）、`triangulate_polygon`（**凸扇形**；凹/带洞抛 NotImplementedError）、`build_navgraph`/`canonical_edge`（共享边邻接+portal，浮点量化 key）、`locate_triangle`/`point_in_triangle`
  - 待填充（NotImplementedError）：凹多边形 Ear Clipping（§3.4.2）、带洞桥接法（§3.4.3）、跨层垂直 portal（§3.6）
  - `NavMeshBuilder` 编排 6 步；**骨架降级策略**：跨层/凹区域未实现 → 跳过+warning，凸区域地图端到端可跑
  - ⚠️ 邻域用**四邻域**（docs §3.2.2 代码为准，与第二节「8 邻域」描述不一致，已按代码实现）
- `terrain/__init__.py` 追加导出 NavMeshBuilder/NavGraph/NavTriangle/WalkableCell/WalkableConfig（旧导出保留）
- 新增 `tests/test_navmesh_builder.py`（31 用例：数据模型/判定/提取/区域生长/轮廓/三角剖分/邻接/点定位/端到端）；全量 209 passed
- 待办（后续迭代）：实现凹多边形 ear clipping + 洞桥接 → 跨层垂直 portal → 切换 movement/nav.py + pathfinding.py 到 NavGraph（docs §7/§8）→ 删除 navmesh/baker.py 与 surface_mesh.py GreedyMesh 路径

## 2026-08-12 导航网格管线完整化 ✅（docs §3.4/§3.6/§4 实施，含 NavGraph 寻路）
- **④ 三角剖分完整实现（移植 Recast RecastMesh.cpp）**：
  - 判定体系：`_area2/_left/_left_on/_collinear/_intersect_prop/_between/_intersect/_vequal`（含 improper 相交）+ `_diagonalie/_in_cone/_diagonal` + loose 变体
  - `_ear_clip` = Recast `triangulate`：预计算 ear 标志 + **最短对角线耳优先** + loose 恢复 + 保险丝；桥接重复顶点由 `vequal` 跳过容忍
  - **洞桥接（Recast mergeContours 思路）**：`_leftmost_index`（最左点，非最右）+ `_find_visible_vertex`（V→H 最短不相交对角线，用 `_intersect` 判定 + 跳过 p 外环邻边/q 洞邻边）+ 缝合序列（V、H 各两次）
  - 🔥 **方向约定（重要）**：Recast 外环 CW、洞 CCW。**shapely 不保证洞方向**（实测传入的洞保持原 CCW 顺序）→ 必须按 `_polygon_signed_area`（shoelace）强制：外环 >0(CCW) 翻转、洞 <0(CW) 翻转
  - 🔥 **桥顶面 z 语义**：`add_span(z0,z1)` 是半开区间 [z0,z1)，顶面世界高度 = `_to_z(z1 层)`。想让桥面 z=4（与地面 z=2 差 2=max_climb）必须 `add_span(3.0, 4.0)`（span [13,14) 层，gap=1 保住地面顶面）；`add_span(4.0,6.0)` 桥面实为 z=6（差 4 无 portal）——跨层测试第一次因此全红
  - 教训：`_orient` 被我重写时删除但 `_segments_intersect_inclusive` 仍引用 → NameError；已改用 `_intersect`（同语义更健壮）
- **⑥ 跨层垂直 portal 完整实现**：`link_cross_layer`（pairs TERRAIN-BRIDGE/BUILDING + BRIDGE-TUNNEL，bbox 空间哈希 cell=50，|z 差|≤max_climb + `xy_overlap`）+ `_inject_vertical_portal`（注入 adjacency + 退化点 portal (p,p)，docs §3.6.3）
- **NavGraph 寻路集成（docs §4，渐进式不破坏旧 API）**：
  - `movement/nav.py`：新增 `navgraph_astar` + `_graph_heuristic/_graph_edge_cost`（z 落差 ×3）
  - `movement/pathfinding.py`：抽 `_collect_portals_generic` + `_funnel_core`（SSFA 主体与图结构解耦）；`funnel`/`_collect_portals` 变薄包装（SurfaceMesh 兼容不变）；新增 `funnel_graph` + `GraphPathfinder`（退化点 portal 强制经过）
- 测试：`tests/test_navmesh_builder.py` 38→39 用例（含 `test_builder_build_ring_with_hole` 环形带洞端到端：外环 8 三角形避开 30×30 洞 + 坑底独立 2 三角形 + 面积断言 2500/1600）；全量 217 passed
- 遗留：`surface_mesh.py`（GreedyMesh）与 `navmesh/baker.py` 仍保留（渐进式）；切换 Pathfinder 默认实现 + 删除旧路径待用户确认

## 2026-08-12 管线切换 + 废弃约定落地 ✅（AGENTS.md 爪爪法第 5 条）
- **AGENTS.md 新增爪爪法 5（废弃约定）**：不再使用的模块/文件/符号通过加 `_` 前缀标记废弃（文件 `foo.py`→`_foo.py`，类/函数 `Foo`→`_Foo`），**只删引用不删文件**；模块内保留兼容别名（`Foo = _Foo`）供旧脚本/测试
- **navmesh/ 文件级废弃**：`baker.py`→`_baker.py`、`tiles.py`→`_tiles.py`（mv 重命名，无 njit 无缓存坑）；同步更新 `navmesh/__init__.py`（import 改 `_baker`/`_tiles`，docstring 指向新管线）、`_tiles.py` 内部 import、`tests/integration_heightfield.py`、`tests/profile_terrain_load.py`
- **movement/ 符号级废弃（SurfaceMesh 面片寻路）**：
  - `nav.py`：`patch_astar`→`_patch_astar`、`patch_centroid_path`→`_patch_centroid_path` + 模块内别名；docstring 更新
  - `pathfinding.py`：`Pathfinder`→`_Pathfinder`、`funnel`→`_funnel` + 别名；`_Pathfinder` 内部改调 `_funnel`
  - `movement/__init__.py`：只导出新主用（GraphPathfinder/funnel_graph/navgraph_astar）+ `_` 废弃版；**包级旧名 ImportError**（符合废弃语义），模块级别名保留（旧测试不破坏）
- **map.py 导航入口切换**：新增 `nav_graph` property（NavMeshBuilder 构建，懒缓存）；`surface_mesh` 降级为**渲染专用**（surface.py 结构层渲染 + 旧测试仍在用 SurfaceMeshBuilder，不可废弃）
- **验证**：全量 217 passed；端到端 `TerrainMap.generate → nav_graph → GraphPathfinder` 60×60 寻路 0.06ms；包级旧名 ImportError ✓、模块级别名 ✓、navmesh `_` 版可导入 ✓
- ⚠️ 依赖方向提醒：`surface_mesh.py` 与 `navmesh_builder.py` 并存——前者渲染，后者导航；后续若要彻底移除 GreedyMesh 需重写 surface.py 结构层渲染（设计文档 §7 未覆盖渲染替代方案）

## 2026-08-12 三角剖分改走 shapely CDT（🔥 fBm 真实地形导航图变空的修复）
- **触发**：用户问"地图大了三角形数量上不上天" → 实测 fBm 地形暴露严重 bug：100×100 只产 2 个三角形（大区域被跳过），导航图几乎空
- **根因链（两处）**：
  1. `_segment_clear` 只跳过 p 的邻边（按位置索引），**没跳过 q（洞最左点）在已缝合 outer 中的共享端点边** → 多洞**点接触**（相邻山峰对角相碰共享顶点，如 hole0(100,182) 与 hole1 共享）时，桥 V→H 与共享端点边 `_between` 误判穿越 → "找不到可见桥接顶点" → 整个区域被跳过。修复：改用 `_vequal` 跳过所有与 p/q 共享端点的边
  2. 修完后 ear clipping 仍卡死（"多边形自相交/退化"）：12 个洞缝合后轮廓太复杂，点接触洞产生自相交
- **正解**：凹/带洞路径改走 **shapely `constrained_delaunay_triangles`（CDT，约束 Delaunay）**——`from shapely import constrained_delaunay_triangles`（顶层 API，非子模块）；洞边界为约束边 → 三角形天然不覆盖洞、全凸、处理任意拓扑（多洞/点接触/凹洞）。凸无洞保留扇形快速路径
- **实测（TerrainFactory.fbm seed=42, cs=2.0）**：
  | 地图 | 格数 | 三角形 | 面积比 | 构建耗时 |
  |---|---|---|---|---|
  | 64×64 | 4k | 16 | 1.000 | 220ms |
  | 100×100 | 1万 | 80 | 1.000 | 415ms |
  | 200×200 | 4万 | 178 | 1.000 | 2.6s |
  | flat 1000×1000 | 100万 | **2** | - | - |
  → **三角形数不随格数增长**，由区域轮廓复杂度（洞数/凹度）决定；面积比 1.000 = 完全覆盖不覆盖洞
- ⚠️ **遗留性能问题**：构建耗时随格数超线性（200×200=2.6s，400×400 估 >10s）——`extract_walkable_cells` Python 双重循环 + `grow_regions` Python BFS + `region_to_polygon` 每格 box unary_union。加载期一次性，运行时寻路无碍；优化方向：提取向量化（numpy 掩码）、grow 用 BFS 数组、region_to_polygon 用 shapely 多边形批量（后续迭代）
- 手写 ear clipping + 桥接（`_ear_clip`/`_bridge_hole`/`_diagonal*`/`_in_cone*` 等）标注废弃保留（供参考/回退）；全量 217 passed

## 2026-08-12 构建性能优化 ✅（400×400 712ms <1s，原 ~10s+）
- **profile 定位瓶颈**：③④ region_to_polygon（每格 box + unary_union）占 85%
- **① `extract_walkable_cells` 向量化**（Python 双重循环 → numpy）：顶面/悬空按 spans 数组批量；坡度用「每格每 mat 顶面 z 网格」四邻域差分（np.roll + nan 边界 + where+max 替代 nanmax 消 warning）。输出与旧版**逐格一致**（测试断言）
- **② `build_neighbor_map` 优化**：dict 逐格查询 → numpy cell_grid 批量邻居索引（233ms→~50ms）；BFS 保留
- **③ `region_to_polygon` 重写为 Marching Squares 轮廓追踪**（O(周长)）：
  - 🔥 **两次踩坑**：a) 「只对边界格建 box」优化是**错的**——外圈 box 并集是回字形（内部格无 box 填充），unary_union 如实产生伪洞（10×10 平地带洞！）；b) Marching Squares 点接触洞（仅共享角）边界相连 → 自相交环 → shapely CDT 报 `Unable to find shell join index`
  - **修复**：`simplify` 后 `is_valid` 检查 + `poly.buffer(0)`（把接触洞正确拆开，9 洞面积 39832 vs 可行走 39816，比值 1.000）
  - `_mask_to_polygon`：numpy 提取每格暴露边（np.pad 边界）→ 端点邻接表连环 → 面积最大环=外环、其余=洞
- **实测（TerrainFactory.fbm seed=42 cs=2.0）**：
  | 地图 | 格数 | ①提取 | ②区域 | ③④ | 总计 | 三角形 |
  |---|---|---|---|---|---|---|
  | 200×200 | 4万 | 71ms | 58ms | 61ms | **192ms** | 170 |
  | 400×400 | 16万 | 321ms | 244ms | 180ms | **774ms** | 2517 |
  | 800×800 | 64万 | 1432ms | 1240ms | 636ms | 3.5s | 10955 |
  面积比全 1.000；NavMeshBuilder 正式入口 400×400 **712ms**，对角寻路 16ms（2517 节点）
- 遗留：800×800 仍 3.5s（①提取+②区域 Python 循环占大头：extract 的 WalkableCell 列表构造 + BFS 逐 cell）；400×400 已达标 <1s；后续若需更大图再优化（extract 直接产出结构化数组 / grow 并查集）

> 本文件此前用「Phase 2」指代 2.5D 地形系统；按 AGENTS.md 正式编号，那是 **Phase 1（坐标/地形/环境）**。
> AGENTS.md 阶段表已于 2026-08-10 更新：Phase 0、Phase 1 ✅ 已完成（Phase 1 完成于 2026-08-09），Phase 2（实体/移动/检测）未开始。
> AGENTS.md 性能数字已统一到 3.4 的容量上限：2048 装甲 + 16384 弹丸。
> 遗留：`core/logging.py`、`core/rng.py` 尚未实现（Phase 0 清单待补项）。

## 架构（2026-08-10 三层子包重构，对齐 AGENTS.md 4.1）
- `Airis/lulilo/`：无头推演引擎（世界语 lulilo=「摇篮」，呼应《Arkady in Cradle》）
  - 子模块：`{core, coordinates, terrain, entities, movement, detection, navmesh, combat, logistics, simulation}`
- `Airis/render/`：渲染官方实现（moderngl 绘制，游戏使用）：`camera.py` / `gl_widget.py` / `hud/`
- `Airis/ui/`：Qt 交互上层：`main_window.py`（承载 render）
- `Airis/narrative/`：独立叙事层，消费 engine 状态，不进底层
- 依赖方向：engine(无头) ← render/narrative；render ← ui；引擎不反向依赖上层
- ⚠️ **Numba 缓存坑（重要）**：移动/重命名带 `njit(cache=True)` 的模块后，`.nbc` 缓存 pickle 引用旧模块路径 → `ModuleNotFoundError: No module named 'Airis.engine'`。修复 = 删除该包下 `__pycache__/` 让 Numba 重新编译
- 顶层旧目录已全部清理完毕（2026-08-10），`Airis/` 仅剩 `{lulilo, render, ui, narrative}`
- AGENTS.md 已重新统一：进度表（Phase 0/1 ✅）、性能数字（2048/16384）、爪爪法按 -1、格式笔误、4.1 补三层子包架构说明

## Phase 2：2.5D 地形系统 ✅（2026-08-09）
- `Airis/game/map.py`：TerrainConfig(Pydantic) + TerrainMap
  - 高度图：fBm value noise（scipy.ndimage.zoom 上采样）
  - 图层：0=地面/1=桥面/2=隧道，LAYER_MAP + LAYER_ALT + ENTRY_MASK
  - Numba 采样：`bilinear_sample` / `get_terrain_z`（数组作参数，字面量 1/2 表示图层）
  - `is_walkable`：坡度 < max_slope_deg（30°）
  - `to_render_data()`：返回 ground/bridge 网格（顶点含颜色）
- `Airis/utils/coordinates.py`：world↔grid 转换（coordinates 层）
- `Airis/render/camera.py`：OrbitCamera（Z 向上 ENU，拖拽旋转/滚轮缩放，返回 row-major，写 uniform 时 .T）
- `Airis/render/gl_widget.py`：地形网格渲染 + 相机交互（detect_framebuffer 保留）
- `tests/test_terrain.py`：9 用例全过；`tests/smoke_gui.py`：渲染一帧存 terrain_preview.png

## 渲染平滑化（2026-08-10，方案 A）✅
- 问题：`surface.py` 地面用 greedy meshing（每格恒定高度 + 相邻差处垂直台阶）→ 体素感；`Heightfield(ch=1.0)` 量化让相邻格差 p50=0；颜色归一化用 hf 固定 z 范围(-50,200) → 压扁
- 修复：地面改**平滑连续网格** `_extract_ground_smooth_region`（顶点在格点、z=四邻格中心均值=双线性中心插值，顶点共享每格2三角，水密无裂缝）；移除地形垂直侧面 `_add_side_quad`（保留结构悬空侧面 `_add_structure_sides`）
- 颜色归一化改到**实际地面高度范围** `_ground_z_min/_ground_z_max`（`_build_ground_maps` 记录）→ 层次可见
- 顺手修 `tests/smoke_gui.py`：`img.save(str(out))`（QImage.save 不接受 Path）
- 验证：17 passed；ground=40401v/80000t，相邻格点 z 差 p50=0.25（原量化 0.0 平块）；GUI 渲染平滑
- 遗留：振幅/噪声尺度未调（用户选方案 A 未选 B）；真悬崖（大高差）现为陡坡非垂直面
- 追加修复（2026-08-10）：渲染出现「一层层灰色等高线」→ 根因是 `_terrain_color` 的坡度岩色着色（slope>10° 掺灰）+ 分段函数端点不连续。改为**纯高度连续渐变**（共享端点色标插值，深墨绿→草绿→亮绿→浅黄绿），删掉 `_approx_slope_at`；效果：平滑绿渐变、无灰色带
- **Phong 光照（2026-08-10，用户纠正「不要用纯高度」）**：
  - 教训：分段线性 quad 的**几何坡度在格边界突变**，per-vertex 坡度着色会重现「一层层」伪影；`dFdx/dFdy`（几何法线）同理。正解 = **顶点法线用高度场中心差分**（跨格连续）+ fragment 插值做 Phong 光照
  - 顶点格式升级 **(N,6)→(N,9)** [x,y,z,r,g,b,nx,ny,nz]（`to_render_data`、gl_widget VAO "3f 3f 3f"、test_terrain 断言 9 列、所有空数组 (0,9)）
  - gl_widget shader 加方向光：`u_light_dir(0.45,-0.6,0.65)`、`u_light_color(1,0.97,0.9)`、`u_ambient 0.42`；fragment 双面（n.z<0 翻转）；结构顶面法线 +Z、侧面法线朝外
  - 效果：平滑明暗立体感，无等高线色带；17 passed
- **细分消除斜纹（2026-08-10）**：
  - 现象：坡面规律斜纹。根因：每格 quad 4 角点不共面（双线性曲面），被切成 2 三角 → 两三角法线夹角 p90≈10°/max≈20° → 对角线两侧光照梯度不连续
  - 修复：`SUBDIV=2` 每格细分 2×2 子格，顶点高度**双线性插值**自格中心场，法线中心差分（间距 cs/s），颜色/索引全 numpy 向量化（`_terrain_color_vec`）；`_vertex_z` 删除
  - 结果：160801v/320000t，生成 713ms，折痕角度≈减半、频率×2 → 斜纹不可见；17 passed
  - 教训：分段线性网格的三角化折痕（diagonal crease）→ 细分（heightfield 标准做法）是正解，不是调光照
- **渲染去量化（2026-08-10，根治「体素感」）**：
  - 现象：即使渲染平滑，地形仍有层状「体素感」。根因：`SurfaceExtractor._ground_z_map` 来自 Heightfield **1m 量化层**（`_to_z(layer_top)`，全图仅 ~24 个高度层，相邻格差 p50=0/p90=1.0）——渲染几何本身被分层
  - 修复：渲染层改采**连续高度场**。`TerrainMap` 新增 `continuous_ground`（fBm 原始值，隧道格覆盖为 tunnel_alt），`to_render_data` 传给 `SurfaceExtractor(hf, ground_z=...)`；`_build_ground_maps` 优先用连续场，缺省回退量化层；模拟层 Heightfield 保持 ch=1.0 量化不变（供碰撞/逻辑）
  - 结果：渲染高度相邻差 p90 1.0→0.23，z 范围与连续场一致；17 passed；地形平滑自然
  - 教训：渲染精细度 ≠ 模拟量化；渲染可从连续场采样，模拟层保持量化（解耦）

## 地形「尖锐」修复（2026-08-11）✅
- 现象：渲染地形尖锐（折痕/尖峰）。根因：`heightmap.py` 的 `generate_fbm` 用 `scipy.ndimage.zoom(order=1)`（双线性）放大白噪声 lattice → lattice 格点处导数不连续（C0 但非 C1）→ 尖锐折痕。证据：二阶差分双峰分布（p50=0 纯线性区 + p99=0.81 突变）；高频八度 lattice 仅 2×2 格放大数百倍加剧锯齿
- 修复：改为标准 **smoothstep value noise**（3t²-2t³）插值上采样（`_smoothstep` + `_upsample_smoothstep`），lattice 格点处 C1 连续；高频八度 lattice 下限 `min_lattice=4`。纯 numpy 实现，移除 scipy 依赖
- 结果：折痕 d2 max 1.79→0.30（↓6x）、p99 0.81→0.21；坡度 mean 12.7°→13.9°（smoothstep 固有略升，S 形斜坡更立体）；接口不变；95 passed；渲染正常
- ⚠️ `clean_output_dir()` 是 `shutil.rmtree` **递归删除**整个 `tests/output/` → 诊断脚本放该目录会被测试运行清掉；对比图已备份到 `Reference/terrain_fbm_compare.png`
- 验证渲染不能用 view_image「看」像素（模型看不到），用数值统计（二阶差分/梯度分布）

## 关键修复记录
- bilinear_sample 边缘 bug：索引被钳制后 tx/ty 需按钳制后索引重算（曾导致右/下边缘采样为 0）
- 导入路径已统一为 `Airis.` 前缀绝对导入（main.py 移除 sys.path hack）
- **moderngl+PySide6 全黑修复（重要，2026-08-09）**：
  - 症状：QOpenGLWidget 渲染全黑，连背景都读不到；`ctx.clear()`/`ctx.fbo`/`ctx.screen.use()` 都不生效
  - 根因（源码级，venv moderngl/__init__.py）：`ctx.clear()` 用 C 层 `self.mglo.fbo`（create_context 时缓存），Python 层 `ctx.fbo=` 只设属性不同步 C 层/GL 绑定；`ctx.clear()` 会 bind 到缓存 fbo(0) 并清默认缓冲，`vao.render()` 画到当前绑定，于是都落到 Qt 的 FBO(1) 之外 → 全黑
  - 修复（官方方案）：paintGL 里 `fb = ctx.detect_framebuffer(); fb.use()`（use() 同步 Python/C/GL 三层）再 `ctx.clear()` + `vao.render()`
  - 备选 hack：原始 `glBindFramebuffer(0x8D40, defaultFramebufferObject())` + 原始 glClear 也能救，但非库内标准，已弃用
  - 教训：验证渲染必须用像素统计（np 统计非黑/颜色计数），不能凭 view_image「看」——之前被误导多次
  - 诊断脚本：tests/debug_fbo.py（FBO id 对比）、tests/debug_glclear.py（原始 glClear 隔离）、tests/debug_bindfb.py（修复方案，已验证三角形可见）
- **PySide6 事件对象不可保存（2026-08-09）**：mouseMoveEvent 里存 `self._last_mouse = event` 会在下次访问报 `RuntimeError: Internal C++ object (QMouseEvent) already deleted`。只存浮点坐标 `self._last_pos = (x, y)`。
- AGENTS.md 第六节已补「moderngl在QOpenGLWidget中渲染全黑」陷阱条目

## 后续

## HeightField 系统 ✅（2026-08-10）
列式 span 替代旧双层存储：heightfield / surface / baker / tiles 全管线实现。
- `terrain/heightfield.py` + `surface.py`：SoA 存储 + greedy meshing 渲染提取
- `navmesh/baker.py` + `tiles.py`：NavGraph 烘焙 + A* + tile 脏标记/后台重建
- `terrain/map.py` 已集成 Heightfield，to_render_data 兼容旧 gl_widget
- 全量 pytest：17 passed；集成测试全过
- **旧兼容层清理（2026-08-10）**：map.py 删除 heightmap/layer_map/layer_alt/entry_mask 属性 + get_terrain_z/grid_height/is_walkable；删除 layers.py、sampling.py；heightmap.py 删 bilinear_sample（保留 generate_fbm）；__init__.py 移除 LAYER_*/bilinear_sample/get_terrain_z 导出；generate() 用 `_expand_rects` 直接驱动 Heightfield span；continuous_ground/generate_fbm 保留；test_terrain.py 重写为基于 Heightfield 断言
- ✅ **隧道 span 语义问题（已修复 2026-08-11）**：隧道顶板起点抬到 `tun+2`（留 2m 空隙）→ 隧道格 `span_count=2`、`ground_z=-6.0`、空隙可通行。修改：`map.py:115` `add_span(z_tun+2, z_tun+4, mat=MAT_TUNNEL)`；`test_terrain.py` 隧道断言更新（span_count=2 / ground_z=tun / solid_at 空隙 False）；新增 `test_heightfield.py::test_tunnel_gap_not_merged`；全量 pytest 95 passed + GUI 冒烟通过；文档 heightfield-design.md §7.1 已标记修复

- Phase 3（SoA 对象池 + Numba 碰撞）核心优先，不依赖地形
- 渲染管线习惯：moderngl uniform mat4 需列主序（mvp.T.tobytes()）

## Phase 2：实体/移动/检测模块 ✅（2026-08-10）

### 新建模块
- `core/logging.py`：模块级 logger，替代裸 print()（Phase 0 遗留补完）
- `core/rng.py`：RNGManager 流式随机数管理器（多流、可种子、独立 Generator）
- `entities/pools.py`：SoA 对象池（ArmorPool 2048 / BoxPool 4096 / ProjectilePool 16384）
  - spawn/destroy + free_stack，动态扩容仅 Tick 间隙触发
  - PartType / ProjectileType / LayerType 枚举
- `movement/pathfinding.py`：Pathfinder 寻路服务，包装 NavGraph.astar + 漏斗简化
- `movement/integrate.py`：Numba 并行移动积分
  - `integrate_armor_movement`（prange）沿路标点推进 + 朝向更新
  - `integrate_projectile_movement` 含重力
  - `compute_speed` 坡度修正
- `detection/los.py`：DDA 视线检测 + 弹道拦截点
  - `line_of_sight` / `line_of_sight_batch` Numba 实现
  - `terrain_intercept` 弹道终点
  - `LosChecker` Python 封装
  - ⚠️ 修复：`is_solid_at` 接收层索引而非世界 z，los.py 调用时需先 `(z-z_min)/ch` 转换
- `detection/sensors.py`：雷达/红外/ESM 模型
  - 雷达方程（RCS 四次方根）、IR 平方反比、ESM 单程
  - 检测概率 S 曲线（logistic）
  - `evaluate_sensor_detection` 批量 Numba 并行评估

### 已有模块（Phase 1 烘焙，Phase 2 复用）
- `navmesh/baker.py`：NavGraph + Baker（全量/分块烘焙）—— 已实现
- `navmesh/tiles.py`：TileManager（脏标记 + 后台重建 + 双缓冲）—— 已实现

### 性能基准（tests/benchmark_phase2.py）
| 基准 | 耗时 | 状态 |
|------|------|------|
| 装甲移动积分 (2048) | 0.02 ms | ✅ 远超 60fps |
| 抛射物移动积分 (16384) | 0.01 ms | ✅ |
| DDA 视线检测 (20000 对) | 5.10 ms | ✅ |
| 传感器评估 (10×2048) | 0.07 ms | ✅ |

### 回归测试
- 全量 pytest：94 passed（原 14 + 新增 80）
- 旧测试无破坏

### 遗留
- ~~隧道 span 语义问题（heightfield-design.md 7.1）仍未修，留 Phase 3~~ → ✅ 已修复（2026-08-11，见上文 HeightField 系统节）
- 漏斗算法尚未完整实现 SSFA（需 baker 提供 portal edges），当前为简化路径平滑
- `integrate_armor_movement` 的 `terrain_z_func` 为预留接口，Phase 3 接入地形高度采样

## AGENTS.md 订正（2026-08-11）
- 阶段概览：Phase 2 标记「✅ 已完成（2026-08-11）」
- Phase 0 清单：core/logging.py、core/rng.py 勾选（2026-08-10）
- Phase 1 清单：模块路径订正为实际实现（coordinates/grid.py、terrain/heightfield.py 替代旧 layers.py+sampling.py）
- Phase 2 清单：全部勾选 ✅，baker_async 注明实为 navmesh/tiles.py（TileManager）
- 3.2 地形系统：从「双层逻辑海拔」重写为「列式 span 高度场」，指向 docs/heightfield-design.md
- 3.3 地形破坏：HEIGHTMAP 引用 → Heightfield.remove_span()
- 整体技术栈「地形存储」+ 常见陷阱「桥/隧海拔二义性」同步订正为 span 语义
- 常用命令：补 `uv run python -m tests.benchmark_phase2`


- 渲染管线习惯：moderngl uniform mat4 需列主序（mvp.T.tobytes()）


## Phase 3 骨架（2026-08-11，战斗/后勤/模拟循环）
- `combat/weapons.py`：WeaponConfig/AmmoConfig(Pydantic) + WeaponRegistry（内置 M256/2A46M + M829A3/M830A1/3BM42/3BK18）+ WeaponSystem（sigma 散布解算）
- `combat/collision.py`：**统一胶囊体 vs AABB**（AABB 各轴膨胀半径 + Slabs），删除球体分支与 proj_fast 维度；空间哈希 broad phase（Python dict）+ Numba narrow phase
- `combat/damage.py`：ArmorType（**普通 int 常量**，Numba 对 IntEnum 的索引/int() 均报错）+ ARMOR_EFFECTIVENESS + 斜距/跳弹/DeMarre 衰减/后效毁伤内核 + DamageResolver
- `logistics/ammo.py` + `fuel.py`：SoA 状态 + Numba 批量消耗 + SupplyRequest 补给队列
- `simulation/commands.py`：CommandQueue（线程安全）；`state.py`：SimulationState/Snapshot；`loop.py`：SimulationLoop（Tick 循环 + EventBus + sim_paused）
- 测试：tests/test_phase3_skeleton.py 37 用例全过；全量 132 passed / EXIT=0（越界写已修复，见下文「🔥 越界写修复」）
- ⚠️ 设计决策：散布因子是 P(hit) 乘子，注入 sigma 取倒数（技能高→sigma 小，design doc §2.3）；初版方向写反已修
- 碰撞检测设计更新（AGENTS.md 3.4 + phase3-combat-design.md §7.1）：统一「胶囊体 vs AABB」（AABB 各轴膨胀弹丸半径 r + Slabs），不依赖弹种/速度判据，多分辨率 Tick 天然正确

## 🔥 越界写修复（2026-08-11，全量 pytest 崩溃的真根因）
- 现象：全量 pytest（132 测试）断言全绿但进程退出堆损坏崩溃；单文件/纯 Python 不崩；一度误判为 numba 0.66.0 环境 bug
- **根因**：`logistics/ammo.py` + `fuel.py` 的 `consume_*_batch` 循环上界用 `reserve.shape[0]`（池长 2048），而 `amounts/out` 长度仅 2 → i≥2 越界写堆
- 修复：循环上界改 `min(amounts.shape[0], reserve.shape[0])`；文档化长度契约 K≤N
- 验证：numba 0.66.0 原版本全量 132 passed / EXIT=0；conftest os._exit hack 已删除
- 教训：**prange 内核里循环上界永远用「需求/输出」数组长度，别用池全长的 shape[0]**；「全量才崩、单测不崩」往往是不确定行为（越界）在特定内存布局下才暴露

## Phase 3 正式迭代 ✅（2026-08-11，战斗/后勤/模拟循环完成）
- **`combat/engagement.py`**（新增）：FiringControl 开火链管理器，把骨架各孤立模块串成完整开火链
  - fire(shooter,target)：WeaponSystem 散布 sigma → RNG 高斯落点 → ProjectilePool.spawn → AmmoManager 扣 1 发 → 冷却(60/rpm)
  - tick(dt)：按 TICK_BULLET=0.01s 细分子步 → integrate_projectile_movement → CollisionDetector.detect → DamageResolver 毁伤 → box 部位毁伤/摧毁 → EventBus 事件
  - 部位毁伤映射（design doc §5.3）：TURRET 毁→失活 box→无法开火；ENGINE/TRACK 毁→失活；AMMO_RACK cookoff→秒杀
  - `_impact_angle` 按「命中点所在表面」判定法线（world_to_local + 占满度→正面/侧面），正击=0°；修了初版「yaw 反平行 → 180° 误跳弹」bug
  - ⚠️ 教训：**跳弹在「命中面法线」模型下极难自然触发**（能命中的弹必近似垂直面），集成测试用 monkeypatch 注入入射角测跳弹分支
- **`simulation/loop.py`**：多分辨率 Tick（AGENTS.md 3.5）——set_rate(name,interval)/configure_default_rates() + 累加器分频；未配置默认每帧执行（兼容骨架测试）；combat 收累计 dt，内部再按 0.01s 细分
- **`simulation/serialization.py`**（新增）：msgpack + numpy bytes（tobytes+dtype/shape）存档/读档；BattlefieldSave / serialize_battlefield / deserialize_battlefield / save_to_file / load_from_file；free_stack 由 active 反推（spawn 不复用活跃索引）
- **`combat/collision.py` 全 Numba 重构（关键性能）**：Python dict broad phase（实测 186ms）→ 密集网格索引（cell_count/cell_start/cell_armor 两遍构建）+ detect_projectiles_batch（prange 并行查相邻 9 格）；16384 弹丸×2048 装甲 **186ms→1.26ms**（AGENTS.md 目标 <3ms 达标）；⚠️ 必须 `from numba import prange`
- **`entities/pools.py` BoxPool**：加 `local` 数组（权威局部偏移）+ `_sync_boxes_kernel` Numba 并行同步；修复「pos 存局部偏移被重复 sync 错误旋转」bug（原实现第二次 sync 起把世界坐标当局部再转一次）
- **`combat/damage.py`**：未击穿结构损伤封顶 `min(15, 2+0.02*|margin|)`（原 `5+0.5*|margin|` 在 margin 大时未击穿也重创 T-90，违反 design doc §4.3「health 微降」语义）
- **`tests/test_phase3_integration.py`**：26 用例（开火基础约束/交战命中毁伤/部位毁伤物理命中/多分辨率 Tick/序列化 round-trip 含交战后与文件往返）
- **`tests/benchmark_collision.py`**（新增）：碰撞 1.26ms ✅ / 开火链 1.40ms(tick 0.01s) ✅ / 命中延迟 33 tick=0.33s ✅；运行 `uv run python -m tests.benchmark_collision`
- 全量 pytest：**158 passed / EXIT=0**（含骨架 37 + 集成 26 + 既有模块）
- ⚠️ 教训：**EventBus 定义了 `__len__`，空总线是 falsy → `events or EventBus()` 会丢弃传入的空 events**（测试一度 KeyError 'fire'）；必须 `events if events is not None else EventBus()`
- ⚠️ 教训：`FireSolution` 无 pen_mm_rha（solve 只返回 sigma+初速），弹药穿深须从 registry.get_ammo 再取
- 遗留：装甲 vs 装甲 Sort & Sweep 编队推开未做；漏斗算法 SSFA 未完成；proj 弹药元数据仍是 dict（可下沉 SoA）

## Phase 3 遗留项收尾（2026-08-11）
- **弹药体系扩展文档 + 代码预留**（docs/ammunition-expansion-design.md）：Lanz-Odermatt/HE_FRAG/PELE/THERMOBARIC/步兵弹药/防空可编程弹（AHEAD/3P/VT）全部规划为「设计预留，未实现」；weapons.py 枚举/模型加预留注释；编号 6~34 预留给 AmmoType、5~10 预留给 WeaponCategory。158 全绿
- **弹丸元数据下沉 SoA ✅**：ProjectilePool 新增 weapon/ammo(int32 注册表索引)/age(float32) 数组；WeaponRegistry 新增 weapon_index/ammo_index/weapon_at/ammo_at；FiringControl 删除 _proj_meta/_proj_age dict，改用数组（fire 写、命中反查、_recycle_stale 用 age）；序列化同步打包/恢复新数组
- **编队推开（空间哈希）✅**：用户定空间哈希（弃 Sort&Sweep）。FormationResolver + separate_armors_hash Numba 内核（collision.py）；复用 build_grid_index；去重=每对只在候选 j 中心格处理一次（j<=i 还不够）；接入 FiringControl.tick
- 🔥 **build_grid_index 既有 bug 修复（重要）**：第二遍填充把 cell_count 当游标但**未清零** → pos 从 start+N 错位写、cell_count 污染翻倍、同格装甲重复登记（如 [0,0,0,1]）。弹丸碰撞每格≤1 碰巧无害；编队推开多辆重叠同一格 → 重复推开累积。修复=第二遍前清零 cell_count 复用为游标。教训：**共享内核的「碰巧正确」行为在密集场景会暴露**
- 新增 tests/test_formation.py（10 用例）；全量 168 passed；碰撞基准 1.38ms / 开火链 2.26ms（<3ms ✅）
- **漏斗 SSFA 完整实现 ✅**：
  - NavGraph 增 portal_edges（poly_id → [(p1,p2)] 共享边）；baker._build_adjacency 改用 shapely intersection（LineString 才算 portal）；tiles._merge_graphs 同步合并 portal（返回三元组 + _copy_portals）
  - NavGraph.astar_poly_ids（返回多边形 ID 序列）；funnel 完整 SSFA（_collect_portals 按形心差叉积定左右端点）
  - 🔥 **死循环教训（重要）**：SSFA「闭合后不前进重试」分支会死循环（left==right==apex 全等 → triarea 全 0 → 永远闭合 + 无限 append → 内存爆炸 2G）。修复=每条 portal 只消费一次（for/单调推进）+ closed_left/right 标记 + max_iter 保险丝 + 每轮必 portal_idx+=1
  - 修复 _simplify_path 既有 bug：`tuple(float,float)` 少括号 → TypeError（SSFA 测试暴露）
  - **设计约束（重要）**：当前 baker 只烘焙 ground 层、**不做凸分解** → 连通域=多边形（通常 1 个）→ portal_edges 实际几乎为空 → SSFA 退化为形心路径简化。AGENTS.md 3.3 的 50~200 多边形 + portal 漏斗需未来加凸分解才充分发挥。SSFA 算法本身已用手动构造走廊 NavGraph 测试覆盖
  - 新增 tests/test_pathfinding.py（10 用例）；全量 178 passed
- 下一项：全量测试 + 基准验证收尾

## 表面网格统一架构重构（2026-08-11，docs/surface-mesh-unification.md 实施）
- **Step 1** `heightfield.py::add_span`：合并条件加 `s[3] == merged[-1][3]`（防地面 BRIDGE 贴地段吞 TERRAIN）
- **Step 2** `terrain/surface_mesh.py`（新建）：GreedyMesh 核心
  - `SurfacePatch`（patch_id/vertices/z/mat/walkable/area/centroid）+ `SurfaceMesh`（patches+adjacency+portal_edges+walkable_patches）
  - `SurfaceMeshBuilder`：_scan_groups（按 layer_z,mat 分组）→ _build_patches（2D GreedyMesh 合并）→ _filter_walkable → 邻接
  - ⚠️ **float32 精度坑（重要）**：numpy 2.x NEP 50 下 `np.float32(x) - 1e-6` 结果仍 float32，1e-6 < ULP(~7.6e-6) 被吃掉 → 边界格多算一列 → 误判悬空。修复 = `_is_hanging` 里先 `float()` 提升 float64
  - ⚠️ **邻接必须扫描线**：地面每格一面片（80×80=6400），O(n²) 全对检测超时。改为扫描线（边按 x/y 分组 + 区间重叠），O(n log n)
  - `_filter_walkable`：人工结构（BRIDGE/BUILDING/TUNNEL）直接可走（本身可站立）；TERRAIN 才查 _is_hanging
- **Step 3** `movement/nav.py`（新建）：`patch_astar`（SurfaceMesh 面片 A*）+ `patch_centroid_path`；边代价=形心距+z落差×3
- **Step 4** `surface.py`：结构层改从 SurfaceMesh 面片渲染（替代逐格四边形）；`struct_offset` 记录结构顶点起始（供上层按顶点分离 ground/结构）
- **Step 5** `pathfinding.py`：NavGraph → SurfaceMesh；funnel 复用（接口兼容 portal_edges/centroid）
- **Step 6** 去 layer：删 `LayerType` / `armor.layer` / spawn(layer=) / 序列化 layer；entities/__init__、serialization 同步
- **Step 7** `navmesh/` 废弃标记（2026-08-11）：类加 `_` 前缀（`_Baker/_NavGraph/_Tile/_TileManager`），`__init__.py` 只导出 `_` 前缀版（旧名从包级导入会 ImportError）；模块内保留兼容别名（`Baker = _Baker` 等，旧调试脚本不破坏）；baker/tiles 文件未删，需用户确认
- **桥放低（用户纠正，2026-08-11）**：弃「桥头连接 hack」，改 `TerrainConfig.bridge_alt(固定14)` → `bridge_lift(相对抬升2.0)`；map.py 桥面顶 = 桥区地面 max + lift → 桥头落差恰 2m=max_climb → 自然连通
- **to_render_data 分离**：弃 bridge_alt 高度阈值（桥面贴近地面后不可靠），改按 `extractor.struct_offset` 顶点偏移分离 ground/结构
- 验证：80×80 全 walkable；桥面合并成 1 面片(328m²)；桥头双向可达；109 相关测试通过
- 遗留：`_find_shared_edge`/`_point_in_polygon` 通用工具保留（未来凹多边形扩展）；`surface_mesh._enrich_portals` 仅去重；删除 baker/tiles 待用户确认

## 🔥 桥侧面 Z-fighting 修复（2026-08-11，用户远程诊断驱动）
- 现象：桥侧面「竖直明暗条纹」，且地面无关（地形不渲染也有）、亮面暗条纹/暗面亮条纹
- **根因（源码级）**：`surface.py` 的 `_add_vertical_edge_sides` / `_add_horizontal_edge_sides` 对**同一条桥边生成两个相距仅 2cm、法线相反的平行条带**（nx=-1 与 nx=+1，一个朝光一个背光）→ 远处 2cm 压不过深度精度 → 像素在亮/暗面间跳 → 明暗条纹。**不是裂缝、不是 dFdx/dFdy**（本项目着色器根本没有）
- **修复 1**：每条边只生成**单个朝外条带**（用 `patch.centroid` 判朝外方向 `out_nx = -1.0 if cx > p1[0] else 1.0`）；朝内条带删除（桥内部不渲染，后续精细建模）
- **修复 2**：着色器法线翻转 `if (n.z<0) n=-n` → **`if (!gl_FrontFacing) n=-n`**（前者只翻顶面，侧面 nx/ny 背面需要按三角形朝向翻，双面才正确受光）。gl_widget.py + diag_viewer.py 同步
- **修复 3（顶面/侧面空隙）**：旧实现侧面顶边 `z_top-0.05`（下沉5cm）+ 缩进桥内1cm → 顶面是薄片遮不住 → 5cm 竖向空隙。改为**顶边对齐 z_top、去掉缩进**，侧面与顶面边缘共边 → 水密实心柱
- 验证：竖直面 z 范围底边贴地面(-8.44)/顶边=15.0=顶面（差 0.0m）；X面 x 间距无 2cm 成对

## 🔥 远处 Z-fighting（深度精度）修复（2026-08-11）
- 现象：远景地形轻微抖动（非网格重叠，深度缓冲远处精度不足）
- **根因**：24 位定点深度 + near=0.5/far=20000 → 1500m 处分辨率仅 **27cm**（dd/dz≈2.2e-7/m，LSB=6e-8）
- **修复**：`camera.py` near 0.5→3.0、far 20000→12000；`main.py`+`diag_viewer.py` `setDepthBufferSize(32)`（Qt 实测给 32）。800m 处精度 → mm 级
- ⚠️ **far 必须 > zoom 上限 10000**：far=5000 时相机拉远 → 整场被 far 裁剪 → 地图消失（用户反馈「放远地图不渲染了」）。far=12000 解决
- **景深（DoF）友好性决策**：保持标准透视投影（还原线性深度一行 `linZ=near*far/(far - z*(far-near))`）。reverse-z 需浮点深度 D32F（近处精度反降）但对 DoF 只多 `1-z` 变换；**对数 z 最坑**（gl_FragDepth 禁 early-z，未来 2048 单位不利，且非线性深度 DoF 还原麻烦）——不优先
- **诊断方法（用户认可）**：亮红背景 #FF0000 + 纯白网格 + 环境光压到 0.15 → 裂缝透红 / 重叠呈明暗条纹；多角度连抓 3 帧比帧间抖动
- 交互诊断脚本：`tests/diag_viewer.py`（类似 main.py，1/2=只桥/全量、3/4=生产色/白、R=红背景、F=打印相机参数）

## 📌 决策：未来上 reverse-z（用户拍板 2026-08-11，暂不实施）
- 现状：标准透视投影 + near=3/far=12000 + 32 位深度（`Depth bits = 32`，AMD 780M 实测）已满足；先作为过渡
- 方向：**reverse-z**（反向 Z）为最终解 —— 浮点深度 D32F 下远处精度最优，且不损 early-z、对 DoF 友好（比对数 z 强）
- 何时上：远处浮点深度精度成为瓶颈（如大场景 far 需求）、或需要极致远景时
- 未来实施清单：
  1. 确认深度缓冲为浮点（D32F）：查询 `GL_DEPTH_BUFFER_FLOAT` 或读深度值判断（Windows/AMD 32 位深度通常即 D32F）
  2. `glDepthFunc(GL_GEQUAL)` 替代 GL_LESS
  3. `glClearDepth(0.0)`（moderngl：`ctx.clear(depth=0.0)`）
  4. 投影：near/far 对调传 `glm.perspective(fov, aspect, far_clip, near_clip)` + `glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)`（OpenGL 4.5+）
  5. 景深还原线性深度公式随之改（z 反向，符号注意）
- 改动面小：仅 `camera.py`（投影）+ `gl_widget.py`（depth func/clear），当前无阴影/多 pass 依赖，风险低


## 半球环境光升级（2026-08-11）
- gl_widget.py 光照模型：常量灰白环境光 `u_ambient(float)` → **半球环境光**（sky/ground gradient）
  - 物理背景（用户讨论）：环境光颜色应由「天空漫射 + 周围表面 bounce」决定，常量灰白是强近似；半球光是性价比最高的近似（IBL/GI 过重）
  - shader 改动：`u_ambient` 拆为 `u_sky_color`/`u_ground_color`(vec3) + `u_ambient_strength`(float，语义同旧 0.42)
  - 公式：`up=clamp(n.y*0.5+0.5,0,1)`；`ambient=mix(u_ground_color,u_sky_color,up)`；`color=v_color*(strength*ambient+(1-strength)*diff*light)`
  - 默认值：晴天淡蓝天光(0.55,0.65,0.85) + 泥土/植被地光(0.30,0.28,0.22)，强度 0.42
  - ⚠️ GLSL 标量→向量广播：float 参与 vec 运算自动广播成同值向量；旧 `u_ambient` 是标量所以环境光其实一直灰白、漫反射才带光源色温（用户第一次提问点破）
  - 兼容性：强度旋钮语义不变；诊断脚本 diag_viewer/diag_zfight_red 仍用旧 `u_ambient`（独立工具未动）
  - 验证：smoke_gui 通过，terrain_preview.png 正常（顶面偏天色、暗部偏地色）
  - 遗留：Phase 5 主题切换时 sky/ground 色可参数化（构成主义/日系/战术全息各配一组）
  - 📌 后续升级路线图（A 地光高度衰减 / B 天穹渐变 / C 地形色采样 / D IBL）已留档 `render-lighting-upgrade.md`，用户拍板「留档不实施」；核心结论：环境光与摄像机无关（朗伯各向同性），依赖仅法线 n + 位置


## 2026-08-12 TerrainMap 纯化 + 测试地形工厂 ✅（217 passed）
- **背景**：战役地形为人工编辑（不走 fBm），`TerrainMap.generate()` 纯属测试用途（用户拍板）
- **map.py 纯化**：`TerrainMap.__init__(heightfield, continuous_ground=None)` 注入式容器；删除 `TerrainConfig` / `generate()` / `_expand_rects` / `_z_min` / `_z_max` / `config` 属性；保留 surface_mesh / nav_graph / to_render_data
- **`tests/terrain_factory.py`（新建）**：`TerrainConfig`（原 map.py 迁入）+ `TerrainFactory`（`flat` 确定性平地 / `fbm` 原 generate 全逻辑 / `default` 默认 3 桥 / `default_config`）
- **gl_widget.py**：默认地形改 `_default_terrain()`（纯平坦 200×200，生产不依赖程序化生成）；相机 target/distance 改从 `heightfield.width / hf.cs` 推（不再读 terrain.config）
- **main.py**：相机对准地图中心（原对准第一座演示桥 260,204）
- **迁移 11 处**：test_terrain.py（`terrain.config`→模块级 `CFG`）、integration_heightfield、debug_projection、diag_bridge_patch/sides/tris、diag_viewer、diag_zfight_red、preview_bridge、render_bridge_diag、profile_terrain_load
- **关键点**：`TerrainMap(hf)` 不传 continuous_ground 时 `to_render_data` 走 SurfaceExtractor 量化层 fallback（人工编辑地形可用）；`TerrainConfig` 已从生产包导出移除（含 `terrain/__init__.py`）
- ⚠️ 约定：UI/测试场景一律走生产侧 `terrain/factory.py`；战役加载侧将来 `TerrainMap(hf)` 一条路

## 2026-08-12 工厂回生产侧 + 命名 TerrainFactoryConfig ✅（217 passed）
- 用户拍板：项目仍在测试阶段，UI 设计需可选测试场景 → 生产侧保留几个 factory
- **`terrain/factory.py`（新建）**：主名 `TerrainFactoryConfig`（战役人工编辑后无独立「地形配置」语义）+ `TerrainFactory`（flat/fbm/default/default_config）
- **`terrain/__init__.py`**：导出 TerrainFactory / TerrainFactoryConfig
- **`tests/terrain_factory.py` → re-export 层**：仅 `from Airis.lulilo.terrain.factory import ...`；测试脚本 import 路径不变（兼容）；测试专用复杂地形（悬崖/迷宫）将来在此追加
- 测试脚本 TerrainConfig → TerrainFactoryConfig（test_terrain / integration_heightfield / profile_terrain_load / diag_bridge_patch）
- 验证：217 passed；生产包/模块/re-export 三路导入一致；default/flat/fbm 场景可用
- **去别名（用户拍板）**：删除 `TerrainConfig = TerrainFactoryConfig` 别名（factory.py / __init__.py / tests re-export 全清），`TerrainConfig` 全项目零残留；217 passed

---

## 2026-08-16 Phase 7 空中部队全量实施完成（已归档，见主 dev-progress 2026-08-17 前的记录）

# 开发进度

## 2026-08-16 Phase 7 空中部队全量实施完成（docs/phase7-air-pipeline.md §十四 步骤①~⑪）

> 空中部队 = 6b 制导管线 + 6c 编队 AI 框架 + EmitterPool 电磁世界的「泛化消费者」，
> 净新增五块：AircraftPool / 气动飞控内核 / 空中 AI / 坠毁链 / 航空弹药。全量 573+ 测试通过。

### 新增文件
- `movement/aerodynamics.py`：点质量 3-DOF + 「解 α」升力/阻力闭环（大气指数衰减、过载/失速/坠毁分支、TAKOFF 滑跑豁免、PARKED 静止）
- `movement/flight_control.py`：三通道定高定航（高度/油门 PI/航路点转向）+ 机动原语（lead pursuit / kinematic defeat / 俯冲-拉起）+ `build_air_movement_hook`
- `simulation/air_ai.py`：`gather_air_contacts`（机载雷达搜索，复用 6b 检测层）+ `AirFormationAI`（意图 FSM + 贪心两轮分配 + 队形）+ `AirUnitAI`（OODA 空战/对地/返航）+ `AirbaseConfig/AirfieldOps`（起降闭环）+ `AirDefenseController`（防空 role=3 激活）+ `score_air_targets` 内核
- `combat/aircraft.py`：`AircraftConfig`（机型静态参数，YAML→Pydantic）
- 测试：`test_phase7_pool/aero/flight/sensor/formation/air_ai/aam/crash/takeoff/sead/integration.py`（10 文件）+ `tests/benchmark_phase7.py`

### 关键接缝改动（追加不重排，旧存档/旧测试兼容）
- `entities/pools.py`：`AircraftPool`（128）/`AirFormationPool`（32）+ `leader_idx`；`PartType.AIRCRAFT=5`；`BoxPool.parent_kind`（0=装甲 1=飞机）；`EmitterPool.EMITTER_PARENT_AIRCRAFT=3` + `activate_aircraft_radar/ir`；`ProjectilePool.guid_target_vel/guid_target_kind/owner_kind`
- `movement/integrate.py`：`integrate_guided_projectile_movement` 加 `guid_target_vel/guid_target_kind`（AAM 前置量 + 空中目标重力补偿；地面目标行为逐位不变）
- `combat/guidance.py`：`control_glide_bomb`（末尾注册，索引 4）
- `combat/engagement.py`：`fire_air()`（飞机开火）+ `_seeker_probe` 飞机目标分支（`_find_nearest_air_target`，排除射手按 `owner_kind`）+ 近炸引信 `_prox_fuse_check` + 飞机毁伤 `_resolve_aircraft_hit`/坠毁链 `_begin_aircraft_crash`/`_check_aircraft_crashes` + 自伤不销毁修复
- `combat/collision.py`：`detect_proj_aircraft_batch`（弹丸 vs 飞机箱全对全）
- `combat/damage_submission.py`：`resolve_aircraft_area_damage`（飞机爆轰结算，装甲路径零侵入）
- `detection/sensors.py`：`RCS_FIGHTER/BOMBER`
- `simulation/ai.py`：`RWRDetector` 泛化到飞机（`rwr_state` 0~3 分档，`_emitter_side` 飞机分支）
- `simulation/loop.py`：调度链插入 `air_movement`（0.1s）/`air_ai`（1s）——「飞→探测→AI→开火→弹丸推进」因果序
- `simulation/serialization.py`：飞机/编队/弹丸新字段序列化往返

### 修复的跨池索引碰撞陷阱（Phase 7 防空特有）
- SAM 射手是装甲（owner 索引在 armor 池），目标飞机索引可能同值 → `_find_nearest_air_target`/`_prox_fuse_check` 排除射手误排除目标。修复 = `ProjectilePool.owner_kind`（0=装甲 1=飞机）区分射手空间

### 物理调参记录
- AAM/SAM 搜索半径 `SEEK_RADIUS_AIR_M=12000`（防空导弹地面雷达引导，末段覆盖全射程）
- AAM 近炸 `prox_fuse_radius=25`（AIM-120 破片散布）；近炸 SPHERE 半径 `max(blast_radius, trigger)`（破片覆盖 ≥ 引信圈）
- 空中目标重力补偿（`guid_target_kind==1` → accel[2]+=g）：横切/悬停目标才能命中（40g 转弯半径 ~4.7km 物理限制，横切仍低命中——真实兵棋亦然）

### 性能基准（tests/benchmark_phase7.py，全部达标 ✅）
- 气动积分 128 架 × 0.1s tick：0.014ms（目标 < 0.05ms）
- 近炸判定 128 AAM × 128 飞机：0.06ms（目标 < 0.1ms）
- 完整空战帧（128 架 + 战斗推进 0.1s）：2.7ms（目标 < 5ms）

### 遗留（Phase 8 数据链，接口已预留）
- 接触源换孪生数据：`AirFormationAI` 的 `gather_air_contacts` 注入点（`contact_provider`）
- A 导 B 射 / 磨盘战术：`score_air_targets` + 编队意图已留扩展
- 起降为闭环简化（跑道 = 配置坐标，不建模跑道物理）；编队 AI 的 RTB→分组降落接线待场景装配



---

## 归档批次（2026-08-18 之前的记录，由 archive_dev_progress.py 移出）



