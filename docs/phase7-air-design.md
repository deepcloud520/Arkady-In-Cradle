# Phase 7 空中部队设计 —— 统一管线体系作战

> 状态：**设计定稿（2026-08-14，规划）**
> 定位：Phase 7 是 `phase6-guidance-pipeline.md`（6b，已实施 ✅）与 `phase6-ai-design.md`（6c）的**承载体**——
> 空中部队是 Phase 6 制导/电子战/AI 基础设施的**最大消费者**，本阶段几乎不新建子系统，
> 核心工作是把现有管线「向上泛化」到航空域。
>
> 读前必读：`phase6-guidance-design.md`（Numba 回调制导）、`phase6-ai-design.md`（编队 AI 框架）、
> `数据链与战争迷雾设计.md`（Phase 8 接口预留）。本文复用其中的 SoA 约定、`EmitterPool` 辐射源模型、
> `detection` 检测层、`FiringControl` 开火链与 `DamageSubmissionPool` 毁伤提交，不重复论证。

---

## 〇、总纲：空中部队是「现有管线的泛化」，不是「新引擎」

### 0.1 核心判断（为什么 Phase 7 工作量可控）

Phase 6 已经为航空域铺好了几乎全部接缝：

| Phase 6 已交付 | Phase 7 直接复用方式 |
|---|---|
| `TICK_AIR = 0.1` 分频常量 | 空中移动/AI hook 直接配 `set_rate("air_*", TICK_AIR)` |
| `ProjectilePool` 制导数组 + `MissileControlRegistry`（PN/PP + 推力曲线） | **AAM 就是制导弹丸**，零制导层改动 |
| `SeekerType`（ACTIVE_RADAR / IR_IMAGING / SEMI_ACTIVE_LASER / PASSIVE_RF） | 雷达 AAM=ACTIVE_RADAR、红外 AAM=IR_IMAGING、激光 AGM=SEMI_ACTIVE_LASER（SAL 已实施）、反辐射=PASSIVE_RF |
| `EmitterPool`（RF/IR/LASER + 充能-残热 + 战术关机） | 机载雷达=kind=0 RF、尾焰=kind=3 IR；RWR 检测=EsmEmitter 单程 |
| `detection` 层（RCS 四次方根 × ECM / ESM 单程 / 被动 IR） | 机载雷达搜索 = 平台级 RCS 检测（泛化 seeker 检测公式，§四.2） |
| `FormationAI` 框架（意图 FSM + 接触表 + 贪心分配 + 单位执行） | 空中编队 AI 同构复刻（§五），决策主体仍是编队 |
| `FiringControl.fire()`（含 `aim_at`）+ `DamageSubmissionPool` | 空对空/空对地开火与毁伤全走既有链 |
| `ArmorPool.role=3 防空`、`role=5 预留空中`（6c 已建枚举） | 地面防空激活（§八）、空中 role 落地 |
| `AmmoConfig.ins_drift_m_s`（注释「Phase 7 启用」） | 惯性导航漂移字段开灯 |
| `guid_comm` / `guid_target_pos`（坐标制导，无索引） | Phase 8 A 导 B 射的现成接口（§十） |

**结论：Phase 7 的净新增 = ① `AircraftPool` ② 气动/飞控 Numba 内核 ③ 空中 AI 两个类 ④ 航空弹药 YAML ⑤ 坠毁链。** 其余全是接线。

### 0.2 范围界定

**做**：
- 一般气动力学（升力/阻力随迎角变化，定高定航飞控）
- 空中单位 AI：定高定航巡航 → 空战自主「发现-攻击-规避-返航」
- 空中编队 AI：编队飞行、分组降落、任务分配（空优/对地/压制）
- 航空弹药：红外 AAM、雷达 AAM（主动/半主动）、激光 AGM、反辐射导弹、自由落体/滑翔炸弹（**全部复用制导管线**）
- 地面防空激活（role=3 + 地空导弹），RWR 告警 + 规避 + 干扰弹（接收 6c 步骤⑨ 的未实施项）
- 被击落/坠毁链、集成测试、性能测试

**不做（延后/Phase 8）**：
- 战争迷雾认知孪生、数据链 A 导 B 射、超视距编队战术（磨盘战术）——Phase 8
- 航母/跑道物理（降落简化，§七.3）；CAS 任务编排（6c 已定性：CAS 是任务类型，执行者是本阶段空中单位）
- UI 展示（Phase 5b 面板）

### 0.3 核心原则（延续 6c 拍板）

1. **万物皆单位**：飞机是单位（AircraftPool），导弹是弹丸（ProjectilePool），巡飞弹/CAS 是任务不建实体。
2. **编队 AI 决策，单位 AI 执行**：空中编队 AI 持有意图 FSM 与任务分配；单位 AI 只飞/只打。
3. **AI 只决策不执行**：一切行动走 `FiringControl.fire()` / 飞控输入 / 命令队列，不直改 SoA。
4. **热路径只做向量化**：气动积分 + 飞控 = `@njit` 批量内核；空战 OODA 是 Python 决策层（≤128 架 × 简单分支）。
5. **感知驱动非上帝视角**：机载雷达/IRST/RWR 走 `EmitterPool` + `detection` 层输出，Phase 8 换孪生数据。

---

## 一、数据模型：`AircraftPool`（新建 SoA 池）

### 1.1 为什么新建池而不是复用 `ArmorPool`

| 维度 | ArmorPool（地面） | 飞机 |
|---|---|---|
| 碰撞体 | BoxPool 2~6 部位碰撞箱 | 单碰撞球/胶囊（BoxPool 挂 1 箱，§1.4） |
| 垂直运动 | span 高度场贴地 + 悬挂 | 自由 6DoF（高度是状态不是约束） |
| 导航 | NavGraph 路标点 | 空间航路点（无图） |
| 毁伤语义 | 部位血量（HULL 归零=摧毁） | 单血量 + 结构损伤系数（气动退化） |

复用会造成大半字段空转且移动积分语义打架。**新建 `AircraftPool`**，但接口风格完全对齐（`spawn/destroy/active_indices/_expand/free_stack`，扩容非零默认补填——沿用 6c 修的坑）。

### 1.2 SoA 布局（`entities/pools.py` 追加）

```python
MAX_AIRCRAFT = 128   # 空优/对地/侦察编队合计（3~5 个中队规模）

class AircraftRole(IntEnum):
    FIGHTER = 0        # 空优（AAM 主战）
    ATTACKER = 1       # 攻击机（AGM/炸弹对地）
    SEAD = 2           # 防空压制（反辐射）
    RECON = 3          # 侦察（预留）

class AircraftState(IntEnum):
    TAKEOFF = 0        # 起飞（滑跑→爬升）
    CRUISE = 1         # 定高定航巡航
    ENGAGE = 2         # 交战机动
    RTB = 3            # 返航
    LANDING = 4        # 降落（进近→着陆滑跑）
    CRASHING = 5       # 失控坠落（被击落）
    PARKED = 6         # 驻停（降落完成 / 基地待命）

class AircraftPool:
    # ---- 运动状态 ----
    pos        = np.zeros((MAX_AIRCRAFT, 3), np.float32)   # 世界坐标 (x,y,z)
    vel        = np.zeros((MAX_AIRCRAFT, 3), np.float32)   # 速度向量
    heading    = np.zeros(MAX_AIRCRAFT, np.float32)        # 航向角（yaw）
    pitch      = np.zeros(MAX_AIRCRAFT, np.float32)        # 俯仰角
    bank       = np.zeros(MAX_AIRCRAFT, np.float32)        # 滚转角
    # ---- 状态/身份 ----
    active     = np.zeros(MAX_AIRCRAFT, np.bool_)
    state      = np.zeros(MAX_AIRCRAFT, np.uint8)          # AircraftState
    team       = np.zeros(MAX_AIRCRAFT, np.int8)
    role       = np.zeros(MAX_AIRCRAFT, np.uint8)          # AircraftRole
    # ---- AI 字段（对齐 ArmorPool 6c 约定）----
    order      = np.zeros(MAX_AIRCRAFT, np.uint8)          # AirUnitOrder（§六）
    target_idx = np.zeros(MAX_AIRCRAFT, np.int32)          # -1=无；指向 AircraftPool 或 ArmorPool
    target_kind= np.zeros(MAX_AIRCRAFT, np.int8)           # 0=空中 1=地面
    formation_id = np.zeros(MAX_AIRCRAFT, np.int32)        # -1=无编队
    comm       = np.zeros(MAX_AIRCRAFT, np.float32)        # 通信质量 0~1
    contact_age= np.zeros(MAX_AIRCRAFT, np.float32)        # 最近接触时间戳
    # ---- 任务载荷 ----
    wp         = np.zeros((MAX_AIRCRAFT, 8, 3), np.float32)  # 空间航路点（含高度）
    wp_count   = np.zeros(MAX_AIRCRAFT, np.uint8)
    wp_idx     = np.zeros(MAX_AIRCRAFT, np.uint8)
    fuel       = np.zeros(MAX_AIRCRAFT, np.float32)        # 燃油（归一化 0~1）
    fuel_burn  = np.zeros(MAX_AIRCRAFT, np.float32)        # 巡航单位时间消耗（per-类型）
    bingo      = np.zeros(MAX_AIRCRAFT, np.float32)        # 强制返航阈值（0.3 典型）
    aam_left   = np.zeros(MAX_AIRCRAFT, np.uint8)          # 剩余 AAM 数
    agm_left   = np.zeros(MAX_AIRCRAFT, np.uint8)          # 剩余 AGM/炸弹数
    # ---- 空战状态 ----
    cruise_alt = np.zeros(MAX_AIRCRAFT, np.float32)        # 巡航定高（m）
    cruise_spd = np.zeros(MAX_AIRCRAFT, np.float32)        # 巡航定速（m/s）
    rwr_state  = np.zeros(MAX_AIRCRAFT, np.uint8)          # 0=静默 1=搜索 2=跟踪 3=锁定（§九.1）
    chaff_left = np.zeros(MAX_AIRCRAFT, np.uint8)          # 箔条剩余
    flare_left = np.zeros(MAX_AIRCRAFT, np.uint8)          # 干扰弹剩余
    # ---- 物理/毁伤 ----
    health     = np.zeros(MAX_AIRCRAFT, np.float32)        # 单血量（100 归一）
    dmg_factor = np.zeros(MAX_AIRCRAFT, np.float32)        # 结构损伤系数 0~1（升力/机动退化）
    rcs        = np.zeros(MAX_AIRCRAFT, np.float32)        # 雷达反射截面 m²
    ecm        = np.zeros(MAX_AIRCRAFT, np.float32)        # 电子对抗能力 0~1
```

> 字段规模 ≈ 36 数组 × 128 元素 ≈ 数十 KB 级，内存可忽略。扩容沿用 `_resize` + 非零默认补填模式。

### 1.3 机型数据（YAML + Pydantic，`combat/aircraft.py` 新建）

```python
class AircraftConfig(BaseModel):
    """机型静态参数（YAML 加载，仿 stochastic-warfare 数据组织）"""
    model_config = ConfigDict(extra="forbid")
    name: str
    role: AircraftRole
    # 气动（§二）
    wing_area_m2: float          # 机翼面积
    cl_max: float                # 最大升力系数
    cd0: float                   # 零升阻力系数
    k_ind: float                 # 诱导阻力因子
    max_alpha_deg: float         # 最大迎角
    # 飞行包线
    v_min_m_s: float             # 失速速度
    v_max_m_s: float             # 最大平飞速度
    max_g: float                 # 过载限制
    climb_rate_m_s: float        # 最大爬升率
    turn_rate_max_deg_s: float   # 最大持续转弯率
    # 载荷/燃油
    aam_capacity: int
    agm_capacity: int
    fuel_capacity_s: float       # 满油续航（秒）
    bingo_fuel: float            # 强制返航阈值 0~1
    chaff_capacity: int
    flare_capacity: int
    # 传感器（§四）
    radar_power_db: float        # 机载雷达功率（EmitterPool kind=0）
    radar_freq_ghz: float
    rwr_sensitivity: float       # RWR 检测灵敏度（复用 ESM 阈值）
    # 毁伤
    health_max: float = 100.0
```

### 1.4 碰撞体：BoxPool 挂单箱

飞机 spawn 时在 `BoxPool` 挂 **1 个简化碰撞箱**（机身级）：

- `BoxPool.parent_kind` 新增字段（0=armor / 1=aircraft），`box_part` 用 `PartType` 之外的新值 `AIRCRAFT = 5`；
- 箱体 = 机身胶囊的包围盒（`box_pos` 随飞机每帧同步，`sync_from_parent` 加 aircraft 分支）；
- **碰撞检测内核零改动**：现有「胶囊体 vs AABB」自动覆盖「弹丸 vs 飞机」；
- 命中解析分流：`parent_kind==1` → 走飞机毁伤（§七.1）而非部位判定。

> 取舍：不做部位级（发动机/座舱/翼面）碰撞箱——飞机结构损伤用「损伤系数」而非离散部位
> （§七.1 论证），控制 Phase 7 范围。`PartType.AIRCRAFT` 留枚举位供未来细化。

---

## 二、气动模型（`movement/aerodynamics.py` 新建，Numba）

### 2.1 模型选择：三自由度点质量 + 升力/阻力曲线

不做完整 6DoF 姿态动力学（舵面/惯性矩/角动量），做**点质量飞行模型**——这是兵棋推演的行业
标准切分（参考 stochastic-warfare 的做法）：

$$
\dot{V} = \frac{T - D}{m} - g\sin\gamma, \qquad
\dot{\gamma} = \frac{L\cos\phi - g\cos\gamma}{V}, \qquad
\dot{\psi} = \frac{L\sin\phi}{V\cos\gamma}
$$

其中升力/阻力满足用户要求的「不同迎风面的升力阻力」：

$$
L = \tfrac{1}{2}\rho V^2 S\, C_L(\alpha), \qquad
D = \tfrac{1}{2}\rho V^2 S\left(C_{D0} + k C_L^2\right), \qquad
C_L(\alpha) = \min\left(C_{L\alpha}\alpha,\ C_{L,\max}\right)
$$

- 空气密度 $\rho(z)$ 随高度指数衰减（$z$ 从 span 高度场采样地表基准）；
- 迎角 $\alpha$ 由飞控解出（§三），转弯用倾斜角 $\phi$ 表达（bank-to-turn），不建模侧滑；
- **损伤退化**：`dmg_factor` 同时乘入 $C_{L,\max}$ 与 $T$（气动面/发动机受损 → 失速速度上升、可用推力下降）。

### 2.2 Numba 内核（`@njit(cache=True, parallel=True)`）

```python
@njit(cache=True, parallel=True)
def integrate_aircraft_motion(
    pos, vel, heading, pitch, bank,       # 运动状态
    throttle, phi_cmd, gamma_cmd,         # 飞控输入（Python 层写好的指令）
    fuel, health, dmg_factor, active,     # 状态
    wing_area, cl_max, cd0, k_ind, max_g, # per-机型的静态参数（分组传参）
    v_min, v_max, thrust, mass,
    rho0, altitude_scale,                 # 大气模型
    dt,                                   # = TICK_AIR 0.1s
):
    # prange over active aircraft
    #   rho = rho0 * exp(-z / altitude_scale)
    #   V = |vel|；alpha = atan2 由 pitch 近似 → CL = min(cl_alpha*alpha, cl_max*dmg_factor)
    #   L = 0.5*rho*V^2*S*CL；D = 0.5*rho*V^2*S*(cd0 + k*CL^2)
    #   T = throttle * thrust_max * dmg_factor
    #   加速度沿速度系分解 → 更新 vel/pos/heading/pitch/bank
    #   过载 |a|/g 超 max_g → 按比例衰减 phi_cmd（限制转弯）
    #   V < v_min（失速，dmg 后可能发生）→ state 置 CRASHING 标记输出
```

**性能目标**：128 架 × 0.1s tick → 内核 < 0.05ms（对比：2048 装甲贴地积分 ~0.2ms，规模小两个数量级）。

### 2.3 与现有移动积分的边界

- 现有 `integrate_armor_movement_3d`（地面贴地）/ `integrate_projectile_movement`（抛射物）/ `integrate_guided_projectile_movement`（制导弹丸）**全部不动**；
- `movement/integrate.py` 新增 `build_aircraft_movement_hook(loop, aircraft, terrain)`，仿 `build_armor_movement_hook` 的挂接方式。

---

## 三、飞行控制（`movement/flight_control.py` 新建，Python + 小内核）

### 3.1 自动驾驶分层

| 层 | 输入 | 输出 | 实现 |
|---|---|---|---|
| 任务层（AI，§五/§六） | 意图/目标 | `wp` 航路点 / 机动指令 | Python |
| 航迹层（飞控） | `wp[wp_idx]` / 目标占位 | `throttle/phi_cmd/gamma_cmd` | Python（逐架，128 次/0.1s 可忽略） |
| 物理层（气动积分） | 飞控输入 | 新运动状态 | `@njit`（§二.2） |

### 3.2 定高定航（CRUISE 的核心）

```python
def cruise_control(a: AircraftPool, i: int) -> tuple[float, float, float]:
    # 高度保持：gamma_cmd = kp * (cruise_alt - z) + kd * vz   （爬升率限幅）
    # 速度保持：throttle = 目标速度误差 PI（限幅 + 满油不再加）
    # 航路点转向：到 wp 水平距离 < WP_RADIUS(200m) → wp_idx+1
    #   phi_cmd = 航向差 * 转弯增益（限幅 max_bank 45°）
```

### 3.3 交战机动（ENGAGE，§六.2 消费）

提供三个机动原语供单位 AI 调用：

- **占位（lead pursuit）**：目标航向截获点，飞控解 phi_cmd；
- **规避（kinematic defeat）**：RWR 锁定告警 → 急转弯 + 抛干扰弹（§九.2）；
- **俯冲/拉起（对地投弹/脱离）**：gamma_cmd 程序化序列。

---

## 四、机载传感器：全复用 `EmitterPool` + `detection`

### 4.1 辐射源绑定（spawn 时）

| 源 | EmitterPool 配置 | 备注 |
|---|---|---|
| 机载雷达（开机） | `kind=0` RF，power=radar_power_db，parent=飞机 | 战术关机可被 AI 控制（静默突防） |
| 发动机尾焰 | `kind=3` IR，cur_power 高（充能-残热模型直接套用） | 红外 AAM / IRST 检测目标 |
| 主动雷达导引头（AAM 发射后） | `kind=1`（6b 已建） | 触发敌方 RWR |
| 干扰弹（箔条/热焰） | 瞬时 RF/IR emitter（§九.3） | |

**零新池、零新检测公式**——飞机继承装甲的完整电磁身份。

### 4.2 机载雷达搜索（唯一需要泛化的检测函数）

`detection/sensors.py` 的 `evaluate_seeker_detection`（RCS 四次方根 ×(1-ECM)）目前服务于导引头。
泛化为平台级版本 `evaluate_radar_detection`：**同一公式 + per-平台功率/口径参数**（不再按弹丸
分组，而是按机载雷达分组），检测半径 = 视距公式

$$R_{\max} = \sqrt{2R_e h} \cdot \text{大气修正} + \text{雷达方程收益}$$

输出直接进编队接触表（§五.2）。探测概率仍由 `RNGManager` 采样（沿用 6b 的 `_seeker_detect` 概率
路径）。ESM（RWR）与被动红外检测**零改动**（`evaluate_esm_emitter_detection` / `evaluate_passive_emitter_detection`）。

---

## 五、空中编队 AI（`simulation/air_ai.py` 新建）

### 5.1 决策主体：`AirFormationAI`（对齐地面 `FormationAI` 框架）

```python
class AirFormationIntent(IntEnum):
    CAP = 0        # 空优巡逻（沿航路点扫荡空域）
    STRIKE = 1     # 对地打击（护航 → 突防 → 投弹 → 脱离）
    SEAD = 2       # 防空压制（反辐射猎杀雷达）
    ESCORT = 3     # 护航（伴随 STRIKE 编队）
    RTB = 4        # 全体返航

class AirFormationAI:
    def __init__(self, aircraft: AircraftPool, fc: FiringControl, ...): ...
    def tick(self, loop, dt):
        # 1. gather_air_contacts：雷达/IRST/RWR 接触表（detection 层输出，非上帝视角）
        # 2. situation_of：兵力比（AAM 剩余数 + 存活数）、威胁等级
        # 3. update_intent：FSM（CAP→STRIKE / 劣势→RTB / 护航任务绑定）
        # 4. assign_targets：贪心两轮分配（沿用 6c 模式：先保覆盖后并发）
        # 5. execute_unit_orders：逐成员写 order/target_idx（§六）
```

- 编队成员=飞机索引（`formation_id` 归属），**不复用地面 `FormationAIPool`**——空中编队的任务
  语义（CAP/SEAD/护航）与地面（机动/进攻/防御）差异大，独立 `AirFormationPool`（MAX_AIR_FORMATIONS=32）
  只存 `intent/order/target_zone/member_count/comm`；
- **编队飞行**：长机航路点 + 僚机相对编队位（slot offset）——僚机 wp 由长机位置 + 编队偏移生成，
  队形（楔形/纵队）按任务切换；
- **分组降落**（§七.3）：RTB 编队到达基地空域 → 按梯队拆分 → 依次进入 LANDING（间隔 30s 程序化进近）。

### 5.2 空战态势感知（复用 6c 接触表模式）

`Contact` dataclass 扩展 `kind`（空中/地面）+ `alt`。感知源优先级：

1. **机载雷达**（主动）：精确（距离/方位/航向），但开机=暴露（敌方 RWR 定位）；
2. **IRST**（被动红外）：隐蔽，距离近、无测距；
3. **RWR**：仅方位告警（来袭辐射源定位）；
4. **数据链/通信簇**（Phase 8）：编队间航迹共享（接口预留，§十）。

---

## 六、空中单位 AI：定高定航 + 发现-攻击-规避-返航

### 6.1 状态机（`AirUnitOrder` + 内部状态）

```python
class AirUnitOrder(IntEnum):
    HOLD = 0      # 待命（机场驻停/空中待机）
    CAP = 1       # 空优巡逻（§3.2 定高定航 + 编队队形）
    ENGAGE_AIR = 2    # 空战（占位→发射 AAM→脱离/追击）
    STRIKE_GROUND = 3 # 对地（俯冲投弹/发射 AGM）
    SUPPRESS = 4      # 反辐射压制
    RTB = 5       # 返航（BINGO 或弹药耗尽强制）
    LAND = 6      # 降落
```

**「接受自由开火命令后自主空战」**（AGENTS.md Phase 7 清单）：编队 AI 下达 CAP 且
`weapons_free=True` → 单位 AI 进入自主 OODA：

```
发现（雷达/IRST 接触）→ 判定（超视距？进入射程？）→ 攻击（发射 AAM，§七）
→ 战果评估（命中/脱靶 → 追击或重射）→ 规避（RWR 锁定告警，§九）→ 返航（BINGO/弹尽）
```

### 6.2 交战几何（简化 BVR/WVR）

- **超视距（BVR）**：迎头接敌 → 雷达锁定（EmitterPool kind=1 导引头激活）→ 发射后按
  6b 制导管线自主飞行（`_seeker_detect` 已有目标保持逻辑）→ 导弹命中/脱靶反馈；
- **近距（WVR）**：转弯率对决简化——占位积分 + AAM 全向射程判定（不做完整狗斗几何，
  WVR 胜负由「位置优势 + 转弯率」打分裁决，参考 `score_targets` 模式写 `@njit` 评分内核）；
- **机炮格斗**：Phase 7 不做（近距只用 AAM，机炮列入范围外）。

### 6.3 返航与燃油

- `fuel -= fuel_burn * dt`（接入 `logistics` hook，复用燃料子系统接口）；
- `fuel < bingo` → 强制 RTB（覆盖任何 order，除规避机动外最高优先）；
- 弹药耗尽且无威胁 → RTB 重装填（基地触发，简化：降落 PARKED 后补满）。

---

## 七、交战链：空对空 / 空对地 / 被击落

### 7.1 空对空：AAM vs 飞机

**AAM 全部是 `ProjectilePool` 制导弹丸**（§〇 表格），新增两个弹药级特性：

1. **近炸引信**（`AmmoConfig` 加 `prox_fuse_radius: float = 0.0`）：AAM 与飞机相对距离
   < 引信半径 → 近炸起爆（飞机是高速目标，直接碰撞概率低；引信在 `_step_sub` 的碰撞检测
   前判定，可视为「点 vs 球」检测）；起爆 → 提交 `DamageSubmissionPool` SPHERE（破片毁伤）；
2. **毁伤结算**（`parent_kind==1` 分支）：
   - 破片/爆炸对飞机扣 `health`（无部位）；
   - `dmg_factor` 随 `health` 分段退化（>70% 无损 / 40~70% 轻微 / <40% 气动与推力乘 0.6）——
     **结构损伤用连续系数而非离散部位**，与「气动退化」直接耦合，避免 BoxPool 部位建模；
   - `health <= 0` → `state = CRASHING`。

### 7.2 坠毁链（CRASHING）

```
CRASHING（气动失效：升力×0.2，推力=0，重力主导）
  → 弹道坠落积分（复用 integrate_projectile_movement 的直线+重力即可，或简化自由落体）
  → 触地（采样 span 高度场地面 z）
  → 提交 SPHERE 爆炸（DamageSubmissionPool，附带半径/威力）
  → 可选弹坑（remove_span，复用 6b SAL 炸桥路径）→ 残骸标记（渲染数据）
```

坠毁造成的对地毁伤自然吃现有提交池，**零新结算逻辑**。

### 7.3 空对地：AGM / 炸弹

| 弹种 | 制导复用 | 发射条件 |
|---|---|---|
| 激光 AGM | SEMI_ACTIVE_LASER（SAL 已实施 ✅） | 吊舱照射（6b 已交付） |
| 滑翔炸弹 | 新增 `control_glide_bomb` 注册进 `MissileControlRegistry`（**只读重力转向 + 无推力**，BALLISTIC 后接 TERMINAL 比例导引） | 俯冲投弹 |
| 自由落体炸弹 | `guid_law=-1` 无制导直线+重力（现有路径） | 平飞/俯冲投放 |

AGM/炸弹对地毁伤 = 现有弹药毁伤链（`_submit_hit_blast` 等），**零改动**。

### 7.4 起飞/降落（简化但闭环）

- **起飞**：基地（测试场景配置坐标）→ TAKEOFF 状态：航向对齐 + 加速到 V_rotate → 爬升到
  `cruise_alt` → CRUISE。起飞阶段不参与空战判定；
- **降落**：RTB 到基地空域 → 按梯队进入 LANDING：进近（降高度减速，程序化 gamma_cmd）→
  触地判定（z ≤ 跑道 z + 余量）→ 滑跑减速 → PARKED；
- **基地抽象**：不建模跑道物理（Phase 10 前无新地图），基地 = `AirbaseConfig(坐标, 跑道航向,
  半径)`，场景 YAML 配置；被击毁/无基地 → 飞机转场（简化：地图外退出）。

---

## 八、地面防空激活（role=3）

6c 已预留 `role=3 防空`，Phase 7 激活其交战行为：

- 防空单位（发射车 + 地空导弹）通过现有 `detection` 层检测空中目标（**雷达或红外**，
  `EmitterPool` 已有完整电磁模型）——空中目标 `rcs` 大、尾焰红外强，天然可检；
- SAM = `ProjectilePool` 制导弹丸（PN 制导 + `guid_target_pos` 坐标制导——**注意：现有制导
  内核读坐标、不查池，SAM 追飞机只需把飞机坐标写入 `guid_target_pos`**，零内核改动）；
- 防空编队接入地面 `FormationAI`（成员 role=3，目标 kind 扩展为空中/地面）；
- **空中单位对抗 SAM**：RWR 检测防空雷达（kind=0 RF emitter）→ 规避机动 + 反辐射导弹
  （SEAD，PASSIVE_RF 导引头已有）压制 → 闭环成立。

---

## 九、RWR 告警 + 规避 + 干扰弹（接收 6c 步骤⑨）

6c 设计定稿但延后的「通信簇 + RWR + 诱饵」在 Phase 7 落地（空中单位是最大受益方）：

1. **RWR 告警**：`rwr_state` 由 ESM 检测驱动（`evaluate_esm_emitter_detection` 对 kind=0/1
   RF emitter 单程检测，已有）→ 静默/搜索/跟踪/锁定四级（按功率与稳定度）；
2. **规避**：锁定告警 → 单位 AI 选规避机动（急转弯 + 降高度杂波掩护，§三.3）+ 抛干扰弹；
3. **干扰弹（chaff/flare）**：复用 6b 的诱饵机制雏形——瞬时高功率 emitter（RF 箔条 / IR 热焰），
   `_seeker_detect` 的概率 × ECM 判定已能消费「高功率诱饵源」（需确认 seeker 检测的
   「最近合格目标」包含瞬时 emitter——否则需小改 `_seeker_detect` 的目标选择列表来源）。
   **这是唯一可能需要动 6b 内核的点**，列入实施管线风险项。

---

## 十、与 Phase 8 数据链的接口预留（本阶段不实施）

| Phase 8 需求 | Phase 7 预留 |
|---|---|
| A 导 B 射 | `guid_target_pos` 坐标制导（发射后改坐标=中继制导，已支持）；`guid_comm` 门控（已建） |
| 认知孪生 / 战争迷雾 | AI 接触表一律经 `contact_provider` 注入（6c 已建接缝），Phase 8 换孪生源 |
| 编队战术（磨盘） | `AirFormationIntent` 枚举追加战术位，FSM 可扩展 |
| 航迹共享 | `AirFormationPool.comm` + 接触表合并接口（`merge_contacts`） |

---

## 十一、调度链与分频接入

`simulation/loop.py` 现有链：`("movement", "detection", "ai", "combat", "logistics")`（全部地面语义）。

Phase 7 新增两个 hook（**插在 `combat` 之前**，保证「飞→探测→AI→开火→弹丸推进」的因果顺序）：

```python
self.set_rate("air_movement", TICK_AIR)   # 0.1s：气动积分 + 飞控
self.set_rate("air_ai", TICK_GROUND)      # 1.0s：编队 AI + 单位 AI 决策（空战 OODA 是慢决策）
self.set_rate("combat", TICK_BULLET)      # 不变：AAM/SAM 弹丸已按 0.01s 推进
```

- 空中决策 1s 一拍与地面一致（OODA 循环本来就是秒级，飞行平滑由 0.1s 气动积分保证）；
- 扩容纪律：`AircraftPool._expand` 只在 Tick 间隙（战役切换时）触发，沿用 AGENTS.md 约束。

---

## 十二、实施管线（步骤拆解，后续写实施级文档 `phase7-air-pipeline.md`）

| 步 | 内容 | 依赖 | 验收 |
|---|---|---|---|
| ① | `AircraftPool` + `AircraftConfig` YAML + BoxPool `parent_kind`/`PartType.AIRCRAFT` | — | spawn/destroy/扩容/序列化单测 |
| ② | 气动内核 `integrate_aircraft_motion` + 大气模型 | ① | 平飞/爬升/转弯数值基准（对比解析解） |
| ③ | 飞控层（定高定航/航路点转向/机动原语） | ② | 巡航 10min 高度/速度误差 < 1% |
| ④ | 机载雷达搜索泛化 + 飞机 emitter 绑定 | ① | 探测距离/概率单测；RWR 检测飞机雷达 |
| ⑤ | `AirFormationAI` + `AirFormationPool`（编队飞行 + 任务 FSM） | ③④ | 编队保持/队形切换/CAP 巡逻 |
| ⑥ | 单位 AI 状态机（发现-攻击-规避-返航）+ 空战评分内核 | ⑤ | 端到端 BVR 空战：双方 AAM 互射有胜负 |
| ⑦ | AAM/AGM/滑翔炸弹弹药定义 + 近炸引信 + `control_glide_bomb` | ⑥ | AAM 近炸起爆；滑翔弹命中静止目标 |
| ⑧ | 飞机毁伤（parent_kind=1）+ 坠毁链 | ⑦ | 被击落→坠地→爆炸→弹坑全链 |
| ⑨ | 起飞/降落/基地 + 分组降落 | ③⑤ | 梯队间隔进近、着陆滑跑驻停 |
| ⑩ | 地面防空激活（role=3）+ RWR/干扰弹闭环 | ④⑥⑧ | SAM 击落突防机；SEAD 压制雷达 |
| ⑪ | 集成测试 + 性能基准 + 文档更新 | 全部 | 全量回归 + 压测达标（§十三） |

---

## 十三、测试与性能目标

- **单元测试**：每步对应 `tests/test_phase7_*.py`（沿用现有命名约定），全量回归零衰减；
- **集成测试**：完整空战战役流程自动化（起飞→CAP→BVR 交战→战果→RTB→降落），
  复用 `tests/terrain_factory.py` 测试地图（Phase 10 前不新增地图）；
- **性能目标**：

| 场景 | 目标 |
|---|---|
| 气动积分（128 架 × 0.1s tick） | < 0.05ms |
| 机载雷达搜索（128 架 × 全部目标） | < 1ms（batch 向量化） |
| 完整空战帧（128 架 + 256 AAM 弹丸 + 2048 装甲） | 全帧 < 5ms（对标 Phase 3 战斗预算） |
| 持续压测（10min 空战战役） | 无内存增长、帧率稳定 |

---

## 十四、范围外（明确不做，防止蔓延）

- 6DoF 姿态动力学 / 舵面建模（点质量模型足够兵棋推演）；
- 近距机炮狗斗、直升机旋翼气动；
- 航母、空中加油、跑道物理毁伤；
- 战争迷雾渲染（Phase 8 认知孪生）；
- 空战战术博弈（磨盘战术等，Phase 8 数据链阶段）。

---

> 一句话：**Phase 7 不发明新轮子——它把 6b 的制导弹丸、6c 的编队 AI 和 `EmitterPool` 电磁世界
> 抬到 3000 米高空，然后补上气动、飞控、坠毁这三块唯一缺的拼图。** 🛩️
