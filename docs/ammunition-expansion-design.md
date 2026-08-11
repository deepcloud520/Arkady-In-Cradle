# 弹药体系扩展设计（设计预留 · 未实现）

> 状态：**设计预留（Roadmap），未实现**（2026-08-11）
> 本文档记录已规划但**暂不实施**的弹药体系扩展，供未来 Phase 迭代时实现。
> 当前主线目标：Phase 3（战斗/后勤/模拟循环）收尾 + 遗留项。
> 设计原则：**数据结构预留升级空间，但不在本阶段实现任何新弹种逻辑**。

## 〇、现状基线（2026-08-11）

| 维度 | 现状 |
|------|------|
| `AmmoType` | 6 种：APFSDS / AP / HEAT / HE / ATGM / HESH |
| `WeaponCategory` | 5 类：MAIN_GUN / SECONDARY / MG / ATGM / HOWITZER |
| `ProjectileType` | 2 类：BULLET / MISSILE |
| 穿深模型 | DeMarre 变体（`pen = pen_mm_rha × decay^1.5`，`damage.py`） |
| 毁伤路径 | 动能/化学能两分支 + 斜距/跳弹/后效（`damage.py`） |
| 弹丸元数据 | `engagement._proj_meta: dict[int, (weapon_id, ammo_id)]`（骨架期 dict） |

> ⚠️ 本阶段**不修改**上述现状。以下均为**未来**扩展蓝图。

## 一、弹药类型枚举扩展（AmmoType）

### 1.1 坦克/车辆主炮（现有 + 规划新增）

| 弹种 | 预留枚举值 | 穿透/毁伤机制 | 目标 |
|------|:----------:|--------------|------|
| APFSDS | `APFSDS=0`（现有） | 动能，Lanz-Odermatt | 重甲 |
| AP | `AP=1`（现有） | 动能，短杆/实心 | 中甲 |
| HEAT | `HEAT=2`（现有） | 化学能，固定穿深 | 重甲 |
| HE | `HE=3`（现有） | 爆破+破片 | 软目标/步兵 |
| ATGM | `ATGM=4`（现有） | 化学能，制导 | 远距重甲 |
| HESH | `HESH=5`（现有） | 碎甲，混凝土加成 | 建筑/软甲 |
| **HE_FRAG** | `HE_FRAG=6`（预留） | 高爆+预制破片 | 步兵/轻甲 |
| **PELE** | `PELE=7`（预留） | 轻穿+横向效应 | IFV/APC |
| **THERMOBARIC** | `THERMOBARIC=8`（预留） | 云雾爆轰，封闭×3 | 碉堡/室内 |
| **APCR/APDS** | `APCR=9`（预留） | 动能，钨芯脱壳（旧式） | 中甲 |
| **APCBC** | `APCBC=10`（预留） | 动能，被帽风帽 | 均质甲 |
| **HEAT_MP** | `HEAT_MP=11`（预留） | 化学能+破片+爆破 | 通用（M830A1） |
| **AMP** | `AMP=12`（预留） | 智能切换 KE/爆破/空爆 | 全目标（XM1147） |
| **CANISTER** | `CANISTER=13`（预留） | 霰弹/钢箭 | 步兵（近距） |
| **GUN_LAUNCHED_MISSILE** | `GUN_LAUNCHED_MISSILE=14`（预留） | ATGM，炮口发射 | 远距重甲 |

### 1.2 步兵弹药（规划新增）

| 弹种 | 预留枚举值 | 口径示例 | 机制 |
|------|:----------:|----------|------|
| **RIFLE_AP** | `RIFLE_AP=15`（预留） | 5.56×45 / 5.45×39 | 对软甲/防弹衣 |
| **RIFLE_BALL** | `RIFLE_BALL=16`（预留） | 7.62×51 | 对人员 |
| **RIFLE_TRACER** | `RIFLE_TRACER=17`（预留） | — | 弹道可见，修正 |
| **RIFLE_INCENDIARY** | `RIFLE_INCENDIARY=18`（预留） | 12.7/.50 | 引燃、压制 |
| **HMG_API** | `HMG_API=19`（预留） | 12.7×99 / 14.5 | 轻甲/器材 |
| **GL_HE** | `GL_HE=20`（预留） | 40×46/40×53 | 爆破+破片 |
| **GL_HEDP** | `GL_HEDP=21`（预留） | 40×53 | 破甲+破片 |
| **GL_THERMOBARIC** | `GL_THERMOBARIC=22`（预留） | 40mm | 封闭空间 |
| **RPG_HEAT** | `RPG_HEAT=23`（预留） | RPG-7 PG-7V | 串联破甲 |
| **RPG_THERMOBARIC** | `RPG_THERMOBARIC=24`（预留） | RPG-7 TBG-7V | 温压 |
| **MORTAR_HE** | `MORTAR_HE=25`（预留） | 60/81/120mm | 曲射爆破 |
| **MORTAR_SMOKE** | `MORTAR_SMOKE=26`（预留） | — | 遮蔽 |
| **MORTAR_ILLUM** | `MORTAR_ILLUM=27`（预留） | — | 照明 |
| **HAND_GRENADE** | `HAND_GRENADE=28`（预留） | 破片/烟雾/闪光 | 步兵近战 |

### 1.3 防空可编程弹（规划新增，重点）

| 弹种 | 预留枚举值 | 口径示例 | 机制 |
|------|:----------:|----------|------|
| **AHEAD** | `AHEAD=29`（预留） | 35mm AHEAD | 炮口编程定时，目标前空爆钨柱 |
| **PROG_3P** | `PROG_3P=30`（预留） | 40mm Bofors 3P | 6 模式：近炸/空爆/碰炸/穿甲… |
| **VT_PROXIMITY** | `VT_PROXIMITY=31`（预留） | 76mm/127mm | 雷达近炸 |
| **AA_APFSDS** | `AA_APFSDS=32`（预留） | 35mm/57mm | 动能，直击 |
| **AA_HEI** | `AA_HEI=33`（预留） | 20mm/23mm/30mm | 高爆燃烧 |
| **SAPHEI** | `SAPHEI=34`（预留） | 57mm | 半穿+爆 |

## 二、引信类型枚举（规划新增 FuseType）

防空可编程弹的核心是**引信可编程**，需要独立引信维度：

```python
class FuseType(IntEnum):           # 规划：combat/weapons.py
    IMPACT = 0       # 碰炸引信（默认）
    DELAYED = 1      # 延时引信（穿后延迟引爆）
    PROXIMITY = 2    # 近炸引信（雷达/光学，靠近目标引爆）
    PROGRAMMED = 3   # 可编程空爆（炮口编程，按飞行时间/距离在目标前方引爆）
    TIMED = 4        # 定时引信（固定时间）
    NONE = 5         # 无引信（AP 实心弹）
```

## 三、AmmoConfig 数据模型扩展（规划）

```python
class AmmoConfig(BaseModel):       # 规划：扩展现有 AmmoConfig
    ammo_id: str
    name: str = ""
    ammo_type: AmmoType = AmmoType.APFSDS

    # ---- 弹道/穿深 ----
    muzzle_velocity: float = 0.0
    drag_coefficient: float = 0.15
    pen_mm_rha: float = 0.0            # 0m 穿深（非 Lanz-Odermatt 时）
    # Lanz-Odermatt 参数（未来新增）
    penetrator_length: float = 0.0
    penetrator_diameter: float = 0.0
    penetrator_density: float = 17500.0

    # ---- 爆炸/破片（HE 家族，未来新增）----
    blast_radius_m: float = 0.0        # 冲击波有效半径
    blast_damage: float = 0.0          # 中心最大伤害
    frag_count: int = 0                # 有效破片数
    frag_pen_mm: float = 0.0           # 破片穿深
    frag_spread_deg: float = 0.0       # 破片锥角

    # ---- 引信（防空可编程核心，未来新增）----
    fuse: FuseType = FuseType.IMPACT
    programmable_modes: list[int] = [] # 可编程弹支持的模式（3P 的 6 种）
    prox_radius_m: float = 0.0         # 近炸距离
    airburst_lead_m: float = 0.0       # AHEAD 空爆前置距离

    projectile_type: ProjectileType = ProjectileType.BULLET
    guidance: GuidanceType = GuidanceType.NONE
    is_heat: bool = False
```

## 四、WeaponCategory 扩展（规划）

```python
class WeaponCategory(IntEnum):       # 规划：combat/weapons.py
    MAIN_GUN = 0       # 坦克主炮
    SECONDARY = 1      # 副武器
    MG = 2             # 机枪
    ATGM = 3           # 反坦克导弹
    HOWITZER = 4       # 榴弹炮
    AUTOCANNON = 5     # 自动炮（25/30/35/40mm，可配 AHEAD）
    SMALL_ARMS = 6     # 步枪/卡宾枪（步兵）
    GRENADE_LAUNCHER = 7  # 榴弹发射器（40mm 下挂/自动）
    MORTAR = 8         # 迫击炮
    AA_GUN = 9         # 高射炮（20/23/35/57mm）
    RECOILLESS = 10    # 无后坐力炮/火箭筒（RPG/Carl Gustaf）
```

## 五、Lanz-Odermatt 穿深公式（规划）

### 5.1 公式

```
V_star = V_impact × √(ρp / σt)          # 无量纲归一化速度
P = L × V_star^β / (k + V_star^β)       # 穿深 (mm)
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $L$ | 弹芯长度 (mm) | APFSDS ~600–800mm |
| $D$ | 弹芯直径 (mm) | ~20–30mm |
| $\rho_p$ | 弹芯密度 (kg/m³) | WHA ≈ 17500, DU ≈ 18600 |
| $\sigma_t$ | 靶板强度 (Pa) | RHA ≈ 1.5×10⁹ |
| $\beta$ | 材料指数 | WHA ≈ 0.67, DU ≈ 0.72 |
| $k$ | 临界常数 | ~3.0 (RHA) |

**速度衰减**：`V_impact = V_muzzle × exp(-α × range)`，`α = CD × ρ_air × A_ref / (2 × m)`

### 5.2 与 DeMarre 对比

| 维度 | DeMarre（现状） | Lanz-Odermatt（规划） |
|------|:------------:|:-------------------:|
| 输入参数 | `pen_mm_rha` + `drag` + `range` | `L` + `D` + `ρp` + `V` + `σt` |
| 物理基础 | 经验拟合 | 流体动力学侵蚀模型 |
| L/D 效应 | ❌ | ✅ 长径比自然影响穿深 |
| 材料区分 | ❌ | ✅ 密度差异直接反映 |
| 速度饱和 | ❌ 无上限 | ✅ V→∞ 时 P→L |
| 实现方式 | `resolve_penetration_single` 分支 | 新增 `compute_penetration_lanz_odermatt` |

> **兼容策略（未来实现时）**：`AmmoConfig` 加 `use_lanz_odermatt: bool = False`，
> 旧弹药默认 false 走 DeMarre，新弹药 true 走 Lanz-Odermatt。渐进迁移，不破坏现有存档。

## 六、防空可编程弹机制（规划，重点设计）

```
AHEAD 弹（35mm KDA，瑞士 Oerlikon）：
  1. 炮口装定 → 弹丸携带编程数据（目标距离+速度解算的飞行时间）
  2. 飞行到目标前方 ~8-10m → 弹底抛射 152 枚钨合金圆柱（3.3g，初速≈1700m/s）
  3. 钨柱形成圆锥形弹幕拦截目标 → 对导弹/无人机/直升机效果显著
  4. 实现：ProjectilePool 中 AHEAD 命中判定改为「空爆点生成破片锥」，非物理碰撞

3P 弹（40mm Bofors，Mk 4 可用）：
  1. 炮口编程 6 种模式（近炸对导弹/直升机、空爆对步兵、穿甲对装甲…）
  2. 作战中通过火控在弹上写入模式 → 与 AHEAD 类似但模式更多
  3. 实现：programmable_modes 列表 + FiringControl 根据目标类型选模式

VT 近炸（76mm OTO）：
  1. 雷达多普勒引信 → 距目标 ~15m 引爆
  2. 实现：prox_radius_m + 命中判定改为「弹丸距目标 < prox_radius → 引爆」
```

## 七、实施路线图（未来）

```
Phase 3A: Lanz-Odermatt 公式（~2天）      ← 弹药体系基础
Phase 3B: 新弹种数据模型（~1天）          ← HE_FRAG/PELE/THERMOBARIC
Phase 3C: 毁伤集成（~1天）                ← 爆炸/破片范围判定
Phase 3D: 弹药体系数据扩展（~1天）        ← FuseType/AmmoType 扩展
Phase 3E: 防空可编程弹机制（~2天）        ← AHEAD/3P/VT
Phase 3F: 步兵弹药联动（~1天）            ← 软目标池
```

## 八、数据结构预留升级空间（本阶段已做）

| 预留点 | 文件 | 方式 |
|--------|------|------|
| `AmmoType` 预留编号 6~34 | `combat/weapons.py` | 枚举注释（见代码） |
| `AmmoConfig` 扩展字段 | `combat/weapons.py` | 注释标注未来字段 |
| `WeaponCategory` 预留 5~10 | `combat/weapons.py` | 枚举注释 |
| 弹丸元数据下沉 SoA | `combat/engagement.py` | 现状 `_proj_meta` dict，未来下沉 `proj_weapon/proj_ammo` 数组 |

> 本阶段**仅加注释**，不新增任何枚举值/字段/逻辑，保证现有 158 测试全绿。
> 未来实现时，按本文件 §一~§七 蓝图扩展即可，枚举值编号已预留避免冲突。
