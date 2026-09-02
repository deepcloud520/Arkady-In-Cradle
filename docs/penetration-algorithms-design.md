# 穿深算法分派管线设计 —— DeMarre / A-T 两段式（长杆）

> 状态：**设计定稿 · 管线已实现**（2026-08-18，路线图步骤①~⑧全绿：
> `combat/damage.py` 内核 + `combat/weapons.py` 数据扩展 + `combat/engagement.py` 接线
> + `tests/test_penetration_algorithms.py` 34 用例；内置 M829A3/3BM42 已补 A-T 长杆参数，
> 生产中 APFSDS 真实走 A-T）
> 理论依据：焦文俊、陈小伟《长杆高速侵彻问题研究进展》（力学进展 2019, 49(1)，
> DOI 10.6052/1000-0992-17-021，全文已由作者提供核对）——A-T 模型控制方程
> 式(13)~(17)、终止条件、Tate(1986) 强度标定式(38)(39)、长径比效应式(3)。
> 目标：按**弹种**分派穿深算法；APFSDS 走 **A-T 两段式长杆模型**、其余动能弹走 DeMarre，
> 统一以「弹种固有参数 + 随距离衰减的速度」作为输入计算穿深。

---

## 〇、现状与差距（2026-08 基线）

| 维度 | 现状（`combat/damage.py`） | 差距 |
|------|---------------------------|------|
| 算法分派 | 无。只有 `is_heat` 布尔二分 | 弹种（APFSDS / AP / APCR）无法各自选算法 |
| DeMarre | `pen = pen_rha × max(0.3, 1 − drag·range/1e5)^1.5` | 「穿深随距离衰减」，**不消费速度**，非德马尔速度标度 |
| 速度输入 | `muzzle_velocity` 仅用于跳弹阈值 | 未实现「随距离衰减的速度」作为穿深输入 |
| 长杆模型 | 未实现 | APFSDS 的 L/D、材料强度、临界速度、两段式（侵蚀 + 刚性残杆）缺失 |

一句话：**现有穿深是"标称穿深 × 距离系数"，本设计改为"距离衰减速度 → 算法穿深（动能按速度标度，长杆按 A-T 两段式）"**。

---

## 一、目标与设计原则

1. **弹种决定算法**：`AmmoType → PenetrationModel` 显式分派表。
2. **速度是公共输入**：所有动能弹先做「距离 → 撞击速度」衰减，再交给各自算法。
3. **长杆按 A-T 两段式**：控制方程严格按综述式(13)~(17)，含临界速度与两种终止情形。
4. **固有参数分侧归属**：弹杆侧（L、ρp、Yp）进 `AmmoConfig`；靶侧（ρt、Rt）按
   `ArmorType` 查表（靶属性不是弹属性）。
5. **兼容不破坏存档**：旧弹药缺参数 → 自动回退 DeMarre；现有测试/存档语义不变。
6. **Numba 内核只做数学**：分派在 Python 层（命中数少）；速度衰减/穿深/ODE 积分为纯函数内核。

---

## 二、管线总览（数据流）

```
┌───────────────────── 开火（combat/engagement.py） ─────────────────────┐
│ fire() / fire_indirect()                                              │
│   ├─ AmmoConfig 快照（弹杆固有参数）                                    │
│   ├─ WeaponSystem.solve() → sigma / 初速 v0                            │
│   └─ ProjectilePool.spawn() → SoA 写入 weapon/ammo 索引                │
└──────────────────────────────┬────────────────────────────────────────┘
                               ▼
┌───────────────────── 命中（_resolve_hit）─────────────────────────────┐
│ ① 反查弹杆固有参数：ammo_at(proj.ammo) → AmmoConfig                    │
│ ② 反查靶侧参数：    TARGET_ARMOR_MODEL[armor_type] → (ρt, Rt)          │
│ ③ 分派：model = penetration_model_of(ammo.ammo_type, has_rod_params)  │
│ ④ 距离衰减速度：v_imp = impact_velocity(v0, alpha, range_m)            │
│ ⑤ 算法穿深：  pen = compute_penetration(model, 弹杆参数, 靶参数, v_imp) │
│      ├─ DEMARRE       → pen0·(v_imp/v0)^1.43                           │
│      ├─ AT_EROSION    → A-T 两段式 ODE 积分（综述 Eq.13~17）            │
│      └─ FIXED         → pen0（化学能）                                  │
│ ⑥ 斜距/跳弹：armor_eff = compute_armor_eff(...)（不变）                │
│ ⑦ 穿透余量：  margin = pen − armor_eff（多层装甲逐层递减 rem_pen）      │
│ ⑧ 后效毁伤：  resolve_behind_armor_single(...)（不变）                 │
└───────────────────────────────────────────────────────────────────────┘
```

**关键改动点（三处）**：
- `combat/damage.py`：速度衰减 + 分派 + DeMarre/A-T 内核，重构 `resolve_penetration_single`。
- `combat/weapons.py`：`AmmoConfig` 扩展弹杆固有参数（L/ρp/Yp）+ 速度衰减系数。
- 靶侧参数表 `TARGET_ARMOR_MODEL`（按 `ArmorType` 提供 ρt/Rt，替代把靶属性塞进 AmmoConfig 的错误设计）。

---

## 三、算法分派（按弹种）

```python
class PenetrationModel(IntEnum):      # 建议置于 combat/damage.py
    FIXED = 0            # 化学能/固定穿深（不衰减）
    DEMARRE = 1          # 德马尔速度标度（全口径 AP / APCR / APDS）
    AT_EROSION = 2       # A-T 两段式长杆（APFSDS，侵蚀 + 刚性残杆）
    LO_APPROX = 3        # 简化 Lanz-Odermatt（可选快速近似/标定参考）
```

| AmmoType | 分派结果 | 说明 |
|----------|:--------:|------|
| `APFSDS` | **AT_EROSION** | 弹杆参数齐（L>0 且 Yp>0）→ A-T；否则回退 DEMARRE |
| `AP` / `APCR` / `APDS` | **DEMARRE** | 全口径/次口径动能，经验速度标度 |
| `HEAT` / `ATGM` / `HESH` / `HE` / `HE_FRAG` / `PELE` / `SAPHEI` | **FIXED**（爆炸/破片走提交池，不在此链） | 化学能穿深恒定 |

```python
def penetration_model_of(ammo_type: int, has_rod_params: bool) -> int:
    """弹种 → 穿深模型；APFSDS 缺长杆参数时回退德马尔。"""
    if ammo_type in _CHEMICAL_TYPES:           # HEAT/ATGM/HESH/HE/HE_FRAG/...
        return PenetrationModel.FIXED
    if ammo_type == AmmoType.APFSDS and has_rod_params:
        return PenetrationModel.AT_EROSION
    return PenetrationModel.DEMARRE
```

> `has_rod_params = penetrator_length_mm > 0 and penetrator_strength_pa > 0`。

---

## 四、速度衰减模型（公共输入）

闭合式指数衰减（弹丸飞行物理保持恒定速简化，穿深速度在命中时解析计算）：

```
v_imp = v0 · exp(−α · range)
```

| 参数 | 含义 | 建议值/单位 |
|------|------|------------|
| `v0` | 初速（`AmmoConfig.muzzle_velocity`） | m/s |
| `range` | 射距（发射点 → 命中点，`_range_of`） | m |
| `α` | 每米速度衰减系数 | 1/m，**每弹药显式配置** |

**α 数量级**：真实 APFSDS 约 40~60 m/s/km → `α ≈ 3×10⁻⁵ ~ 1×10⁻⁴`。
M829A3（v0≈1555）取 α=3.5×10⁻⁵，2 km 处 v_imp≈1495。

**α 理论推导**（仅文档；本项目 `drag_coefficient` 是 /1e5 尺度经验系数，非物理 CD）：
```
长杆：  α = CD·ρ_air / (2·ρp·L)          # A_ref=πD²/4，m=ρp·πD²/4·L
全口径：α = CD·ρ_air·A_ref / (2·m)
```

```python
@njit(cache=True)
def impact_velocity(v0: float, alpha: float, range_m: float) -> float:
    """闭合式速度衰减。alpha<=0 或 v0<=0 时原速返回。"""
    if alpha <= 0.0 or v0 <= 0.0:
        return v0
    return v0 * np.exp(-alpha * range_m)
```

---

## 五、算法设计

### 5.1 DeMarre（德马尔）—— 全口径/次口径动能弹

经典德马尔式（法向）：`v = K·d^0.75·b^0.7/m^0.5`，指数 `1/0.7 ≈ 1.4286` 是速度标度核心。

**实现采用"锚定比例"形式**（0m 标称穿深 `pen_mm_rha` 为校准锚，消费 v_imp）：

```
P(v) = pen_mm_rha · ( v_imp / v0 )^1.4286
```

```python
@njit(cache=True)
def penetration_demarre(pen0: float, v0: float, v_imp: float) -> float:
    """德马尔速度标度：P = pen0 · (v_imp/v0)^(1/0.7)。"""
    if v0 <= 0.0:
        return pen0
    return pen0 * (max(v_imp, 0.0) / v0) ** 1.428571
```

### 5.2 A-T 两段式长杆模型 —— APFSDS（按综述原文校核）

**① 控制方程组（综述式(13)~(16)）**：状态变量 = 弹尾速度 `v`、剩余杆长 `l'`、穿深 `p`。

```
½·ρp·(v − u)² + Yp = ½·ρt·u² + Rt        (13)  界面动量平衡（修正伯努利）
ρp·l'·(dv/dt) = −Yp                       (14)  弹体减速（弹杆流动应力控制）
dl'/dt = −(v − u)                         (15)  弹杆侵蚀
dp/dt = u                                 (16)  穿深推进
```

**② 界面侵彻速度与临界速度（综述式(17)）**：

```
μ = √(ρt / ρp)                                        # 弹靶密度比（开根号！）
u = ( v − √( μ²v² + (1−μ²)·Vc² ) ) / ( 1 − μ² )       # 界面侵彻速度
Vc = √( 2·|Rt − Yp| / ρp )                             # 弹尾临界速度（绝对值）
```

**③ 侵彻终止（两段式，综述 §4.2.1）**——按弹靶强度关系分两种情形：

| 情形 | 终止条件 | 行为 |
|------|----------|------|
| **Rt > Yp**（靶强于杆，APFSDS 典型） | v 降至 `Vc` | u→0，**侵彻停止**（无刚性次级） |
| **Rt < Yp**（杆强于靶） | v 降至 `Vc/μ` | u=v，**剩余杆以刚性弹继续侵彻**（次级） |

刚性段（仅 Rt < Yp，l' 恒定）：
```
dv/dt = −Rt / (ρp·l')      dp/dt = v         直至 v = 0
```

> ⚠️ **与我此前设计的差异（原文校核结果）**：上一版把「v≤Vc 一律转刚性段」写错了。
> 原文明确：仅当**杆强于靶**（Rt<Yp，如钨杆打软靶）才有刚性残杆次级侵彻；
> APFSDS 打装甲钢（Rt>Yp）在 v=Vc 处 u→0 直接停止。

**④ 数值积分（前向欧拉 + 自适应步长，示意；实现可用 RK4）**：

```python
@njit(cache=True)
def penetration_at_erosion(
    L0: float, rho_p: float, rho_t: float,
    Yp: float, Rt: float, v_imp: float,
) -> float:
    """A-T 两段式长杆穿深（SI 内部，返回 mm）。综述 Eq.(13)~(17) + 终止条件。"""
    if L0 <= 0.0 or rho_p <= 0.0 or v_imp <= 0.0:
        return 0.0
    mu2 = rho_t / rho_p                # μ²
    mu = math.sqrt(mu2)                # μ = √(ρt/ρp)
    Vc = math.sqrt(max(0.0, 2.0 * abs(Rt - Yp) / rho_p))
    rigid = Rt < Yp                    # 仅杆强于靶才有刚性次级
    v_end = Vc / mu if rigid else Vc

    L = L0 * 1e-3                      # m
    v = v_imp                          # m/s
    P = 0.0                            # m
    # 段①：流体动力学侵蚀（v > v_end）
    while L > 1e-6 and v > v_end + 1e-3:
        disc = mu2 * v * v + (1.0 - mu2) * Vc * Vc
        disc = max(disc, 0.0)
        if abs(1.0 - mu2) > 1e-9:
            u = (v - math.sqrt(disc)) / (1.0 - mu2)   # Eq.(17)
        else:
            u = (v * v - Vc * Vc) / (2.0 * v)         # μ=1 退化
        u = max(u, 0.0)
        dL = -(v - u)                  # 侵蚀
        dv = -Yp / (rho_p * L)         # 弹体减速
        dP = u                         # 穿深推进
        dt = 0.02 * L / max(v - u, 1e-3)              # 自适应：每步消耗 ~2% 杆长
        L += dL * dt; v += dv * dt; P += dP * dt

    # 段②：刚性残杆次级侵彻（仅 Rt < Yp）
    if rigid and v > 1e-3:
        while L > 1e-6 and v > 1e-3:
            dv = -Rt / (rho_p * L)     # 刚性减速
            dP = v
            dt = 0.02 * L / max(v, 1e-3)
            v += dv * dt; P += dP * dt
    return P * 1000.0                  # → mm
```

- 侵蚀段总时长 ~ `L0/(v−u)`，APFSDS 约 1~2 ms；自适应步长下约 50~100 步，单发成本 < 1µs 量级。
- **物理上界自洽**：Yp=Rt=0 纯伯努利时 `P → L/μ = L·√(ρp/ρt)`（综述式(8)流体动力学极限），
  可作单元测试不变量。
- **斜距一致性**：综述 §9.2.1 的斜侵彻处理 = 沿速度方向靶厚按 `H/cosβ` 折算 →
  与本管线 `armor_eff = armor/cos(angle)` 的既有做法一致，A-T 内核只算法向半无限靶穿深。

### 5.3 弹靶强度标定（Tate 1986，综述式(38)(39) + σy=4.2BHN）

A-T 参数 `Yp` / `Rt` 是重点难点，综述给出了可操作的标定公式：

```
Yp = 1.7·σyp                               # 弹杆强度（Tate 1986）
Rt = σyt·( 2/3 + ln(0.57·Et/σyt) )         # 靶体阻力（Tate 1986）
σy = 4.2·BHN                               # 动态压缩屈服强度 ← 布氏硬度
HEL = (1−ν)/(1−2ν)·σyp                     # Hugoniot 弹性极限（Tate 1967 初版）
```

| 材料 | BHN | σy (GPa) | Yp 或 Rt (GPa) | 说明 |
|------|:---:|:--------:|:--------------:|------|
| WHA 弹杆 | ≈420 | 1.76 | Yp = 1.7σyp ≈ **3.0** | 弹杆动态强度 |
| DU 弹杆 | ≈380 | 1.60 | Yp ≈ **2.7** | 自锐、略低强度 |
| RHA 靶 | ≈350 | 1.47 | Rt ≈ **7.4**（Et=207 GPa） | Vc = √(2(7.4−3.0)e9/17500) ≈ **710 m/s** |
| 高硬度装甲钢 | ≈500 | 2.10 | Rt ≈ **9.8** | 更抗侵彻 |

> 上表为**可直接落地的标定数值**（把之前的「待标定」钉死）。数据作者仍可覆写 Yp/Rt。

### 5.4 长径比效应与经验校验（综述 §7.1 / 式(3)）

综述 §4.2.3 明确指出 **A-T 模型无法反映长径比效应**（侵彻效率随 L/D 增大而下降）。
提供 Anderson 等经验式作**可选校验/快速近似**（仅兵器速度范围有效）：

```
P/L = −0.209 + 1.044·V̄ − 0.194·ln(L/D)，  V̄ = V/1500，V ∈ [1.2, 2.7] km/s
```

- 用途①：A-T 积分结果的**交叉校验**（误差诊断）。
- 用途②：数据作者想跳过 ODE 积分时的快速近似（`PenetrationModel.LO_APPROX` 同级）。
- 注意：式(3)对 L/D>30 与 V<1.2 km/s 外推不可靠，仅作参考。

### 5.5 简化 Lanz-Odermatt（可选快速近似 / 标定参考）

单段 logistic 饱和（原 `ammunition-expansion-design.md` §5.1），`LO_APPROX` 用：

```
V* = v_imp · √(ρp / Rt)          # 无量纲归一化速度（σt 取靶侧 Rt）
P  = L · V*^β / (k + V*^β)       # V*→∞ 时 P→L（速度饱和）
```
`β`（WHA≈0.67 / DU≈0.72）、`k`（≈3.0）为材料常数。缺省不启用。

### 5.6 化学能 / 固定穿深（不变）

```
P = pen_mm_rha      # HEAT/ATGM/HESH，不随距离/速度衰减，仍受 ERA/复合装甲克制
```

### 5.7 长杆跳飞判据（Rosenberg 式(60)，综述 §9.2.2）—— 路线图步骤⑨

> 状态：**已实现**（2026-08-18：`damage.interface_velocity` 公共辅助 +
> `resolve_penetration_v2` 接入；仅适用于 `AT_EROSION`（长杆）；DeMarre 全口径弹
> 维持既有固定阈值（>75° 必跳、60~75° 且 v_imp<500 → 跳）。

综述 §9.2.2 给出长杆斜击跳飞的物理判据，两个候选模型：

```
Tate (1979)         tan³β > (2ρp·V²/(3Yp)) · ((L²+D²)/(L·D)) · V/(V−U)    (59)  转动惯量，含 L/D
Rosenberg (1989)    tan²β > (ρp·V²/Rt) · (V+U)/(V−U)                      (60)  塑性铰，无杆长项
Rosenberg et al. (2007)  tan²βc^min = ρp·Vc²/Rt = 2(1−Rt/Yp)             (61)  仅 Rt<Yp；Rt/Yp=1/3 → βc^min≈49°
```

**采用 Rosenberg 式(60)**（综述称「模型较好地解释了实验」，且不需要 D，只需 v_imp、u、ρp、Rt）：

```
跳弹 ⟺  tan²β > T(v_imp) ， T(v) = (ρp·v²/Rt) · (v+u)/(v−u)
β = 入射角（自法线，即本管线的 impact_angle_deg）；u = 界面侵彻速度（A-T 内核产出）
```

**临界角 βc(v) = atan(√T(v))** 随速度连续变化（M829A3：ρp=17500，Rt=7.4e9，Vc≈709）：

| v_imp (m/s) | u (m/s) | T(v) | βc（临界跳飞角） |
|:-----------:|:-------:|:----:|:----------------:|
| 1555（0m 初速） | ≈704 | 15.2 | ≈ **75.6°** |
| 1500 | ≈663 | 13.8 | ≈ **74.9°** |
| 1000 | ≈268 | 4.10 | ≈ **63.7°** |
| 710（≈Vc） | →0 | 1.19 | ≈ **47.5°** |

**语义对照既有阈值**：近距高速（贴近初速）βc≈75°，与现「>75° 必跳」自然吻合；
低速段 βc 从 ~64° 逐步降到 ~47°，替代现「60~75° 且 v<500 才跳」的粗糙分段——**一条连续
物理判据同时取代硬/软两段**，且「低速更易跳」的物理直觉由公式自带。

**实现约束**：
1. 判据仅当 **v_imp > Vc**（半流体侵蚀段）成立；v_imp ≤ Vc 时 u→0 无穿深，回退既有
   固定阈值（Rosenberg 判据在刚性段未推导，不适用）。
2. `u` 由 A-T 界面方程给出 → 抽出公共 `@njit interface_velocity(v, μ², Vc²)` 供
   跳弹判定与 `penetration_at_erosion` 共用（单一真源，避免两处求 u 漂移）。
3. 跳弹判据在 `resolve_penetration_v2` 的 AT_EROSION 分支内完成（那里有 ρp/Rt/v_imp）。

### 5.8 标定锚（比例锚，0m 对齐既有标称穿深）—— 路线图步骤⑩

> 状态：**设计定稿（待实现）**。解决「A-T 物理模型输出 ≠ 数据表 pen_mm_rha」的平衡漂移。

**问题**：A-T 未标定时 M829A3 0m≈523mm，弱于数据表标称 700mm。锚定可让物理曲线形状
保留、同时把 0m 值对齐标称。

**公式**（锚到 RHA 基准，故「mm RHA」语义自洽）：

```
s    = pen_mm_rha / P_RHA(v0)              # 比例锚（每弹药常数）
P_eff(v_imp) = P(v_imp) × s                # 命中时再乘锚
P_RHA(v0) = penetration_at_erosion(L, ρp, ρt_RHA, Yp, Rt_RHA, v0)
```

- **基准靶恒为 RHA**（ρt=7850，Rt=7.4e9）：`s` 只依赖弹药侧参数 + RHA 常量 → 纯每弹药常数，
  与命中目标无关（穿深本就以「mm RHA 当量」为统一量纲）。
- **形状保留**：`P_eff(v_imp) = P(v_imp)·s` 只整体重标定，A-T 的速度衰减形状不动。
- **DeMarre 天然已锚**：`P = pen0·(v_imp/v0)^1.43` 在 v0 处恒等于 pen0 → s=1，无需锚。
  锚仅作用于 AT_EROSION / LO_APPROX（LO 的 P(v0) 也≠pen0，同样可锚）。

**M829A3 示例**（s = 700/523.4 = 1.3375）：

| range (m) | P_A-T (mm) | P_eff = P×s (mm) |
|:---------:|:----------:|:----------------:|
| 0 | 523 | **700**（对齐标称） |
| 500 | 507 | 678 |
| 2000 | 456 | 610 |
| 4000 | 388 | 519 |

**数据模型**：`AmmoConfig` 加 `pen_anchor: bool = False`。`s` 为派生值，不序列化：
加载/首用时可缓存（`s` 计算 = 一次 A-T 积分，单发 ~1µs，命中数少时甚至可每发现算；
推荐在注册表加载时预缓存于 `{ammo_id: s}` 派生表）。

**与「反标定 Yp/Rt」的关系**（二选一，见 §九）：锚是「数据对齐」捷径、物理曲线形状
不变；反标定 Yp/Rt 是「物理参数重校」，曲线形状会变。数据作者偏好物理一致性 → 反标定；
偏好零漂移对齐既有数据 → 锚。辅助标定工具（`docs/penetration-calibration-tool-design.md`）
同时提供两者。

---

## 六、统一分派内核（Numba 落地）

公共辅助（A-T 界面侵彻速度，跳弹判定与穿深积分共用单一真源）：

```python
@njit(cache=True)
def interface_velocity(v: float, mu2: float, Vc2: float) -> float:
    """综述式(17)：u = (v − √(μ²v² + (1−μ²)Vc²)) / (1−μ²)。"""
    disc = max(mu2 * v * v + (1.0 - mu2) * Vc2, 0.0)
    if abs(1.0 - mu2) > 1e-9:
        return max((v - np.sqrt(disc)) / (1.0 - mu2), 0.0)
    return max((v * v - Vc2) / (2.0 * v), 0.0)   # μ=1 退化
```

```python
@njit(cache=True)
def compute_penetration(
    model: int,
    pen0: float, v0: float, v_imp: float,        # DeMarre 输入
    L0: float, rho_p: float, Yp: float,          # 弹杆固有参数
    rho_t: float, Rt: float,                     # 靶侧参数（ArmorType 查表）
    lo_beta: float, lo_k: float,                 # 简化 LO 输入
    anchor_s: float = 1.0,                       # 标定锚比例（§5.8）
) -> float:
    if model == 1:                                # DEMARRE
        return penetration_demarre(pen0, v0, v_imp)
    if model == 2:                                # AT_EROSION
        return penetration_at_erosion(L0, rho_p, rho_t, Yp, Rt, v_imp) * anchor_s
    if model == 3:                                # LO_APPROX
        return penetration_lanz_odermatt(L0, rho_p, Rt, lo_beta, lo_k, v_imp) * anchor_s
    return pen0                                   # FIXED
```

> 注：`anchor_s`（标定锚比例）属步骤⑩ 设计，**当前实现未含此参数**（⑩ 仅留文档）；
> 落地时在 `compute_penetration` 签名追加并按 §5.8 语义乘入 A-T/LO 分支。

重构后的单发判定（`resolve_penetration_v2`，含长杆跳飞判据）：

```python
@njit(cache=True)
def resolve_penetration_v2(
    model, pen0, v0, alpha,                       # 分派 + 速度
    L0, rho_p, Yp, rho_t, Rt, lo_beta, lo_k,      # 弹杆 + 靶参数
    anchor_s,                                     # 标定锚比例
    armor_eff, impact_angle_deg, range_m,
    out_margin, out_ricochet,
) -> None:
    out_ricochet[0] = False
    v_imp = impact_velocity(v0, alpha, range_m)

    if model == _PEN_AT_EROSION:
        # 长杆跳飞判据（Rosenberg 式(60)，§5.7）：仅 v_imp>Vc 半流体段适用
        mu2 = rho_t / rho_p
        Vc2 = max(0.0, 2.0 * abs(Rt - Yp) / rho_p)
        Vc = np.sqrt(Vc2)
        if v_imp > Vc:
            u = interface_velocity(v_imp, mu2, Vc2)
            tan2 = np.tan(np.radians(abs(impact_angle_deg))) ** 2
            T = (rho_p * v_imp * v_imp / Rt) * ((v_imp + u) / max(v_imp - u, 1e-3))
            if tan2 > T:
                out_ricochet[0] = True
                out_margin[0] = -armor_eff
                return
        # 刚性段/退化：回退固定阈值（>75° 必跳；60~75° 且 v_imp<500 → 跳）
        if abs(impact_angle_deg) > RICHOCHET_HARD_ANGLE:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return
        if abs(impact_angle_deg) > RICHOCHET_SOFT_ANGLE and v_imp < 500.0:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return
    else:
        # DeMarre/FIXED/LO：维持既有固定阈值（硬 75° / 软 60°+低速）
        if abs(impact_angle_deg) > RICHOCHET_HARD_ANGLE:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return
        if abs(impact_angle_deg) > RICHOCHET_SOFT_ANGLE and v_imp < 500.0:
            out_ricochet[0] = True
            out_margin[0] = -armor_eff
            return

    pen = compute_penetration(model, pen0, v0, v_imp, L0, rho_p, Yp, rho_t, Rt,
                              lo_beta, lo_k, anchor_s)
    out_margin[0] = pen - armor_eff
```

> 改进点：软跳弹速度阈值由 `muzzle_velocity` 改为 `v_imp`（距离衰减速度）；
> 长杆（AT_EROSION）半流体段用 Rosenberg 判据取代固定角度阈值（§5.7），
> DeMarre 全口径弹维持固定阈值不变。

---

## 七、数据模型扩展

### 7.1 `AmmoConfig`（弹杆侧，`combat/weapons.py`）

追加字段，**全部带默认值**，旧 YAML/测试构造零改动：

```python
class AmmoConfig(BaseModel):
    # ...既有字段不变...
    # ---- 穿深算法分派 ----
    penetration_model: PenetrationModel = PenetrationModel.DEMARRE  # 兜底；推荐由 ammo_type 派生
    # ---- 速度衰减 ----
    velocity_alpha: float = 0.0             # 每米衰减系数 (1/m)；0=不衰减（旧行为）
    # ---- A-T 弹杆固有参数（弹杆侧，靶参数不在此）----
    penetrator_length_mm: float = 0.0       # 弹芯初始长度 L；>0 时 APFSDS 走 A-T
    penetrator_diameter_mm: float = 0.0     # 弹芯直径 D（L/D 效应校验、Tate 跳飞式(59)、开坑）
    penetrator_density: float = 17500.0     # ρp (kg/m³) WHA=17500 / DU=18600
    penetrator_strength_pa: float = 0.0     # Yp 杆动态流应力 (Pa)；Tate 1986: 1.7·σyp
    # ---- 标定锚（§5.8）----
    pen_anchor: bool = False                # 比例锚：P_eff = P × (pen_mm_rha/P_RHA(v0))
    # ---- 简化 Lanz-Odermatt（可选近似）----
    lo_beta: float = 0.67                   # β WHA=0.67 / DU=0.72
    lo_k: float = 3.0                       # k
```

### 7.2 靶侧参数表（新，`combat/damage.py`）

靶属性（ρt、Rt）按 `ArmorType` 查表——**不再塞进 AmmoConfig**：

```python
# 靶侧长杆侵彻参数（Tate 1986 标定；ρt kg/m³，Rt Pa）
TARGET_ARMOR_MODEL: dict[int, tuple[float, float]] = {
    ArmorType.RHA:       (7850.0, 7.4e9),   # BHN≈350 → σyt=1.47GPa → Rt≈7.4GPa
    ArmorType.COMPOSITE: (4500.0, 8.5e9),   # 陶瓷/复合等效（待标定）
    ArmorType.REACTIVE:  (7800.0, 5.5e9),   # ERA 板（待标定）
    ArmorType.SPACED:    (7850.0, 4.0e9),   # 间隙装甲（待标定）
}
```

内置示例（M829A3 / 3BM42，A-T 参数可直接落地）：

| 弹药 | L(mm) | ρp | Yp(Pa) | α(1/m) |
|------|:----:|:----:|:----:|:----:|
| M829A3 APFSDS | 780 | 17500 | 3.0e9 | 3.5e-5 |
| 3BM42 APFSDS | 660 | 17500 | 3.0e9 | 3.0e-5 |

---

## 八、与命中链的接线点（`combat/engagement.py`）

`_resolve_hit` 中穿透部分改为：

```python
ammo = self.weapon_system.registry.ammo_at(ammo_idx) if ammo_idx >= 0 else None
if ammo is None:
    model = PenetrationModel.FIXED
else:
    has_rod = ammo.penetrator_length_mm > 0.0 and ammo.penetrator_strength_pa > 0.0
    model = penetration_model_of(ammo.ammo_type, has_rod)

# 靶侧参数（按命中部位的装甲类型查表；默认 RHA）
armor_type = self.damage_resolver.armor_type_of(armor_idx)
rho_t, Rt = TARGET_ARMOR_MODEL.get(armor_type, TARGET_ARMOR_MODEL[ArmorType.RHA])

# 速度衰减一次（各层装甲路径长度可忽略，v_imp 全层共享）
v_imp = impact_velocity(ammo.muzzle_velocity, ammo.velocity_alpha, range_m)

for _t, part, bidx in box_hits:
    result = self.damage_resolver.resolve(
        ...,
        model=model,
        penetrator_length_mm=ammo.penetrator_length_mm,
        penetrator_density=ammo.penetrator_density,
        penetrator_strength_pa=ammo.penetrator_strength_pa,
        target_density=rho_t, target_strength_pa=Rt,
        lo_beta=ammo.lo_beta, lo_k=ammo.lo_k,
        velocity_alpha=ammo.velocity_alpha,
        ...
    )
```

- 多层装甲的 `rem_pen`（剩余穿深递减）语义不变：`rem_pen` 初始化为算法穿深 `pen`，每穿一层减 `armor_eff`。
- 攻顶链（`_resolve_top_attack`）继续走 `is_heat=True` 固定穿深，不受影响。

---

## 九、兼容与渐进迁移

| 场景 | 行为 |
|------|------|
| 旧 APFSDS（`penetrator_length_mm=0`） | 回退 DEMARRE；`velocity_alpha=0` → 不衰减 → 与旧 `decay^1.5` 近似对齐 |
| 旧 AP/APCR | DEMARRE + α=0 → 穿深恒等于 pen_mm_rha（迁移时给 α 赋合理值） |
| 新 APFSDS（含长杆参数） | AT_EROSION + α>0，获得两段式 + 距离衰减 |
| 存档 | `AmmoConfig` 仅加默认字段，msgpack 池/配置不涉及新字段 → 无损往返 |

**标定锚（0m 对齐既有 `pen_mm_rha`，详见 §5.8）**：A-T 输出绝对穿深 P；要旧数据零破坏，
对特定弹药开启 `pen_anchor=True` → `P_eff = P × (pen_mm_rha/P_RHA(v0))`。默认**不启用**
（物理模型优先）；数据作者二选一：按 §5.3 公式标定 Yp/Rt 使 P(v0)≈pen_mm_rha，或开启比例锚。

**跳飞（详见 §5.7）**：长杆（AT_EROSION）半流体段改用 Rosenberg 式(60) 速度相关判据；
DeMarre 全口径弹维持既有固定阈值。旧存档不含新字段，行为回退到固定阈值路径。

> 迁移后需重跑一次 `uv run python clear_numba_cache.py -y`（AGENTS.md 六：改 Numba 核必清缓存）。

---

## 十、测试用例

- [ ] **分派正确性**：`penetration_model_of` 各弹种 → 预期模型（APFSDS→AT_EROSION、AP→DeMarre、HEAT→FIXED）。
- [ ] **μ 约定**：μ = √(ρt/ρp)；Yp=Rt=0 时 `u = v/(1+μ)`、`P → L/μ`（综述式(7)(8)）。
- [ ] **0m 锚定**：DeMarre 满足 `P(v0) == pen_mm_rha`；A-T 开启比例锚时亦满足（误差 < 1e-3）。
- [ ] **A-T 临界速度**：v_imp ≤ Vc 时 `u=0` 无穿深（Rt>Yp）；恰好 Vc 处连续。
- [ ] **A-T 两段式**：Rt<Yp 时存在刚性残杆次级（P 有第二段贡献）；Rt>Yp 时无。
- [ ] **A-T 单调性**：P 随 v_imp、L、ρp 单调不减；随 Rt 单调不增。
- [ ] **标定公式**：WHA(BHN 420)/RHA(BHN 350) → Yp≈3.0GPa、Rt≈7.4GPa、Vc≈710m/s（±5%）。
- [ ] **距离衰减**：α>0 时 v_imp 随 range 单调下降、穿深随之下降；α=0 时不变。
- [ ] **速度标度**：DeMarre `P ∝ v^1.4286`（v 减半 → P × 2^(−1.4286)）。
- [ ] **回退**：旧 APFSDS（L=0）走 DeMarre，行为与迁移前可比。
- [ ] **软跳弹速度**：远距低速（v_imp<500）在 60~75° 触发跳弹，近距高速不触发。
- [ ] **长杆跳飞判据（Rosenberg）**：AT_EROSION 高速（≈1555 m/s）βc≈75°、低速（≈1000 m/s）βc≈60°、趋近 Vc βc≈47°；v_imp≤Vc 回退固定阈值；DeMarre 不受影响。
- [ ] **标定锚**：`pen_anchor=True` 时 `P_eff(v0) == pen_mm_rha`（误差 < 1e-3）；P_eff 随距离仍单调衰减（形状保留）；DeMarre 锚恒等（s=1）。
- [ ] **性能**：`compute_penetration` 单发 < 1µs；2048 发批量内核 < 1ms（沿用 phase3 §六 预算）。

---

## 十一、实施路线图（2026-08-18：①~⑧ 已完成 ✅）

```
步骤①  ✅ 枚举 + 分派函数         combat/damage.py（PenetrationModel / penetration_model_of）
步骤②  ✅ 速度衰减 + DeMarre 内核  impact_velocity / penetration_demarre
步骤③  ✅ A-T 两段式内核           penetration_at_erosion（综述 Eq.13~17 + 终止条件）
步骤④  ✅ 靶侧参数表              TARGET_ARMOR_MODEL（按 ArmorType，Tate 1986 标定）
步骤⑤  ✅ 统一分派 + 重构单发      compute_penetration / resolve_penetration_v2（跳弹阈值改 v_imp）
步骤⑥  ✅ 数据模型扩展            AmmoConfig 追加弹杆字段 + 内置 M829A3/3BM42 补 A-T 参数
步骤⑦  ✅ 命中链接线              engagement._resolve_hit 传 model/弹杆参数/靶参数/alpha
步骤⑧  ✅ 测试 + 基准             tests/test_penetration_algorithms.py（34 用例）+ 全量回归绿
步骤⑨  ✅ 已实现（2026-08-18）长杆跳飞判据  Rosenberg 式(60)（§5.7）：interface_velocity + resolve_penetration_v2，半流体段速度相关 βc，刚性段回退固定阈值
步骤⑩  📐 设计定稿（文档 · 待实现）标定锚  pen_anchor（§5.8）：P_eff = P × (pen_mm_rha/P_RHA(v0))；按用户要求仅留文档
步骤⑪  📐 设计定稿（文档 · 准备后续）标定工具  docs/penetration-calibration-tool-design.md（forward/fit/compare）
```

> **落地注记**：内置 M829A3 现走 A-T（0m≈523mm，物理模型，未启用比例锚，弱于标称 700mm）。
> 步骤⑨ 长杆跳飞判据（Rosenberg 式(60)）**已实现**：高速临界角≈75° 与既有硬阈值自然衔接、
> 低速段 βc 降到 ~47°，DeMarre 全口径弹不受影响。步骤⑩ 标定锚（`pen_anchor`）与
> 步骤⑪ 辅助标定工具按用户要求**仅留设计档**，择期实施。

---

## 附录：公式与参考

**综述原文公式映射**（焦文俊、陈小伟 2019）：
| 本文 | 综述 | 内容 |
|------|------|------|
| §5.2 ① | 式(13)~(16) | A-T 控制方程组 |
| §5.2 ② | 式(17) | 界面侵彻速度 u 与临界速度 Vc |
| §5.2 ③ | §4.2.1 | 两种终止情形（Rt>Yp / Rt<Yp） |
| §5.3 | 式(38)(39)、σy=4.2BHN | Yp/Rt 标定 |
| §5.4 | 式(3) | 长径比效应经验式 |
| §5.2 ④ | 式(8) | 流体动力学极限 P/L = 1/μ |
| §5.7 | 式(59)(60)(61) | 长杆跳飞判据（Tate/Rosenberg/最小斜角） |

- 焦文俊、陈小伟《长杆高速侵彻问题研究进展》，力学进展 2019, 49(1)，DOI 10.6052/1000-0992-17-021
  （[期刊页](https://pubs.cstam.org.cn/article/doi/10.6052/1000-0992-17-021)）
- Lanz & Odermatt《Penetration limits of conventional large caliber anti-tank guns / kinetic energy projectiles》
- War Thunder 开发公告《Improved Calculation of Armour Penetration》（[链接](https://warthunder.com/en/news/6010-development-improved-calculation-of-armour-penetration-in-the-game-en)）
- 本项目 `docs/ammunition-expansion-design.md` §五、`docs/phase3-combat-design.md` §四
