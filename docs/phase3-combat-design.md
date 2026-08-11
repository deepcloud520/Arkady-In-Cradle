# Phase 3 战斗系统设计 —— 散布·命中·毁伤模型

> 状态：**设计定稿，待实现**（2026-08-11）
> 对应 AGENTS.md Phase 3（战斗/后勤/模拟循环）+ 参考项目 stochastic-warfare 的学习吸收
> 设计来源：`Reference/` 及 `stochastic-warfare/combat/`（PolyForm Noncommercial 1.0.0，仅参考思路，独立实现）

## 一、设计原则

在「物理碰撞 + 真实弹道」的主架构上，**吸收 sw 的统计命中模型作为散布注入器**：

- 物理层决定"弹道落点、命中部位、穿透判定"
- 统计层决定"这发炮的落点散布有多大、会不会偏"（由技能/运动/能见度/车辆特性修正）
- 一句话：**sw 的公式算 sigma（散布），我们的物理引擎做判定**

## 二、散布模型（命中概率内核）

### 2.1 基础散布

源自炮口散布（base_accuracy_mrad）与距离的线性关系：

```
sigma_m = base_accuracy_mrad * 0.001 * range_m        # 1σ 散布半径 (m)
```

- `base_accuracy_mrad`：武器固有精度（mrad）。M256 ≈ 0.2，2A46M ≈ 1.0
- 距离越远 sigma 越大 → P(hit) 越低（服从高斯面积比模型）

### 2.2 修正因子（sw 学习的核心）

每个因子在 [0,1] 区间对 P(hit) 做乘法修正，最终注入 sigma：

| 因子 | 公式 | 含义 |
|------|------|------|
| **乘员技能** | `0.5 + 0.3 * crew_skill` | 熟练炮手提升命中 |
| **目标运动** | `1.0 - 0.15 * (target_speed/10)`，下限 0.5 | 目标移动越难命中 |
| **射手运动** | `1.0 - 0.25 * (shooter_speed/10)`，下限 0.3 | 行进间射击更难（火控稳定不足） |
| **能见度** | `0.3 + 0.7 * visibility` | 0~1，影响观测与瞄准 |
| **姿态** | DUG_IN 0.6 / FORTIFIED 0.4 / DEFENSIVE 0.85 / MOVING 1.0 | 依托掩体的暴露面积 |
| **武器状况** | `0.5 + 0.5 * weapon_condition` | 装备退化（sw 的 EquipmentItem） |
| **位置不确定度** | `1.0 - 0.01 * uncertainty_m`，下限 0.3 | 目标位置估计误差 |
| **车辆特性（我们新增）** | 火控稳定系数 × 炮口修正 | M1A2 先进 FCS > T-90（见 2.4） |

### 2.3 命中判定流程（我们的实现方式）

```
statistical: sigma ← base_accuracy × 距离 × Π(修正因子)
     ↓
physical:    落点 = 瞄准点 + 高斯散布(sigma)
     ↓
physical:    真实 胶囊体 vs AABB 碰撞 → 命中部位 (PartType)
     ↓
statistical: 若 sigma 过大导致落点超出目标 → 未命中（无需物理判定）
```

**关键设计**：不做"概率骰子"，而是**把修正因子换算成散布半径，让物理决定结果**。
这样技能高 → sigma 小 → 落点贴脸 → 物理必然命中；技能低 → sigma 大 → 落点偏 → 物理判定未中。可解释、可追溯、戏剧性强。

### 2.4 车辆特性修正（新增，sw 未覆盖）

| 车辆特性 | 对 sigma 的影响 |
|----------|----------------|
| **火控稳定** | 行进间射击惩罚按火控等级缩放（M1A2 稳定器好 → shooter_motion 惩罚减半） |
| **炮口精度** | 直接来自 `base_accuracy_mrad`（口径/炮管寿命） |
| **乘员训练等级** | `crew_skill`（M1A2 乘员 TRAINED≈0.9，T-90 BASIC≈0.7） |

## 三、跳弹与斜距效应

### 3.1 斜距效应（obliquity）

命中时目标装甲厚度按入射角折算：

```
armor_eff = armor_mm / cos(impact_angle)      # impact_angle 从法线起算
armor_eff *= armor_effectiveness               # 装甲类型修正（见 3.3）
```

- 正击（0°）→ armor_eff = armor_mm
- 斜击（60°）→ armor_eff = 2 × armor_mm（更难穿透）

### 3.2 跳弹判定（ricochet）

```
if |impact_angle| > 75°:  → 跳弹（未穿透，可能偏离继续飞行或碎甲）
if |impact_angle| > 60° 且 弹速低于阈值: → 高概率跳弹
```

- 跳弹是**物理判定**，落在我们的碰撞内核里，不需要概率表
- 跳弹弹丸可保留部分动能继续飞行（对周边单位二次威胁，叙事可用）

### 3.3 装甲类型有效性（ARMOR_EFFECTIVENESS）

| 装甲类型 | vs 动能弹 (KE) | vs 化学能 (HEAT) |
|----------|:--------------:|:----------------:|
| RHA | 1.0 | 1.0 |
| COMPOSITE（复合） | 1.5 | **2.5** |
| REACTIVE（反应装甲） | 1.0 | **2.0** |
| SPACED（间隙） | 0.9 | 1.3 |

> **复合装甲对 HEAT 有 ~2.5× 加成**——这是 M1A2 抗 Javelin/Kornet 的关键。映射到我们的 CSV：`Protection_Level` → ArmorType 枚举。

## 四、穿透判定（DeMarre 变体）

### 4.1 动能弹衰减

```
penetration = pen_mm_rha × (1 - drag_coeff × range/100000)^demare_exponent
            = pen_mm_rha × decay ^ 1.5
```

- 近距满穿深，远距衰减，`decay` 下限 0.3
- `pen_mm_rha`：0m 标称穿深（从弹药定义读）
- `drag_coefficient`：弹芯阻力（APFSDS 长杆弹 ≈ 0.1~0.2）

### 4.2 化学能弹（HEAT/ATGM）

```
penetration = pen_mm_rha（固定，不随距离衰减）
```

- 化学能穿深不受距离影响，但受 ERA/复合装甲克制（3.3 表格）

### 4.3 穿透判定

```
margin = penetration - armor_eff
penetrated = margin > 0
```

- `margin > 0` → 击穿，进入后效毁伤
- `margin ≤ 0` → 未击穿，仅累积结构损伤（health 微降 + 士气影响）

## 五、后效毁伤（behind-armor effects）

### 5.1 Overmatch（穿深余量）

```
overmatch = margin / max(1, armor_eff)，上限 3.0
```

- 余量越大，后效越惨烈（乘员阵亡概率越高）

### 5.2 毁伤结果（映射到 PartType）

| 事件 | 概率基础 | 对应部位 (PartType) |
|------|----------|--------------------|
| **崩落 (spall)** | 0.3 × (0.5 + 0.5×overmatch)，逐乘员 | 车体内 → HULL |
| **起火 (fire)** | 0.1 | ENGINE / TURRET |
| **弹药殉爆 (cookoff)** | 0.05 | AMMO_RACK |
| **结构损伤** | 必然（未击穿也累积） | 命中部位 |

- 乘员伤亡分级：`kia / critical / serious / minor`（阈值随 overmatch 上移）
- **殉爆 = 秒杀**（AMMO_RACK 被击穿 → 直接摧毁单位，最戏剧性的叙事事件）

### 5.3 与现有 SoA 的映射

```
armor.health  ← 结构损伤累积（血条）
box.active    ← 部位毁伤（炮塔废 → box 失活 → 单位失去开火能力）
armor 层      ← 用 PartType 关联：HULL 毁 → 无法移动；TURRET 毁 → 无法开火
```

## 六、Numba 落地（性能内核）

所有散布/穿透/斜距逻辑编译为纯 Numba 内核，向量化批量执行：

```python
@njit(cache=True, parallel=True)
def resolve_hit_dispersion(
    sigma: np.ndarray,           # (N,) 每发的散布半径（已含全部修正因子）
    aim_pos: np.ndarray,         # (N, 3) 瞄准点
    rng_state: np.ndarray,       # (N,) 流式随机数（RNGManager 传入）
    out_hit_pos: np.ndarray,     # (N, 3) 实际落点（含高斯散布）
    out_hit: np.ndarray,         # (N,) 是否命中（由上层碰撞判定细化）
):
    for i in prange(N):
        theta = rng[i] * 2π
        r = sigma[i] * sqrt(-2 * ln(rng[i+1]))   # Box-Muller 高斯
        out_hit_pos[i, 0] = aim_pos[i, 0] + r * cos(theta)
        out_hit_pos[i, 1] = aim_pos[i, 1] + r * sin(theta)
        out_hit_pos[i, 2] = aim_pos[i, 2]


@njit(cache=True, parallel=True)
def resolve_penetration(
    pen_rha: np.ndarray,         # (N,) 弹药标称穿深
    armor_eff: np.ndarray,       # (N,) 等效装甲（斜距+类型修正后）
    impact_angle: np.ndarray,    # (N,) 入射角
    range_m: np.ndarray,         # (N,) 射距
    is_heat: np.ndarray,         # (N,) 化学能标志
    out_margin: np.ndarray,      # (N,) 穿透余量（>0 击穿）
    out_ricochet: np.ndarray,    # (N,) 跳弹标志
):
    for i in prange(N):
        if abs(impact_angle[i]) > 75.0:
            out_ricochet[i] = True
            out_margin[i] = -armor_eff[i]
            continue
        if is_heat[i]:
            pen = pen_rha[i]
        else:
            decay = max(0.3, 1.0 - drag_coeff * range_m[i] / 100000.0)
            pen = pen_rha[i] * decay ** 1.5
        out_margin[i] = pen - armor_eff[i]
```

**性能目标**：2048 发同时开火 → 散布+穿透两内核合计 **< 1ms**（远低于 3ms 预算）。

## 七、与物理碰撞的分工

```
┌─────────────── 开火链 ───────────────┐
│ 1. 选择目标 + 瞄准点（AI/FireControl） │
│ 2. 计算 sigma（散布模型，§二）          │
│ 3. 生成实际落点（高斯散布，Numba）       │
│ 4. 抛射物按落点方向发射（物理弹道）       │
│ 5. 空间哈希碰撞检测 → 命中部位            │
│ 6. 斜距/跳弹/穿透判定（§三、四，Numba）   │
│ 7. 后效毁伤（§五） → 更新 health/box      │
│ 8. 命中事件 → 叙事层（可跳过战斗消费结果） │
└──────────────────────────────────────┘
```

### 7.1 碰撞检测：统一胶囊体模型（2026-08-11）

开火链第 5 步的物理碰撞**不再由弹丸类型/速度决定**算法，统一为「胶囊体 vs AABB」：

- 弹丸轨迹 = 半径 r 的球沿线段扫掠（胶囊体），同时覆盖「位移（防穿透）」与「半径（大弹头）」两个维度
- 数学等价：胶囊体 ∩ AABB ⟺ 线段 ∩ (AABB ⊕ 球r)，⊕ 后仍为 AABB → 半轴加 r 后复用 Slabs，成本 ≈ 线段检测
- 天然处理：高速细弹（防穿透）、低速大弹（半径覆盖）、**超高速大弹**（两者兼得）
- 多分辨率 Tick（地面 1s / 空中 0.1s / 子弹 0.01s）下无需任何判据切换

```python
# 内核签名（combat/collision.py）：AABB 膨胀 + Slabs，radius 缺省 0
@njit(cache=True)
def segment_aabb_single(p0, p1, box_c, box_h, radius=0.0) -> (bool, t):
    ex, ey, ez = box_h + radius      # 各轴膨胀弹丸半径（胶囊体 ⊕ AABB）
    ...                              # 标准 Slabs 线段裁剪
```

## 八、参数来源映射（CSV → 模型）

| 模型参数 | 来源 | 示例（M1A2 / T-90） |
|----------|------|---------------------|
| `base_accuracy_mrad` | 武器固有精度（估算，参考 sw M256=0.2, 2A46M=1.0） | 0.2 / 1.0 |
| `crew_skill` | 乘员训练（CSV 无此字段 → 从 Generation 估算） | 0.9 / 0.7 |
| `armor_type` | CSV `Protection_Level` → 枚举映射 | COMPOSITE / REACTIVE |
| `armor_front/side` | 从 sw 参考值或 RHA 等效估算 | 600/200 / 800/300 |
| `pen_mm_rha` | 弹药定义（内嵌 WeaponConfig 或独立表） | 700 / 650 |
| `drag_coefficient` | 弹芯类型（APFSDS 0.1~0.2） | 0.15 / 0.18 |

## 九、测试用例（M1A2 vs T-90）

- [ ] **近距正击**：500m 正面互射，双方均大概率击穿（验证 sigma 小 + 穿深足）
- [ ] **远距衰减**：3000m 正面，穿深衰减后 M1A2 仍可击穿 T-90（M829A4 优势）
- [ ] **斜击**：T-90 车体 45° 暴露 → armor_eff 上升，命中但不一定击穿
- [ ] **跳弹**：60°+ 入射 → 大概率跳弹，弹丸继续飞行
- [ ] **行进间射击**：M1A2（稳定器好）vs T-90（惩罚大）→ 命中率差异可感知
- [ ] **部位毁伤**：命中 TURRET → 失去开火能力；命中 AMMO_RACK → 殉爆秒杀
- [ ] **性能**：2048 对同时交火，散布+穿透内核 < 1ms

---

> 猫猫注释：sw 的统计命中模型是"骰子决定命运"，我们把它降格为"决定散布大小"，把"命运"交还给物理碰撞——这样既有军事模拟的合理性，又有玩家看得见的戏剧性喵。
