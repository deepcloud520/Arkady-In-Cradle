# 穿深标定工具设计（辅助工具 · 设计档）

> 状态：**设计定稿（待实现）** · 仅设计档，不编写代码
> 关联：`docs/penetration-algorithms-design.md`（步骤⑪；§5.7 长杆跳飞判据 / §5.8 标定锚 / §5.3 强度标定）
> 定位：**dev-only 命令行工具**，供数据作者/平衡调参使用，不进游戏运行时热路径。

---

## 〇、定位与原则

1. **只编排既有内核**：复用 `damage.py` 的 `impact_velocity` / `penetration_demarre` /
   `penetration_at_erosion` / `penetration_lanz_odermatt` / `compute_penetration` /
   `TARGET_ARMOR_MODEL`，**不新增任何物理模型**。
2. **正反双向**：正向算「给定参数 → 穿深曲线」，反向标定「给定目标穿深 → 参数」。
3. **产物可控**：图表/报告一律走 `tests/output/`（AGENTS.md §4.5），不污染仓库。
4. **只读不写游戏数据**：工具输出 YAML 片段供人工粘贴，不直接改 `weapons.py` 内置数据。

---

## 一、输入

| 输入 | 来源 | 说明 |
|------|------|------|
| 弹药 | `WeaponRegistry.with_builtin()` / `from_yaml` | 按 `ammo_id` 查，或 CLI 直接给原始参数 |
| 靶 | `TARGET_ARMOR_MODEL[ArmorType]` | 按装甲类型查 ρt/Rt；或直接给 ρt/Rt |
| 靶强度换算 | Tate 1986 公式 | 给 `BHN + Et` → `σy=4.2BHN` → `Rt=σyt(2/3+ln(0.57Et/σyt))`（§5.3） |
| 场景 | CLI | `--ranges 0,500,1000,2000,4000`、`--angle 0,30,60`、`--armor-mm 600 --armor-type RHA` |
| 目标数据点 | CLI | `--target-pen 700 --at-range 0`（fit 模式用） |

---

## 二、输出

- **穿深曲线表**：`range → v_imp → P → margin(=P−armor_eff) → 是否击穿`。
- **校准结果**：`fit` 给出标定后的 `Yp`/`Rt` 或锚比例 `s`，并输出可直接粘贴的
  `AmmoConfig` YAML 片段（`penetrator_*` / `pen_anchor` / `velocity_alpha` 字段）。
- **对比表**（compare）：多弹药/多模型同靶并排。
- **可选图表**：`matplotlib` 折线（P–range 曲线），存 `tests/output/`。

---

## 三、三个模式

### 3.1 `forward`（正算穿深曲线）

流程：
```
for r in ranges:
    v_imp = impact_velocity(v0, α, r)
    P = compute_penetration(model, …)         # 按弹种自动选模型
    armor_eff = armor_mm / cos(angle) × ARMOR_EFFECTIVENESS[armor_type, is_heat]
    输出 (r, v_imp, P, P − armor_eff, 击穿?)
```
用途：调参看曲线形状（衰减斜率、临界速度截断、锚前后对比）。

### 3.2 `fit`（反标定）

| 子模式 | 反解目标 | 方法 | 约束 |
|--------|----------|------|------|
| `fit-anchor` | 锚比例 `s = pen_mm_rha / P_RHA(v0)` | 直接闭式（§5.8） | P_RHA(v0)>eps，否则 s=1 |
| `fit-rt` | 反解 `Rt` 使 `P(0m)==目标` | **二分**（P 随 Rt 单调递减） | 收敛 < 1mm；Rt > Yp（否则进刚性段） |
| `fit-yp` | 反解 `Yp` | 二分（P–Yp 有极值点，仅局部单调，见 §五） | 区间内单调；标注非全局 |
| `fit-multi`（进阶） | 多数据点（0m 与 2km 同时）→ 双自由度 `(Yp,Rt)` 或 `(α,Rt)` | 最小二乘 + 网格/梯度 | 不可行时报告残差，不静默 |

> `fit-rt` 示例（M829A3）：物理模型 0m≈523mm，要锚到 700mm →
> 反解 Rt ≈ 5.3~5.5 GPa（量级估计，实现后由工具二分确认），或 `fit-anchor` 直接得
> `s=1.3375`（见 §5.8 表）。

### 3.3 `compare`（多模型/多弹药对比）

- 同一靶下并排输出 DeMarre / A-T / LO 与**旧 DeMarre（decay^1.5）**四列，辅助迁移诊断
  （`docs/penetration-algorithms-design.md` §九 兼容表）。
- 支持多弹药：`--ammos m829a3,3bm42,57mm_apfsds`，输出「平衡对照表」。

---

## 四、数值方法（fit 用）

- **二分**（`fit-rt`）：`P(Rt)` 单调递减 → 区间 `[Rt_lo, Rt_hi]` 二分至 `|P−target| < 1mm`；
  稳健优先，不求快。
- **牛顿/割线**：可选加速，但需导数/差商，仅作优化不做默认。
- **多目标点**（`fit-multi`）：目标函数 = Σ(残差²)，网格粗扫 + 局部梯度细化；
  **报告最终残差**，拒绝静默收下不可行解。

---

## 五、已知数值注意点

1. **P–Yp 非单调**：综述 §5.2.2 指出侵彻深度-弹体强度曲线存在极值点（弹体强度「传递
   更高压力增深」与「导致弹尾减速减深」两种作用对立）→ `fit-yp` 必须限定搜索区间，
   且文档明示「仅局部单调」。
2. **v_imp ≤ Vc → P=0**：fit 目标值必须 > 0 才有解；`fit-rt` 反解时若目标穿深过高而
   Rt 下限已触及 Yp（刚性段），应报「目标不可达」而非硬凑。
3. **长径比效应缺失**：A-T 无法反映 L/D 效应（§5.4）→ `fit` 拟合的 Yp/Rt 只在单一
   长径比下有效；换长径比需重拟合，工具应在输出中注明「本拟合仅对 L/D=X 有效」。

---

## 六、CLI 约定（设计示意，不实现）

```text
python tools/penetration_calibrate.py forward   --ammo m829a3 --ranges 0,500,1000,2000,4000 [--angle 0] [--armor-mm 600 --armor-type RHA]
python tools/penetration_calibrate.py fit-anchor --ammo m829a3 --pen0 700
python tools/penetration_calibrate.py fit-rt     --ammo m829a3 --target-pen 700 --at-range 0 --Yp 3.0e9
python tools/penetration_calibrate.py fit-yp     --ammo m829a3 --target-pen 700 --at-range 0 --Rt 7.4e9 --Yp-lo 1e9 --Yp-hi 5e9
python tools/penetration_calibrate.py compare    --ammos m829a3,3bm42,57mm_apfsds --ranges 0,1000,2000,4000 --armor-type RHA
```

- 复用 `WeaponRegistry`（内置 + YAML）；图表产物存 `tests/output/`。

---

## 七、不做什么

- 不改游戏运行时热路径（工具只读内核，不进 `Airia/lulilo`）。
- 不做 GUI / 数据库 / 网络。
- 不新增物理模型（跳飞判据、锚、L/D 效应等仍属主设计文档，工具仅调用其内核）。

---

## 八、实施顺序建议

```
① forward 骨架（读 registry → 逐 range 算 P 打表）
② fit-anchor（§5.8 闭式，最简）
③ fit-rt 二分 + 输出 YAML 片段
④ compare 四模型并排（含旧 DeMarre 迁移诊断）
⑤ （进阶）fit-multi 双自由度 + 残差报告
```

> 工具落地依赖主设计文档 §5.7/§5.8 先实现（`interface_velocity` 公共辅助、
> `pen_anchor` 字段、`anchor_s` 入核）。当前三者均为设计定稿、未写代码。
