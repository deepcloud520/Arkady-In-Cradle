# 部件血量体系（Part Health）设计

> 2026-08-14 定稿。将「整车 `armor.health` + 部位 `box.active` 二值」两套割裂系统
> 统一为**「部件血量体系」**：每个碰撞箱有独立血量，部位毁伤从「一发击穿即失活」
> 升级为「击穿扣部位血，归零才失效」；整车存亡绑定 HULL 部位血量。
> 弹药殉爆改为向爆炸池提交小范围大威力爆炸，由统一区域毁伤结算。

## 一、动机（为什么改）

旧模型（AGENTS.md 3.4 之前）：
- `armor.health`：整车结构血量（默认 100），未击穿也扣，归零整车毁
- `box.active`：二值部位毁伤，**一发击穿直接失活**，无「部位受损但还能用」的过渡态

问题：
1. 两套血量/毁伤语义并存，割裂
2. 部位毁伤是二值开关，不符合「履带挨几发破片才断 / 炮塔敲几发才卡死」的真实累积
3. 弹药殉爆直接秒杀本车，不经过爆炸池，无法体现殉爆波及周围、泄压阀降低威力的语义

## 二、数据结构改动

### 2.1 BoxPool 新增部位血量 + 哨兵槽

```python
# BoxPool（MAX_BOXES=4096，实际分配 capacity+1，末位为哨兵槽）
self.hp      = np.zeros(cap + 1, dtype=np.float32)   # 当前部位血量
self.hp_max  = np.zeros(cap + 1, dtype=np.float32)   # 部位满血（spawn 时按 part 设定）
```

- **哨兵槽**：`index == capacity` 的槽恒 `hp=0`、`active=False`、不进 free_stack。
  `armor.hull_idx = -1`（未绑定 HULL）时，numpy fancy indexing
  `boxes.hp[armor.hull_idx]` 会读出数组**最后一个元素**——哨兵槽，天然为 0，
  避免负索引读垃圾值（numpy 负索引不报错）。

### 2.2 ArmorPool：health → hull_idx

```python
# ArmorPool：health 数组退役，改存「该装甲的 HULL 碰撞箱索引」
self.hull_idx = np.full(self.capacity, -1, dtype=np.int32)

def bind_hull(self, armor_idx: int, box_idx: int) -> None:
    self.hull_idx[armor_idx] = box_idx
```

- 血量**权威来源**是 `boxes.hp`；`armor.hull_idx` 只是指针。
- 读取整车血量 = `boxes.hp[armor.hull_idx[armor_idx]]`（标量）或
  `boxes.hp[armor.hull_idx]`（向量化 fancy indexing，AI/渲染/AI 全量读取用）。

### 2.3 部位满血表

```python
PART_MAX_HP = {
    PartType.HULL: 100.0,        # 接替原 armor.health 默认值 → AI/触发器阈值语义不变
    PartType.TURRET: 60.0,
    PartType.ENGINE: 50.0,
    PartType.TRACK: 40.0,
    PartType.AMMO_RACK: 25.0,
}
```

`BoxPool.spawn` 创建 box 时按 part 填 `hp_max`，`hp = hp_max`。

## 三、伤害语义（新版）

| 命中结果 | 扣谁的血 | 后果 |
|---------|---------|------|
| 未击穿（结构震荡） | HULL | 2~15/发，多次命中耗损整车 |
| 击穿 HULL | HULL | 15~50/发，归零 → 整车摧毁 |
| 击穿 TURRET | TURRET | 归零 → 炮塔失活 → 无法开火 |
| 击穿 ENGINE | ENGINE | 归零 → 引擎失活 → 失去机动 |
| 击穿 TRACK | TRACK | 归零 → 履带失活 → 减速 |
| 击穿 AMMO_RACK | AMMO_RACK | 归零 **或** cookoff 掷骰 → 殉爆 |

- **整车存亡 = HULL 血量**：HULL 归零即 `_destroy_armor`（取代原 `armor.health<=0`）。
- 区域毁伤（球形/破片锥）只扣 **HULL** 血（整车整体承伤；部位级破片分摊不做）。

## 四、弹药殉爆（cookoff）

击穿 AMMO_RACK 后：
1. `damage.py` 的 cookoff 掷骰（`FIRE_BASE + COOKOFF_BASE`）命中，**或** AMMO_RACK 血归零
   → 触发殉爆
2. 殉爆 **不再直接 `_destroy_armor`**，改为向爆炸池提交一发**小范围大威力 SPHERE 爆炸**：

```python
submit_damage(
    kind=DamageKind.SPHERE,
    center=本车位置,
    radius=8.0,                          # 小范围
    pen_mm=300.0 * cfg.ammo_cookoff_power_mult,   # 大威力（本车 HULL=100 必归零）
    target_idx=本车,
)
```

3. Tick 末 `_resolve_damage_pool` 统一结算：本车 HULL 归零 → 摧毁；半径内周围单位受波及
   （体现殉爆波及友军）。
4. **泄压阀（blow-out panel）**：单位数据预留 `ammo_cookoff_power_mult` 字段（默认 1.0）。
   当前不实现泄压逻辑；未来该字段 < 1 时本车可能不殉爆（HULL 不归零但炮塔已炸飞）。

## 五、波及文件清单

| 文件 | 改动 |
|------|------|
| `entities/pools.py` | BoxPool 加 hp/hp_max + 哨兵槽；ArmorPool health→hull_idx + bind_hull |
| `combat/damage_submission.py` | `resolve_area_damage` 签名：`armor_health`→`box_hp`+`armor_hull_idx`（原地扣 HULL） |
| `combat/engagement.py` | 单发/攻顶扣部位血；`_apply_part_damage` 重写；cookoff 走爆炸池；摧毁判据=HULL |
| `combat/damage.py` | docstring 更新（逻辑不变，`health_dmg` 输出复用） |
| `simulation/ai.py` | `a.health`→`boxes.hp[a.hull_idx]`（向量化） |
| `simulation/serialization.py` | armor `hull_idx`、boxes `hp/hp_max` 存档/读档 + 旧档兼容 |
| `render/adapter.py` | `armor_health = boxes.hp[armor.hull_idx]` |
| `narrative/triggers.py` | 触发器血量判断读 HULL 血 |
| `ui/main_window.py` | spawn 参数、HULL box 绑定 |
| 测试 | test_damage_submission / test_phase3_integration / test_phase6_ai / test_sfm_sal / test_entities |

## 六、验收标准

- 未击穿命中：HULL 血微降，多次命中可耗损整车
- 击穿 TURRET：TURRET 血归零 → 失活 → 无法开火（车还活着）
- 击穿 AMMO_RACK 触发 cookoff：本车 + 半径内单位受殉爆爆炸波及，本车必毁
- HULL 归零：整车摧毁（`_destroy_armor`）
- 旧存档读入：缺 `hull_idx`/`hp` 字段保持构造默认，不崩溃
- 全量测试通过（402 → 新增/更新后全绿）
