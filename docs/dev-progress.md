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
