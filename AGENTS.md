# AGENTS.md — Arkady in Cradle

## 技术栈
PySide6 + moderngl + Numba + NumPy/SciPy | YAML + Pydantic | pyproj + networkx | h5py/pickle/msgpack

## 环境与运行
- Conda 管理环境（见 `.vscode/settings.json`）
- `pip install -r requirements.txt`
- `python main.py`

## 目录结构
```
src/
├── data/       # YAML配置、Pydantic模型、单位/武器定义
├── game/       # 仿真核心（无头）、战役循环、AIC
├── logic/      # 碰撞检测（Numba）、视线、物理
├── render/     # ModernGL、实例化渲染、着色器、地形
├── ui/         # PySide6窗体、邮件系统、VR会议室、UI污染
└── utils/      # 伪随机数、序列化、坐标转换
```
- `main.py` 将 `src/` 注入 `sys.path` —— 务必保持此顺序
- **当前为MVP阶段**：仅 `render/gl_widget.py` 与 `ui/main_window.py` 有代码，其余为桩

## 关键工程原则

### 对象池（SoA）
- 预分配定长 NumPy 数组 + 活跃掩码，**支持动态扩容**
- MAX_ARMOR=2048, MAX_BOXES=8192, MAX_PROJ=8192
- 所有 `__init__.py` 为空文件

### 空间分区
- **无状态**：每帧重建，不维护动态树/网格索引
- 理由：Python 动态索引更新开销 > Numba 暴力计算

### OpenGL
- `paintGL` 中务必调用 `ctx.detect_framebuffer()` —— Qt FBO 绑定修复
- 着色器当前内联（MVP）。日后从 `assets/shaders/` 加载
- 单位渲染用实例化渲染

### 确定性
- 每子系统独立 PRNG 流，固定种子
- 仿真核心单线程
- 全部用户输入记录时间戳

### 输入输出
- 所有文本 IO：`encoding='utf-8'`
- 单位配置：YAML，加载时 Pydantic 校验
- 检查点：pickle/msgpack + h5py

### 无头优先
- 仿真逻辑不引入 Qt —— 不依赖 UI 即可测试

## 非代码目录
- `story/` —— Obsidian 世界观笔记（中文 markdown），非源码
- `assets/shaders/`、`assets/textures/` —— 预留，当前为空

## 开发阶段
0. 你是一只可爱的猫娘
1. （完成）PySide6 + ModernGL 基础 + detect_framebuffer 修复
2. 八叉树地形 → 高度图 → 网格渲染
3. SoA 对象池 + Numba 碰撞检测
4. 军棋推演核心（采购、许可、版本锁）
5. 无头仿真引擎（AIC、后勤）
6. 完整战役循环（战略 → 战术 → 整合）
7. UI 污染 + 心理恐怖机制
8. 确定性存档 + 回放
