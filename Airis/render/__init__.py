"""render 层：渲染官方实现（moderngl 绘制，游戏实际使用）。

Airis 是无头推演引擎，渲染作为引擎外的独立子包提供可视化：
- camera.py    ：轨道相机（透视投影 + 自由旋转/缩放）
- gl_widget.py ：QOpenGLWidget 2.5D 战场视图
- hud/         ：基于 moderngl 的 HUD / UI 覆盖层（render 子项，Phase 5）

依赖方向：render 依赖 engine（读取地形等状态），不反向依赖。
"""
from Airis.render.camera import OrbitCamera
from Airis.render.gl_widget import GLWidget

__all__ = ["OrbitCamera", "GLWidget"]
