"""render 层：渲染官方实现（moderngl 绘制，游戏实际使用）。

Airia 是无头推演引擎，渲染作为引擎外的独立子包提供可视化：
- camera.py    ：轨道相机（透视投影 + 自由旋转/缩放）
- gl_widget.py ：QOpenGLWidget 2.5D 战场视图（Phase 5 接入主题系统 + WorldAdapter）
- adapter.py   ：WorldAdapter 渲染数据源抽象（Phase 5 新增）
- theme.py     ：ThemeManager 着色器主题系统（Phase 5 新增）
- hud/         ：基于 moderngl 的 HUD / UI 覆盖层（render 子项，Phase 5c）

依赖方向：render 依赖 engine（读取地形等状态），不反向依赖。
"""
from Airia.render.adapter import UnitRenderStatus, WorldAdapter
from Airia.render.camera import OrbitCamera
from Airia.render.gl_widget import GLWidget
from Airia.render.picking import Picker
from Airia.render.theme import Theme, ThemeManager

__all__ = [
    "OrbitCamera",
    "GLWidget",
    "Theme",
    "ThemeManager",
    "WorldAdapter",
    "UnitRenderStatus",
    "Picker",
]
