"""GUI CPU 诊断：计时 _on_frame 模拟部分 vs paintGL 渲染部分。

用法：uv run python tests/diag_cpu.py
运行 3 秒（~180 帧），统计两类耗时分布，判断 CPU 占用来自逻辑还是渲染。
"""
from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Airis.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sim_times: list[float] = []
    render_times: list[float] = []
    total_times: list[float] = []
    frames = {"n": 0}
    t_start = time.perf_counter()

    # 计时模拟部分（包一层 step）
    orig_step = window.simulation.step

    def timed_step(dt):
        t0 = time.perf_counter()
        orig_step(dt)
        sim_times.append(time.perf_counter() - t0)

    window.simulation.step = timed_step

    # 计时渲染部分（包一层 paintGL）
    gl = window.gl_widget
    orig_paint = gl.paintGL

    def timed_paint():
        t0 = time.perf_counter()
        orig_paint()
        render_times.append(time.perf_counter() - t0)

    gl.paintGL = timed_paint

    def on_tick() -> None:
        frames["n"] += 1
        total_times.append(time.perf_counter() - t_start)

    def finish() -> None:
        app.quit()

    timer = QTimer()
    timer.timeout.connect(on_tick)
    timer.start(16)
    QTimer.singleShot(3500, finish)  # 3.5s 后结束
    app.exec()

    def stats(name: str, xs: list[float]) -> str:
        if not xs:
            return f"{name}: 无数据"
        import numpy as np

        a = np.array(xs) * 1000
        return (
            f"{name}: n={len(a)} mean={a.mean():.3f}ms p50={np.median(a):.3f} "
            f"p95={np.percentile(a, 95):.3f} p99={np.percentile(a, 99):.3f} max={a.max():.3f}ms"
        )

    print("=" * 70)
    print(stats("sim.step  ", sim_times))
    print(stats("paintGL   ", render_times))
    n = len(total_times)
    if n > 1:
        dur = total_times[-1] - total_times[0]
        print(f"帧间隔: {dur / max(1, n - 1) * 1000:.3f}ms/帧 (~{1000 / max(dur / max(1, n - 1), 1e-9):.0f} fps)")
    # 每帧总耗时（模拟 + 渲染最大值近似）
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
