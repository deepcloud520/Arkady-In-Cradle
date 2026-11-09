"""2D 结构连续倒塌演示（纯演示，验证 AIC 的「支撑图 + 悬挑极限 + 塌落成废墟」思路）。

不依赖引擎任何模块，单文件可跑：

    .venv\\Scripts\\python.exe demo\\collapse_2d.py

模型（与 docs 讨论一致，见对话里的三档 P0/P1/P2）：
- 每个格子 = 一个结构体素（地形 / 柱 / 楼板 / 废墟）。
- 支撑 = 0-1 BFS：竖向贴合不耗代价，横向每跨一格耗 1，累计 ≤ CANTILEVER 才算
  有支撑（= 梁的悬挑能力）。地面是根。
- 失稳 = BFS 到不了的实体格 → 本帧坍落，材料不消失：同列底部生成废墟，堆高封顶。
- 单位：脚下支撑消失 → 自由落体 → 落差 > 2 格判定阵亡。

输出：demo/output/collapse_steps.png（3×2 拼图）+ 单帧 PNG。
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 中文字体（Windows 自带；缺失时退回 DejaVu，标题会显示豆腐块）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Polygon
from PIL import Image

logger = logging.getLogger("collapse_2d")

OUT_DIR = None  # 在 main 里按脚本位置解析

# --- 材质 -------------------------------------------------------------------
EMPTY, GROUND, COLUMN, SLAB, RUBBLE = 0, 1, 2, 3, 4

COLORS = {
    EMPTY: "#0f1115",
    GROUND: "#3a3f46",
    COLUMN: "#8d8f96",
    SLAB: "#b9bcc4",
    RUBBLE: "#5c5a55",
}
CRITICAL = "#e05a4a"      # 失去支撑（尚未坍落）
UNIT_ALIVE = "#4ab85a"
UNIT_DEAD = "#e05a4a"
HEAT = LinearSegmentedColormap.from_list(
    "cantilever", ["#2a6fdb", "#4ad2c8", "#e8d24a", "#e05a4a"]
)

# 悬挑极限：横向最多离竖向支撑几格（≈ 混凝土梁 3~4 m）
CANTILEVER = 4
MAX_PASSES = 24
RUBBLE_CAP = 3


# ---------------------------------------------------------------------------
# 结构模型
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    x: float
    y: float
    floor: int
    alive: bool = True
    falling: bool = False


@dataclass
class World:
    w: int = 30
    h: int = 20
    grid: np.ndarray = field(init=False)
    units: list[Unit] = field(default_factory=list)
    collapsed_cells: int = 0
    passes: int = 0

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.h, self.w), dtype=np.int8)
        self._build()

    # -- 建造 ---------------------------------------------------------------
    def _build(self) -> None:
        g = self.grid
        g[0, :] = GROUND                       # 地坪
        cols = [4, 9, 14, 19, 24]
        floors = [5, 9, 13, 16]                # 楼板/屋面板高度（层索引）
        for x in cols:
            g[1:17, x] = COLUMN
        for y in floors:
            g[y, cols[0]:cols[-1] + 1] = SLAB  # 柱顶到柱顶连成楼板
        # 侧向剪力墙（让端跨更强，演示"不是所有跨都同时塌"）
        for x in (cols[0], cols[-1]):
            g[1:17, x] = COLUMN
        # 单位：地面 1 个 / 二层 1 个 / 屋面 1 个
        self.units = [
            Unit(x=27.5, y=1.0, floor=0),
            Unit(x=13.5, y=6.0, floor=2),
            Unit(x=18.5, y=17.0, floor=4),
        ]

    # -- 破坏 ---------------------------------------------------------------
    def blast(self, cx: float, cy: float, r: float) -> list[tuple[int, int]]:
        """圆形爆破：去掉半径内所有实体格（不含地坪）。"""
        removed: list[tuple[int, int]] = []
        for y in range(1, self.h):
            for x in range(self.w):
                if self.grid[y, x] == EMPTY:
                    continue
                if (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r:
                    self.grid[y, x] = EMPTY
                    removed.append((x, y))
        return removed

    # -- 支撑判定（0-1 BFS）-------------------------------------------------
    def support_distance(self) -> np.ndarray:
        """每格到「竖向支撑」的横向步数（= 悬挑用量；地面/柱为 0）。

        竖向（上下相邻）代价 0：柱/墙的堆叠；
        横向（左右相邻）代价 1：楼板/梁的悬挑，累计 ≤ CANTILEVER。
        不可达（空心或超限）为 int32 上限。
        """
        h, w = self.h, self.w
        solid = self.grid != EMPTY
        dist = np.full((h, w), np.iinfo(np.int32).max, dtype=np.int32)
        dq: deque[tuple[int, int, int]] = deque()

        for x in range(w):                      # 地面 = 根
            if solid[0, x]:
                dist[0, x] = 0
                dq.append((0, x, 0))

        while dq:
            y, x, d = dq.popleft()
            if d != dist[y, x]:
                continue
            # 竖向：0 代价
            for ny in (y - 1, y + 1):
                if 0 <= ny < h and solid[ny, x] and d < dist[ny, x]:
                    dist[ny, x] = d
                    dq.appendleft((ny, x, d))
            # 横向：1 代价，受悬挑极限约束
            if d < CANTILEVER:
                for nx in (x - 1, x + 1):
                    if 0 <= nx < w and solid[y, nx] and d + 1 < dist[y, nx]:
                        dist[y, nx] = d + 1
                        dq.append((y, nx, d + 1))
        return dist

    def supported(self) -> np.ndarray:
        """返回同形状 bool 掩码：该格是否有通往地面的传力路径。"""
        solid = self.grid != EMPTY
        mask = (self.support_distance() <= CANTILEVER) & solid
        mask[0, :] |= solid[0, :]
        return mask

    def unstable_cells(self) -> list[tuple[int, int]]:
        sup = self.supported()
        out = []
        for y in range(1, self.h):
            for x in range(self.w):
                if self.grid[y, x] != EMPTY and not sup[y, x]:
                    out.append((x, y))
        return out

    # -- 一次坍落 -----------------------------------------------------------
    def collapse_pass(self) -> int:
        """失稳格 → 移除 → 同列底部堆废墟；返回本帧坍落格数。"""
        cells = self.unstable_cells()
        if not cells:
            return 0
        per_col: dict[int, int] = {}
        for x, y in cells:
            self.grid[y, x] = EMPTY
            per_col[x] = per_col.get(x, 0) + 1
        for x, n in per_col.items():
            top = int(np.max(np.nonzero(self.grid[:, x])[0])) + 1 if np.any(self.grid[:, x]) else 1
            h_rub = min(RUBBLE_CAP, max(1, int(round(n * 0.6))))
            for k in range(h_rub):
                y = top + k
                if y < self.h and self.grid[y, x] == EMPTY:
                    self.grid[y, x] = RUBBLE
        self.collapsed_cells += len(cells)
        self.passes += 1
        return len(cells)

    def settle(self) -> list[int]:
        """反复坍落直到稳定，返回每帧坍落格数。"""
        history = []
        for _ in range(MAX_PASSES):
            n = self.collapse_pass()
            history.append(n)
            if n == 0:
                break
        return history

    # -- 单位 ---------------------------------------------------------------
    def _support_cell(self, y: float) -> int:
        """脚下支撑格索引（feet 在 y，支撑 = 脚下第一格）。"""
        return int(math.floor(y - 1e-6))

    def resolve_units(self) -> None:
        for u in self.units:
            if not u.alive:
                continue
            gx = int(min(max(u.x, 0), self.w - 1))
            c = self._support_cell(u.y)
            below = self.grid[c, gx] if c >= 0 else GROUND
            if u.falling:
                start = u.y
                while u.y > 1.0:
                    u.y -= 0.4
                    c = self._support_cell(u.y)
                    if c < 0 or self.grid[c, gx] != EMPTY:
                        u.y = float(c + 1)
                        break
                if start - u.y > 2.0:
                    u.alive = False
                u.falling = False
            elif below == EMPTY:
                u.falling = True
                self.resolve_units()
            else:
                u.y = float(c + 1)

    def casualties(self) -> int:
        return sum(1 for u in self.units if not u.alive)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render(ax, world: World, title: str, critical: list[tuple[int, int]] | None = None,
           subtitle: str = "", heat: bool = False) -> None:
    ax.set_facecolor(COLORS[EMPTY])
    if heat:
        dist = world.support_distance()
        img = np.full(dist.shape, np.nan, dtype=np.float32)
        solid = world.grid != EMPTY
        img[solid] = np.clip(dist[solid], 0, CANTILEVER)
        ax.imshow(img, origin="lower", extent=(0, world.w, 0, world.h),
                  cmap=HEAT, vmin=0, vmax=CANTILEVER, interpolation="nearest",
                  zorder=1)
        ax.text(0.4, 1.4, "0 = 柱/地面", color="#2a6fdb", fontsize=9, zorder=6)
        ax.text(world.w - 6.4, 1.4, f"{CANTILEVER} = 悬挑极限", color="#e05a4a",
                fontsize=9, zorder=6)
    else:
        for y in range(world.h):
            for x in range(world.w):
                m = int(world.grid[y, x])
                if m == EMPTY:
                    continue
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=COLORS[m],
                                       edgecolor="none", linewidth=0))
    for x, y in critical or []:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=CRITICAL, alpha=0.85, edgecolor="none"))

    alive = [u for u in world.units if u.alive]
    dead = [u for u in world.units if not u.alive]
    if alive:
        ax.scatter([u.x for u in alive], [u.y for u in alive], s=120,
                   c=UNIT_ALIVE, marker="^", edgecolors="white",
                   linewidths=1.0, zorder=5)
    if dead:
        ax.scatter([u.x for u in dead], [u.y for u in dead], s=120,
                   c=UNIT_DEAD, marker="x", linewidths=1.8, zorder=5)

    ax.set_xlim(0, world.w)
    ax.set_ylim(0, world.h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a2d33")
    ax.set_title(title, color="#e8e8ea", fontsize=12, pad=24, loc="left")
    if subtitle:
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, color="#9aa0a6",
                fontsize=9, va="bottom", ha="left")


def frame(world: World, name: str, title: str, subtitle: str = "",
          critical: list[tuple[int, int]] | None = None, heat: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
    fig.patch.set_facecolor("#0f1115")
    render(ax, world, title, critical, subtitle, heat=heat)
    fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def montage(paths: list[tuple[str, str]], out_name: str) -> str:
    """把帧图拼成 3×2 联系表。"""
    imgs = [Image.open(p) for p, _ in paths]
    w, h = imgs[0].size
    cols, rows = 3, math.ceil(len(imgs) / 3)
    sheet = Image.new("RGB", (w * cols, h * rows), "#0f1115")
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    out = OUT_DIR / out_name
    sheet.save(out)
    return str(out)


# ---------------------------------------------------------------------------
# 演示脚本
# ---------------------------------------------------------------------------
def main() -> None:
    global OUT_DIR
    from pathlib import Path

    OUT_DIR = Path(__file__).resolve().parent / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    frames: list[tuple[str, str]] = []

    world = World()
    world.resolve_units()

    frames.append((frame(world, "01_initial", "① 初始：5 根柱 + 4 层板",
                         "绿=单位；地形=根，柱竖向堆叠，楼板靠柱悬挑"), "初始"))

    # --- 第一爆：中柱基础 ---------------------------------------------------
    removed = world.blast(14.5, 1.5, 1.6)
    logger.info("爆破删除 %d 格", len(removed))
    unstable = world.unstable_cells()
    frames.append((frame(world, "02_blast1", "② 炸掉中柱柱脚（红 = 已失去支撑）",
                         f"爆破移除 {len(removed)} 格 / 判出失稳 {len(unstable)} 格",
                         critical=unstable), "爆破"))

    # 第一次坍落：柱身一段塌落，展示"塌一点，不塌全楼"
    n1 = world.collapse_pass()
    history = world.settle()
    world.resolve_units()
    frames.append((frame(world, "03_collapse1", "③ 第一次坍落：只断一根柱线",
                          f"坍落 {n1} 格后稳定（后续 {history}）；楼板靠相邻柱悬挑撑住"),
                   "坍落1"))

    # 悬挑用量热力图：解释"为什么相邻跨没塌"
    near_limit = int(((world.support_distance() == CANTILEVER)
                      & (world.grid != EMPTY)).sum())
    frames.append((frame(world, "04_heat", "④ 悬挑用量：每格离竖向支撑几格",
                          f"黄/红 = 已到/接近悬挑极限（{near_limit} 格贴着极限）；再断一根就破",
                          heat=True), "支撑热图"))

    # --- 第二爆：相邻柱（展示连续倒塌放大）---------------------------------
    world.blast(9.5, 1.5, 1.6)
    world.blast(19.5, 1.5, 1.6)
    unstable = world.unstable_cells()
    frames.append((frame(world, "05_blast2", "⑤ 再炸两根柱（两侧同时断）",
                         f"失稳 {len(unstable)} 格 → 悬挑极限被彻底突破",
                         critical=unstable), "爆破2"))

    history = world.settle()
    world.resolve_units()
    frames.append((frame(world, "06_final", "⑥ 终局：大面积连续倒塌",
                         f"累计坍落 {world.collapsed_cells} 格 / 阵亡 {world.casualties()}/{len(world.units)} 人",
                         critical=None), "终局"))

    sheet = montage(frames, "collapse_steps.png")
    logger.info("联系表：%s", sheet)
    logger.info("单位状态：%s", [(u.floor, u.alive) for u in world.units])
    print(f"\n总坍落 {world.collapsed_cells} 格，单位阵亡 {world.casualties()}/{len(world.units)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
