"""临时：测量地形加载各阶段耗时（生成 vs navmesh）。运行后删除。"""
import cProfile
import io
import pstats
import time

from tests.terrain_factory import TerrainFactory, TerrainFactoryConfig


def main() -> None:
    # 1. 地形生成（测试工厂 fbm 路线）
    t0 = time.perf_counter()
    tm = TerrainFactory.default()
    t1 = time.perf_counter()
    print(f"[工厂路径] TerrainFactory.default() : {t1 - t0:.3f}s")

    # 1.1 fBm 高度图单独计时
    from Airis.lulilo.terrain.heightmap import generate_fbm

    cfg = TerrainFactoryConfig()
    t0 = time.perf_counter()
    generate_fbm(cfg.width, cfg.height, cfg.amplitude, cfg.noise_scale, cfg.octaves, cfg.seed)
    t1 = time.perf_counter()
    print(f"[子步骤]   generate_fbm          : {t1 - t0:.3f}s")

    # 2. 渲染网格提取（main.py initializeGL 实际调用）
    t0 = time.perf_counter()
    rd = tm.to_render_data()
    t1 = time.perf_counter()
    print(f"[main 路径] to_render_data()      : {t1 - t0:.3f}s (顶点 {len(rd['ground_vertices']) + len(rd['bridge_vertices'])})")

    # 3. navmesh 烘焙（main.py 当前并不调用，仅作对比）
    from Airis.lulilo.navmesh._baker import Baker

    t0 = time.perf_counter()
    ng = Baker(tm.heightfield).bake()
    t1 = time.perf_counter()
    print(f"[非 main 路径] Baker.bake()       : {t1 - t0:.3f}s (多边形 {len(ng.polys)})")

    # 4. cProfile 细粒度：工厂 fbm() 内部热点
    pr = cProfile.Profile()
    pr.enable()
    TerrainFactory.default()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(18)
    print("\n===== cProfile: TerrainFactory.default() 热点 =====")
    print(s.getvalue())


if __name__ == "__main__":
    main()
