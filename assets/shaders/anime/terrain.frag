#version 330 core
// 地形片元着色器（anime 主题：日系卡通渲染）
// 设计见 docs/anime-theme-design.md（Phase 5d）
//   1. Cel shading：漫反射量化 3 级（暗/中/亮），smoothstep 窄过渡带防抖
//   2. 暗部天光补色：阴面染蓝紫（u_shadow_tint），日系特征
//   3. 描边：法线边缘 + 平方锐化（fwidth）
//        - 深度通道（fwidth(NDC z)）实测 early-z 失效 → u_depth_edge_scale 默认 0
//        - 平方锐化把地形网格微变压到阈值下、桥/结构棱线保留
//   4. 大气透视：远景染天光（u_fog_color）
// 约束（重要）：fwidth 必须在 main() 开头无条件调用（2×2 quad 一致执行），
// 不能放进任何 if 分支，否则结果未定义。
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_ambient_strength;
// cel shading
uniform float u_shadow_threshold;
uniform float u_highlight_threshold;
uniform vec3 u_shadow_tint;
// 粉彩
uniform float u_saturation;
uniform float u_brightness;
// 描边
uniform vec3 u_outline_color;
uniform float u_outline_threshold;
uniform float u_depth_edge_scale;
uniform float u_normal_edge_scale;
// 大气透视
uniform float u_near;
uniform float u_far;
uniform float u_fog_near;
uniform float u_fog_far;
uniform vec3 u_fog_color;
uniform float u_fog_strength;
// 调试（默认 0，不影响生产）：1=可视化 d_edge 2=可视化 n_edge 3=可视化 edge
uniform float u_debug_edge;

void main() {
    // 无条件求导（fwidth 约束：必须在分支前调用）
    // 注意：深度通道（fwidth(gl_FragCoord.z)）在 early-z 下实测恒 ≈0
    // （背景像素被深度测试提前丢弃 → 2×2 quad 不完整 → 导数退化），
    // 故 depth_scale 默认 0；保留通道代码供未来 G-Buffer 方案（可调描边宽度）。
    float d_edge = fwidth(gl_FragCoord.z) * u_depth_edge_scale;
    vec3 n_raw = normalize(v_normal);
    float n_edge_raw = (fwidth(n_raw.x) + fwidth(n_raw.y) + fwidth(n_raw.z))
                       * u_normal_edge_scale;
    // 平方锐化：弱边缘（地形网格法线微变 ~0.02~0.04）被压缩，
    // 强棱线（桥/结构 ~0.06+）保留 → 地面干净、结构描边清晰。
    float n_edge = n_edge_raw * n_edge_raw;

    vec3 n = n_raw;
    if (!gl_FrontFacing) {
        n = -n;
    }

    // ── 粉彩化（保留 v_color 色相，仅调饱和/亮度） ──
    float lum = dot(v_color, vec3(0.299, 0.587, 0.114));
    vec3 base = mix(vec3(lum), v_color, u_saturation) * u_brightness;

    // ── Cel shading：3 级色阶 ──
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    float t = 0.04;   // 过渡带半宽
    float band = 0.0;
    band += smoothstep(u_shadow_threshold - t, u_shadow_threshold + t, diff);
    band += smoothstep(u_highlight_threshold - t, u_highlight_threshold + t, diff);
    float lit = band * 0.5;   // {0, 0.5, 1.0}

    // 暗部天光补色（阴面染蓝紫）
    vec3 shaded = mix(base, base * u_shadow_tint, 1.0 - lit);

    // 半球环境光 + 量化漫反射
    float up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(u_ground_color, u_sky_color, up);
    vec3 light = u_ambient_strength * ambient
               + (1.0 - u_ambient_strength) * lit * u_light_color;
    vec3 color = shaded * light;

    // ── 描边（深度 + 法线双通道） ──
    float edge = max(d_edge, n_edge);
    float outline = smoothstep(u_outline_threshold, u_outline_threshold + 0.02, edge);
    color = mix(color, u_outline_color, outline);

    // ── 大气透视（远景染天光） ──
    float lin = (2.0 * u_near * u_far)
              / (u_far + u_near - (2.0 * gl_FragCoord.z - 1.0) * (u_far - u_near));
    float fog = smoothstep(u_fog_near, u_fog_far, lin);
    color = mix(color, u_fog_color, fog * u_fog_strength);

    // 调试可视化（u_debug_edge>0 时输出边缘值，灰度 = clamp(show*10, 0, 1)）
    if (u_debug_edge > 0.5) {
        float show;
        if (u_debug_edge < 1.5) {
            show = d_edge / max(u_depth_edge_scale, 1e-5);
        } else if (u_debug_edge < 2.5) {
            show = n_edge_raw;   // 平方前原始法线边缘
        } else {
            show = edge;
        }
        frag_color = vec4(vec3(clamp(show * 10.0, 0.0, 1.0)), 1.0);
        return;
    }

    frag_color = vec4(color, 1.0);
}
