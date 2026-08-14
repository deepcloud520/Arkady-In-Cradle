#version 330 core
// 地形片元着色器（constructivism 主题：苏式构成主义）
// 红黑配色 + 硬边阴影（step() 量化连续光照为 2 级）
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_ambient_strength;

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) {
        n = -n;
    }
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    // 硬边：光照量化为 3 级
    float band = step(0.4, diff) + step(0.8,diff);
    // 红黑几何色板：按朝向在「黑 ↔ 红」间切换
    float up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 geom = mix(vec3(0.10, 0.06, 0.05), vec3(0.80, 0.12, 0.10), up);
    // 保留少量原色灰度信息，避免纯色块过于单调
    float lum = dot(v_color, vec3(0.299, 0.587, 0.114));
    float lit = mix(u_ambient_strength, 1.0, band);
    frag_color = vec4(geom * (0.7 + 0.3 * lum) * lit, 1.0);
}
