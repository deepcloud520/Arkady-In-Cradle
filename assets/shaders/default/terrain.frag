#version 330 core
// 地形片元着色器（default 主题：现实军事风格）
// 半球环境光 + 方向光，双面法线翻转（2026-08-11 升级，替代灰白常量环境光）
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;

uniform vec3 u_light_dir;        // 世界空间方向光（指向光源）
uniform vec3 u_light_color;      // 光源颜色
uniform vec3 u_sky_color;        // 天光色（半球环境光：朝上面）
uniform vec3 u_ground_color;     // 地光色（半球环境光：朝下面）
uniform float u_ambient_strength;  // 环境光强度（0~1）

void main() {
    vec3 n = normalize(v_normal);
    // 双面：按三角形朝向翻转法线，背面也正确受光
    if (!gl_FrontFacing) {
        n = -n;
    }
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    // 半球环境光：由法线 Y 分量在「地光 ↔ 天光」间插值
    float up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(u_ground_color, u_sky_color, up);
    vec3 color = v_color * (u_ambient_strength * ambient
                            + (1.0 - u_ambient_strength) * diff * u_light_color);
    frag_color = vec4(color, 1.0);
}
