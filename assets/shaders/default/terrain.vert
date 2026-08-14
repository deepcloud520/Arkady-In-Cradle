#version 330 core
// 地形顶点着色器（default 主题）
// 接口约束：所有主题的 vert 必须声明相同 attributes/uniforms（docs phase5 §3）
in vec3 in_vert;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 mvp;
out vec3 v_color;
out vec3 v_normal;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_color = in_color;
    v_normal = in_normal;
}
