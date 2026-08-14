# Phase 5d：Anime 主题（卡通渲染）算法设计

> 状态：**已实施**（2026-08-13）
> 对应 `docs/phase5-ui-pipeline-design.md` §3.5 / §6.4 的 Phase 5d
> 视觉定位：**Cel shading 2~3 级色阶 + 法线边缘描边 + 柔和粉彩色 + 大气透视**
> 前置：Phase 5a（主题系统）✅ / 5b / 5c ✅

---

## 〇、实施记录（2026-08-13，与设计的差异）

1. **深度通道实测失效（early-z）**：`fwidth(gl_FragCoord.z)` 在 Qt FBO + GL_LESS 深度测试下
   实测恒 ≈ 0——背景像素（clear depth=1.0）被早期深度测试在 fragment shader 之前丢弃 →
   2×2 quad 不完整 → 导数退化。设计稿推导的「NDC 深度常数性」在**有背景像素参与**的
   剪影处无法成立。**结论**：单 pass fwidth 无法做深度剪影，`u_depth_edge_scale` 默认 0
   （通道代码保留，供未来 G-Buffer 方案复用）。
2. **法线边缘平方锐化**：原始法线边缘 `n_edge_raw` 在平滑地形也有 ~0.02~0.045（中心差分
   法线跨格连续，但相邻三角形插值后屏幕导数仍有微变）→ 直接线性阈值会描出地面网格线。
   **平方锐化** `n_edge = n_edge_raw²` 把弱边缘（0.02²=0.0004）压到阈值之下、强棱线
   （0.1²=0.01）保留 → 地面干净、结构描边清晰（实测网格线消失）。
3. **cel 亮度峰实测 2 个**（非 3）：smoothstep 过渡带 + 大气透视 + 环境光让暗/中两档
   部分合并；视觉上亮/暗分明已满足需求，不强制 3 档。
4. **最终参数**见 §五（threshold=0.008 / depth=0 / normal=1.5）。
5. **验证**：`tests/smoke_theme_anime.py`（描边开关对照 diff_px≈10536 → 0.89%）；
   全量 299 passed；热切换 5 轮稳定。

---

## 一、目标与视觉定位

| 维度 | default（现实军事） | constructivism（构成主义） | **anime（日系动漫）** |
|------|--------------------|---------------------------|----------------------|
| 光照 | 连续 Phong（半球光 + 方向光） | 硬边 2 级（step 量化） | **3 级色阶（暗/中/亮）+ smoothstep 过渡** |
| 描边 | 无 | 无 | **法线边缘（平方锐化）** |
| 调色 | 自然色（高度渐变） | 红黑几何色板 | **粉彩（低饱和提亮）+ 暗部天光补色** |
| 大气 | 无 | 无 | **远景大气透视** |

核心观感：宫崎骏/吉卜力式——平滑地形干净无杂线、桥与建筑有清晰深色轮廓、阴影呈蓝紫色调（天光补色）。

---

## 二、渲染管线现状与约束（设计前提）

```
paintGL():
  fb = ctx.detect_framebuffer(); fb.use()     # 绑定 Qt FBO
  ctx.clear()
  vao_ground.render()    # 地形（theme program）
  vao_bridge.render()    # 桥/结构（同一 theme program）
  _render_hud()          # HUDOverlay（独立 program，屏幕空间圆点）
```

**关键约束**（决定算法选型）：

1. **单 pass 直渲 Qt FBO**：无离屏 FBO，无 G-Buffer。任何需要"采样上一帧/场景纹理"的后处理都会要求新增 pass。
2. **主题接口约束**：所有主题 `terrain.vert` 必须声明相同 attributes（`in_vert/in_color/in_normal`）与 uniforms（`mvp` + theme.toml `[uniforms]`）。frag 自由扩展，但**新 uniform 必须全部在 theme.toml 声明**（`ThemeManager.apply_uniforms` 自动注入，零代码改动）。
3. **桥与地形共用同一 program**：描边算法必须对两者统一生效（桥棱线 + 地形悬崖 + 结构剪影）。
4. **HUD 独立**：单位圆点不受主题影响（Phase 5d 不改 HUD，单位描边留待 HUD 主题化）。
5. **性能纪律**：60fps 目标，新增开销必须接近零。

---

## 三、核心算法设计

### 3.1 Cel shading：3 级色阶量化

**数学**：把连续漫反射系数 $d=\max(\hat n \cdot \hat L, 0)$ 量化到 3 档。

```glsl
// 硬阈值 + smoothstep 窄过渡带（防 aliasing 抖动）
float diff = max(dot(n, normalize(u_light_dir)), 0.0);
float t = 0.04;                                    // 过渡带半宽（~4%）
float band = 0.0;
band += smoothstep(u_shadow_threshold - t, u_shadow_threshold + t, diff);      // 暗→中
band += smoothstep(u_highlight_threshold - t, u_highlight_threshold + t, diff); // 中→亮
float lit = band * 0.5;                            // {0, 0.5, 1.0}
```

| band | 语义 | 视觉 |
|------|------|------|
| 0 | 暗部（背光/阴面） | 环境光主导，**偏天光补色** |
| 1 | 中调 | 半漫反射 |
| 2 | 亮部（受光面） | 满漫反射 |

**暗部天光补色**（日系特征）：暗部不是简单压暗，而是向 `u_shadow_tint`（蓝紫色）偏移——模拟阴面被天空散射光染色：

```glsl
vec3 shaded = mix(base, base * u_shadow_tint, 1.0 - lit);   // 暗部 → 染蓝紫
```

**半球环境光沿用**（与 default 一致，仅换色）：

```glsl
float up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
vec3 ambient = mix(u_ground_color, u_sky_color, up);
```

**最终合成**：

```glsl
vec3 light = u_ambient_strength * ambient
           + (1.0 - u_ambient_strength) * lit * u_light_color;
vec3 color = shaded * light;
```

> 注：`u_ambient_strength` 调高（~0.55）→ 亮调粉彩；`lit` 只贡献漫反射部分。

---

### 3.2 描边（Outline）—— 本主题核心

#### 方案对比

| 方案 | 实现 | 优点 | 缺点 | 结论 |
|------|------|------|------|------|
| **A. 单 pass 屏幕空间导数**（fwidth） | 片元着色器对深度/法线求屏幕导数 | 零管线改动、零 FBO、硬件免费、性能近零 | 边缘 ~1px、宽度不可调、阈值需实机标定 | ✅ **推荐（5d 实施）** |
| B. G-Buffer + Sobel 后处理 | 离屏 FBO 存颜色/法线/深度 → 全屏 quad 采样 Sobel | 宽度/阈值可控、效果最精细（业界标准） | 新增 pass 管线、与 Qt FBO 协调复杂、性能开销 | 📌 留档（未来精细描边） |

**选 A 的理由**：项目渲染管线刚定型（5a/5c），单 pass 架构简洁；2.5D 战术地图的描边需求（桥棱线 + 结构剪影 + 悬崖）用 fwidth 完全够；GLSL 的 `dFdx/dFdy/fwidth` 基于硬件 2×2 quad，**无需纹理采样**，天然绕过"Qt FBO 不可采样"的约束。

#### 方案 A 算法（实施版）：法线边缘 + 平方锐化

**通道 1 —— 深度边缘（剪影）—— ⚠️ 实测失效（2026-08-13）**

```glsl
float d_edge = fwidth(gl_FragCoord.z);     // NDC 深度（非线性）
```

设计稿推导「NDC 深度常数性」：透视下 $\frac{\partial z_{ndc}}{\partial p}$ 对平滑面 = 常数，远处地面不误描。
但**实测恒 ≈ 0**：Qt FBO + GL_LESS 深度测试下，背景像素（clear depth=1.0）被**早期深度测试
（early-z）在 fragment shader 之前丢弃** → 2×2 quad 不完整 → 剪影处的导数退化。
**结论**：单 pass fwidth 无法做深度剪影检测；`u_depth_edge_scale` 默认 0，代码保留供
未来 G-Buffer 方案复用。若要剪影描边，需方案 B（离屏 FBO 邻域采样）。

**通道 2 —— 法线边缘（内部棱线）+ 平方锐化**：

```glsl
vec3 n = normalize(v_normal);
float n_edge_raw = (fwidth(n.x) + fwidth(n.y) + fwidth(n.z)) * u_normal_edge_scale;
float n_edge = n_edge_raw * n_edge_raw;   // 平方锐化（关键）
```

- **地形**：中心差分平滑顶点法线 → 相邻像素法线连续，但插值后屏幕导数仍有微变
  （实测 `n_edge_raw` p50≈0.029 / p90≈0.065）→ 直接线性阈值会描出**地面网格线**
- **平方锐化**把弱边缘压缩（0.029²≈0.0008）到阈值之下，强棱线（0.065²≈0.0042+）保留
  → 地面干净、桥/结构棱线清晰（实测网格线消失）
- **桥/建筑**：平面片顶面↔侧面交界法线突变 → 描边 ✓
- **悬崖**：法线突变 → 描边 ✓

**组合与阈值**：

```glsl
float edge = max(d_edge * u_depth_edge_scale, n_edge);   // d_edge 默认关（depth_scale=0）
float outline = smoothstep(u_outline_threshold, u_outline_threshold + 0.02, edge);
color = mix(color, u_outline_color, outline);
```

> 实测边缘值量级（256×256 演示地形，960×640 视口）：`n_edge_raw` p50=0.029 / p75=0.043 /
> p90=0.065；平方后 p50≈0.0008（被阈值压住），棱线区 p95+≈0.005~0.07。
> 最终 `u_outline_threshold=0.008`，描边像素实测 ~0.89%（桥/结构轮廓）。

#### 已知局限与缓解（实机标定项）

| 风险 | 机理 | 缓解 |
|------|------|------|
| **深度剪影不可用**（early-z） | 背景像素被提前丢弃 → quad 不完整 → fwidth 退化 | 深度通道默认关闭；剪影需方案 B（G-Buffer） |
| 掠射角下地形网格描边 | 相机贴地时法线屏幕导数偏大 | `u_normal_edge_scale` 压低 / `u_outline_threshold` 调高 |
| 1px 边缘在低分辨率偏细 | fwidth 天然 1px | 可接受；或未来上方案 B |
| 阈值标定依赖场景尺度/相机 | 无解析最优解 | 交互标定脚本（`tests/diag_anime_edge.py` 输出边缘分布分位数） |

---

### 3.3 粉彩调色（色相保持，仅调饱和/亮度）

**设计决策**：消费渲染管线的 `v_color`（高度渐变地形色 / 结构材质色），**保留色相与高度信息**（区别于 constructivism 完全几何化），只做粉彩化——保持"绿色大地、蓝色水域"的自然可读性，同时柔和化：

```glsl
// 饱和度 + 亮度（Lab 近似：亮部提亮、暗部保留→粉彩高光感）
float lum = dot(base, vec3(0.299, 0.587, 0.114));
base = mix(vec3(lum), base, u_saturation) * u_brightness;
```

### 3.4 大气透视（可选，推荐包含）

日系动画的远景特征——远山染天光色：

```glsl
// 需从 gl_FragCoord.z 反算线性深度（需 near/far uniform）
float lin = (2.0 * u_near * u_far) / (u_far + u_near - (2.0 * gl_FragCoord.z - 1.0) * (u_far - u_near));
float fog = smoothstep(u_fog_near, u_fog_far, lin);
color = mix(color, u_fog_color, fog * u_fog_strength);
```

> 代价：+2 uniform（near/far 需由相机提供，主题系统目前无此通道）。若引入相机参数注入，需改 `ThemeManager`/`GLWidget` 写 uniform——**此为 5d 唯一可能的管线小改**。可降级为"固定近似"（不依赖相机，用 NDC 深度近似）或留档不做。

---

## 四、着色器完整伪代码

### `terrain.vert`（与 default 一致，接口约束）

```glsl
#version 330 core
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
```

### `terrain.frag`（anime，核心算法）

```glsl
#version 330 core
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_sky_color;        // 粉紫天光
uniform vec3 u_ground_color;     // 暖灰地光
uniform float u_ambient_strength;
// cel shading
uniform float u_shadow_threshold;    // 暗↔中 阈值（~0.18）
uniform float u_highlight_threshold; // 中↔亮 阈值（~0.55）
uniform vec3 u_shadow_tint;          // 暗部补色（蓝紫）
// 粉彩
uniform float u_saturation;
uniform float u_brightness;
// 描边
uniform vec3 u_outline_color;
uniform float u_outline_threshold;
uniform float u_depth_edge_scale;
uniform float u_normal_edge_scale;
// 大气透视（可选）
uniform float u_near;   uniform float u_far;
uniform float u_fog_near; uniform float u_fog_far;
uniform vec3 u_fog_color; uniform float u_fog_strength;

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) { n = -n; }

    // ── 粉彩化（保留色相） ──
    float lum = dot(v_color, vec3(0.299, 0.587, 0.114));
    vec3 base = mix(vec3(lum), v_color, u_saturation) * u_brightness;

    // ── Cel 3 级色阶 ──
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    float t = 0.04;
    float band = 0.0;
    band += smoothstep(u_shadow_threshold - t, u_shadow_threshold + t, diff);
    band += smoothstep(u_highlight_threshold - t, u_highlight_threshold + t, diff);
    float lit = band * 0.5;

    // 暗部天光补色
    vec3 shaded = mix(base, base * u_shadow_tint, 1.0 - lit);

    // 半球环境光 + 量化漫反射
    float up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(u_ground_color, u_sky_color, up);
    vec3 light = u_ambient_strength * ambient
               + (1.0 - u_ambient_strength) * lit * u_light_color;
    vec3 color = shaded * light;

    // ── 描边（法线边缘 + 平方锐化；深度通道默认关） ──
    float d_edge = fwidth(gl_FragCoord.z) * u_depth_edge_scale;   // early-z 下失效，scale=0
    float n_edge_raw = (fwidth(n.x) + fwidth(n.y) + fwidth(n.z)) * u_normal_edge_scale;
    float n_edge = n_edge_raw * n_edge_raw;                        // 平方锐化（关键）
    float edge = max(d_edge, n_edge);
    float outline = smoothstep(u_outline_threshold, u_outline_threshold + 0.02, edge);
    color = mix(color, u_outline_color, outline);

    // ── 大气透视 ──
    float lin = (2.0 * u_near * u_far)
              / (u_far + u_near - (2.0 * gl_FragCoord.z - 1.0) * (u_far - u_near));
    float fog = smoothstep(u_fog_near, u_fog_far, lin);
    color = mix(color, u_fog_color, fog * u_fog_strength);

    frag_color = vec4(color, 1.0);
}
```

---

## 五、theme.toml 参数表（最终值，2026-08-13 实机标定）

```toml
[theme]
id = "anime"
name = "日系动漫"
author = "Arkady in Cradle"
version = "1.0"
description = "日系卡通渲染：3 级 cel 色阶 + 法线描边 + 粉彩调色 + 大气透视"

[uniforms]
# 光照（沿用统一接口）
u_light_dir = [0.45, -0.60, 0.65]
u_light_color = [1.0, 0.98, 0.95]      # 柔和白光
u_sky_color = [0.82, 0.72, 0.95]        # 粉紫天光（§3.5 指定）
u_ground_color = [0.52, 0.47, 0.42]     # 暖灰地光（§3.5 指定）
u_ambient_strength = 0.55               # 亮调粉彩
# cel shading
u_shadow_threshold = 0.18
u_highlight_threshold = 0.55
u_shadow_tint = [0.62, 0.66, 0.92]      # 蓝紫补色
# 粉彩
u_saturation = 0.85
u_brightness = 1.10
# 描边
u_outline_color = [0.12, 0.10, 0.16]    # 深紫黑（非纯黑，日系）
u_outline_threshold = 0.008             # 平方锐化后量级（实测）
u_depth_edge_scale = 0.0                # early-z 失效，默认关（留档供 G-Buffer 方案）
u_normal_edge_scale = 1.5
# 大气透视
u_near = 3.0
u_far = 12000.0
u_fog_near = 300.0
u_fog_far = 3000.0
u_fog_color = [0.78, 0.70, 0.88]        # 远景染天光
u_fog_strength = 0.35
# 调试（0=正常渲染；1=d_edge 2=n_edge_raw 3=edge 可视化）
u_debug_edge = 0
```

---

## 六、与主题系统的集成（✅ 零代码改动）

| 项 | 说明 |
|----|------|
| 目录 | `assets/shaders/anime/`（vert + frag + theme.toml） |
| 发现 | `ThemeManager.discover()` 自动扫描到 → 菜单自动出现 |
| 切换 | `switch("anime")` 热切换，回调重绑 VAO（5a 已就绪） |
| uniforms | `apply_uniforms()` 自动注入 theme.toml（含新 uniform，无 Python 改动） |
| 坑 | 主题切换 GL context 非 current → `makeCurrent()`（5a 已修）；切回 `InvalidObject`（5a 已修） |
| 大气透视 near/far | 相机 near=3/far=12000 是**固定常量**（camera.py），theme.toml 写死即可，无需注入 |

---

## 七、测试（✅ 已实施）

### 7.1 无头测试

- 全量 pytest **299 passed**（无回归；anime 为纯数据目录 + 着色器，无 Python 逻辑）

### 7.2 GUI 冒烟 `tests/smoke_theme_anime.py`（✅）

1. 依次切换 `default → constructivism → anime → default → anime`（热切换稳定）
2. cel 量化：anime `lum_peaks=2`、`lum_mean=74.5`（粉彩亮调，default 53.1）
3. **描边开关对照**（关键方法）：正常渲染 vs `u_outline_threshold=10`（描边关）两帧差异像素
   = 真实描边（不受 cel 暗部补色干扰）→ 实测 `diff_px=10536 → outline_ratio=0.89%` ✓

> ⚠️ 教训：**不能用「颜色接近 outline_color」做描边统计**——cel 暗部天光补色
> 本就接近描边色（24% 假阳性）；必须用开关对照法。

### 7.3 阈值标定脚本 `tests/diag_anime_edge.py`（✅）

- 用 `u_debug_edge` 把 d_edge / n_edge_raw / edge 可视化输出（灰度=clamp(show*10,0,1)）
- 实测边缘值分位数（p50/p75/p90/…），据此定 threshold
- 另 `tests/diag_anime_outline.py`：阈值/权重扫描 + 消融截图（depth_only/normal_only/both/none）

---

## 八、实施步骤（✅ 全部完成）

1. **建目录 + 着色器**：`assets/shaders/anime/`（vert 复制 default，frag 按 §四）
2. **theme.toml**：按 §五 最终值
3. **冒烟验证**：`tests/smoke_theme_anime.py`（新脚本）
4. **阈值标定**：`tests/diag_anime_edge.py` + `tests/diag_anime_outline.py`
5. **像素统计测试**：描边开关对照法固化
6. **文档收尾**：`phase5-ui-pipeline-design.md` 状态 + `dev-progress.md`

---

## 九、风险与坑清单

| # | 坑 | 说明 |
|---|----|------|
| 1 | **fwidth 在分支内** | `dFdx/dFdy/fwidth` 必须在 fragment 开头无条件调用（2×2 quad 一致执行），不能放进 `if` 分支，否则结果未定义 |
| 2 | **深度通道 early-z 失效（实测）** | 背景像素被早期深度测试提前丢弃 → quad 不完整 → `fwidth(gl_FragCoord.z)` 恒 ≈0。设计稿的「NDC 常数性」推导对剪影不成立（剪影需要背景像素参与导数）。深度通道默认关 |
| 3 | **地形网格线（法线微变）** | 平滑法线跨格连续，但相邻三角形插值后屏幕导数仍有 ~0.02~0.04 微变 → 直接线性阈值描出网格线。**平方锐化** `n²` 是正解（弱边缘压到阈值下） |
| 4 | **cel 暗部污染描边统计** | 暗部天光补色接近描边色 → 「颜色接近 outline」的统计有 ~24% 假阳性。必须用**描边开关对照法**测真实描边 |
| 5 | **掠射角误描** | 相机几乎贴地时法线屏幕导数偏大 → 阈值标定时留余量 |
| 6 | **主题切换黑屏/InvalidObject** | 5a 已修（makeCurrent + pop 缓存），新主题同样适用 |
| 7 | **GLSL 语法错误静默** | moderngl 编译失败抛异常但可能在非 current context 下被吞 → 标定脚本在 makeCurrent 下跑 |

---

## 十、范围控制（5d 不做，留档）

- **方案 B（G-Buffer Sobel）**：可调宽描边、精细边缘 + **深度剪影**（单 pass 深度通道因 early-z
  失效，剪影描边必须走离屏 FBO 邻域采样），需离屏 FBO 管线（见 §3.2 对比）
- **硬边高光（specular 色块）**：对地形无意义，等单位 3D 模型（Phase 7/8）再做
- **HUD 主题化**：单位圆点描边/卡通化，需 HUD program 支持主题参数（Phase 8 孪生置信度渲染一并做）
- **描边宽度参数化**：fwidth 方案天然 1px，宽度控制属方案 B
