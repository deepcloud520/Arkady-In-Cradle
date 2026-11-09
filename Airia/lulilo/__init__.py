"""无头推演引擎（headless）Airia.lulilo：Airia 的核心子包.
(
仅包含纯逻辑模块，不依赖任何渲染 / UI / 叙事：
    core → coordinates → terrain → entities → movement → detection
         → combat → logistics → simulation

渲染（Airia.render）是引擎外的官方实现；叙事（Airia.narrative）是独立的
上层消费层，均不进入引擎核心。
)
"""
