"""无头推演引擎（headless）：Airis 的核心子包，代号 **lulilo**（世界语「摇篮」）。

呼应《Arkady in Cradle》——引擎如摇篮，承载整场推演。
仅包含纯逻辑模块，不依赖任何渲染 / UI / 叙事：
    core → coordinates → terrain → entities → movement → detection
         → combat → logistics → simulation

渲染（Airis.render）是引擎外的官方实现；叙事（Airis.narrative）是独立的
上层消费层，均不进入引擎核心。
"""
