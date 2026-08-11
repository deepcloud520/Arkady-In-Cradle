"""narrative 层：独立于引擎核心的叙事层（AGENTS.md 3.7）。

消费 engine 的 simulation 状态（Simulation.get_state()），驱动剧情与对话；
不嵌入无头引擎核心，不控制模拟流程（通过 Command 队列间接交互）。—— Phase 4 待实现。
"""
