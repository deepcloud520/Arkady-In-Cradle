"""暂停策略测试（2026-11-03）。

规则：
    - 暂停只能由玩家 / 教程触发；
    - 真正战役（``real_campaign`` flag）**禁用暂停**（玩家 / 教程 / 剧情都不行）；
    - 上锁时立即清空既有暂停，并解除引擎的 ``sim_paused``。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from Airia.narrative.pause import (
    FLAG_REAL_CAMPAIGN,
    PauseController,
)


# ----------------------------------------------------------------------
# 纯策略
# ----------------------------------------------------------------------

def test_pause_sources_combine() -> None:
    pc = PauseController()
    assert not pc.paused
    assert pc.request_manual()
    assert pc.paused
    assert pc.active_sources == ("manual",)

    pc.request_tutorial(True)
    assert pc.active_sources == ("manual", "tutorial")
    pc.request_tutorial(False)
    pc.request_manual(False)
    assert not pc.paused


def test_manual_toggle() -> None:
    pc = PauseController()
    assert pc.request_manual() and pc.manual
    assert pc.request_manual() and not pc.manual      # None = 切换


def test_campaign_locks_everything() -> None:
    pc = PauseController()
    pc.request_manual(True)
    pc.request_tutorial(True)
    pc.set_narrative(True)
    assert pc.paused

    pc.set_locked(True)
    # 上锁：清空既有暂停，且所有来源被拒
    assert not pc.paused
    assert pc.active_sources == ()
    assert not pc.can_pause
    assert not pc.request_manual(True)
    assert not pc.request_tutorial(True)
    pc.set_narrative(True)
    assert not pc.paused

    # 解锁后恢复能力
    pc.set_locked(False)
    assert pc.request_manual(True)
    assert pc.paused


def test_campaign_blocks_narrative_pause() -> None:
    pc = PauseController()
    pc.set_locked(True)
    pc.set_narrative(True)
    assert not pc.narrative and not pc.paused


# ----------------------------------------------------------------------
# 与叙事引擎接线
# ----------------------------------------------------------------------

def _engine_with_stub_loop():
    from Airia.narrative.engine import NarrativeEngine

    loop = SimpleNamespace(sim_paused=False)
    eng = NarrativeEngine(command_queue=None, recorder=None, loop=loop)
    return eng, loop


def test_engine_campaign_flag_syncs_loop() -> None:
    eng, loop = _engine_with_stub_loop()
    assert eng.request_manual_pause(True)
    assert loop.sim_paused is True

    eng.set_campaign_mode(True)
    assert eng.has_flag(FLAG_REAL_CAMPAIGN)
    assert eng.pause_locked
    assert loop.sim_paused is False          # 上锁即解除暂停
    assert not eng.request_manual_pause(True)

    eng.set_campaign_mode(False)
    assert not eng.pause_locked
    assert eng.request_manual_pause(True)
    assert loop.sim_paused is True


def test_engine_dialogue_pause_ignored_in_campaign() -> None:
    eng, loop = _engine_with_stub_loop()
    # 不依赖 story/*.yaml：直接桩掉对话树激活
    eng.dialogue_engine.activate = lambda did, ctx: SimpleNamespace(
        dialogue_id=did, text="x", options=[], history=[], is_finished=False
    )

    # 非战役：剧情对话正常暂停推演
    eng._activate_dialogue("prologue_001", pause_sim=True)
    assert eng.is_paused is True
    assert loop.sim_paused is True

    # 战役模式：同一次对话不再暂停，且已有暂停被清空
    eng.set_campaign_mode(True)
    assert loop.sim_paused is False
    eng._activate_dialogue("first_blood", pause_sim=True)
    assert eng.is_paused is False
    assert loop.sim_paused is False


def test_tutorial_pause_respects_lock() -> None:
    eng, loop = _engine_with_stub_loop()
    assert eng.request_tutorial_pause(True)
    assert loop.sim_paused is True
    eng.request_tutorial_pause(False)
    eng.set_campaign_mode(True)
    assert not eng.request_tutorial_pause(True)
    assert loop.sim_paused is False
