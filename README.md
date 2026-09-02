# Airis Engine — Arkady in Cradle（AIC）
---
## Profile
《Arkady in Cradle》 is a sci-fi visual novel set in a fractured near-future world, blending real-time tactical wargaming with data-link warfare and fog-of-war mechanics. You play as *Arcadia*, a newly appointed EU commander, interacting with self-aware giant AIs, international megacorps, and splintered nations in the year 2060. Every tactical decision ripples through the narrative.

**An unofficial fan work of Neuro-sama and the Neuroverse — not affiliated with or endorsed by the original creators.**

## Highlights

- Narrative-first: the story is never skippable, but battles are — the simulation keeps running in the background either way
- Simulation core built on SoA object pools + Numba JIT (benchmarked at 2048 armour + 16k projectiles)
- 2.5D column-span heightfield terrain, free terrain destruction, background NavGraph re-baking
- Ground, artillery, and air forces integrated through data-link network-centric warfare (A-guides-B firing, twin-based fog of war)
- PySide6 + ModernGL renderer with hot-swappable shader themes

## Develop

*Arkady in Cradle* is still in an early development stage.

**Current Status**: Phase 8 (data-link network-centric warfare) is essentially complete — step ⑨ (the "millstone" formation tactic) is deferred; Phase 9 (performance / large maps) has not started yet. 2048 armour + 16k projectiles are supported, along with complex team AI and multi-type guided missiles.

### Prerequisites

- **Python 3.13** (`pyproject.toml` pins `>=3.13,<3.14`)
- **[uv](https://docs.astral.sh/uv/)** — used exclusively for package management (never bare `pip`)

### Setup

```bash
uv sync                 # creates .venv and installs all deps
uv sync --extra dev     # dev-only: pytest, pytest-cov, matplotlib
```

### Running Tests

```bash
uv run python -m pytest
```

All commands use `uv run` to ensure the correct venv is used without manual activation.

## Quick Start

```bash
uv run python main.py
```

## Docs

- `AGENTS.md` — architecture & engineering conventions
- `docs/` — per-phase design docs, e.g. `phase8-datalink-design.md`, `phase7-air-pipeline.md`

## Contribution

Special thanks: **Neuro-sama**

World-setting writer & narrative designer: **Archotech Spore Eta**

Programmer: **DeepSeek**
