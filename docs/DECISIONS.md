# SnakeMind Architecture Decisions

## ADR-001: One shared environment

All agents will interact with one Snake environment. Algorithms must not
implement their own independent game mechanics.

Reason: fair comparisons and consistent evaluation.

## ADR-002: Gymnasium interface

The environment follows Gymnasium's reset and step API.

Reason: interoperability with RL libraries and standard evaluation tools.

## ADR-003: Relative actions

The action space contains three actions:
- 0: straight
- 1: right
- 2: left

Reason: the agent controls movement relative to its current heading.

## ADR-004: Three-channel spatial observation

The observation contains binary channels for:
- Snake head
- Snake body
- Food

Reason: a consistent spatial representation for future neural agents.

## ADR-005: Separate termination and truncation

Collisions and filling the board terminate an episode.
The configured step limit truncates an episode.

Reason: distinguish environmental terminal states from external time limits.

## ADR-006: Python 3.14.2 baseline

The initial environment dependencies were installed and verified under
Python 3.14.2 in GitHub Codespaces.

Reason: avoid an unnecessary interpreter downgrade when the initial
dependency stack works.

## ADR-007: RGB-array rendering

The environment exposes RGB-array rendering through NumPy.

Reason: support visualization while keeping training independent of
a graphical desktop.

A separate interactive viewer remains planned.
