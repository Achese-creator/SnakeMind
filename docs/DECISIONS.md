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

## ADR-008: Common agent interface

All agents implement a shared interface with:
- `reset(seed=None)` for episode initialization.
- `act(observation, info)` for action selection.

Reason: allow different algorithms to operate through the same environment and evaluation pipeline.

Learned agents must preserve learned parameters when resetting episode-specific state.

## ADR-009: Hamiltonian-cycle baseline constraints

The Hamiltonian agent uses a deterministic cycle covering every cell on supported even square boards of size 6 or greater.

It is a planning baseline, not a learned reinforcement learning algorithm. Its current implementation assumes the standard initial Snake environment state.

Reason: establish a deterministic survival and board-coverage reference point.

## ADR-010: Shared evaluation protocol

The initial baseline evaluation uses:
- Board size: 10 × 10.
- Maximum episode steps: 10,000.
- Evaluation seeds: 100–109.
- Identical environment rules and episode limits for both agents.

Per-episode metrics include score, survival steps, episode return, final length, termination status and termination reason.

Results are saved as CSV files.

Reason: make baseline experiments reproducible and enable future comparisons.

The initial ten-seed experiment is a pilot. Final research comparisons will require expanded evaluation, statistical uncertainty estimates and appropriate statistical tests.

## ADR-011: Provisional Q-Learning Exploration Schedule

**Status:** Accepted provisionally

**Decision:** Use `epsilon_decay=0.999` as the current working exploration
schedule for tabular Q-Learning experiments.

**Evidence:** Three 5,000-episode experiments compared decay values of
0.995, 0.999 and 0.9995. Their 10-episode evaluation mean scores were
14.8, 16.0 and 15.8 respectively.

**Rationale:** The 0.999 schedule produced the strongest observed evaluation
result and a gradual training improvement. The 0.9995 schedule retained
more exploration but did not demonstrate a corresponding evaluation gain.

**Limitations:** Only 10 evaluation seeds were used. Differences have not
been established as statistically significant. This decision is a working
experimental configuration, not a claim of optimality.

**Revisit when:** Evaluation is expanded to more seeds, confidence intervals
are calculated, or the state representation and training methodology change.
