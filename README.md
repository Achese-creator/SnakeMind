# SnakeMind

**Comparative Reinforcement Learning Research Environment for Autonomous Snake Agents**

SnakeMind is a research project investigating how autonomous agents learn to collect food and survive in a constrained Snake environment.

Rather than implementing separate games for different algorithms, SnakeMind provides one shared Gymnasium-compatible environment and a common agent interface. This makes it possible to compare random, deterministic planning and learned policies under consistent conditions.

## Research question

Can an agent learn to maximize food collection and survival in a constrained environment, and how does combining learned policies with safety mechanisms affect its performance?

## Current features

- Shared Gymnasium-compatible Snake environment.
- Three-channel spatial observations.
- Relative movement actions: straight, right and left.
- Deterministic environment seeding.
- Collision, food collection and board-completion handling.
- RGB-array rendering.
- Common agent interface.
- Random baseline agent.
- Hamiltonian-cycle planning baseline.
- Multi-seed evaluation framework.
- Per-episode CSV metrics.

## Architecture

```text
SnakeMind
├── snakemind/
│   ├── env.py
│   ├── evaluation.py
│   └── agents/
│       ├── base.py
│       ├── random_agent.py
│       └── hamiltonian_agent.py
├── scripts/
│   └── evaluate_baselines.py
├── tests/
├── docs/
│   ├── PROJECT_PLAN.md
│   └── DECISIONS.md
└── results/
    └── baselines/
```

## Setup

Python 3.14.2 was used for the initial verified development environment.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

## Run baseline experiments

```bash
python -m scripts.evaluate_baselines
```

The evaluation script runs both baseline agents using the same board configuration and evaluation seeds.

Results are written to:

```text
results/baselines/
```

Each CSV contains per-episode metrics, including score, survival steps, episode return, final length and termination reason.

## Initial baseline results

The initial pilot experiment used:
- Board size: 10 × 10.
- Ten evaluation seeds: 100–109.
- Maximum episode length: 10,000 steps.

| Metric | RandomAgent | HamiltonianAgent |
|---|---:|---:|
| Mean score | 0 | 97 |
| Mean survival steps | 16.5 | 2,419.6 |
| Mean final length | 3 | 100 |
| Board completion rate | 0% | 100% |

The Hamiltonian agent completed the board in all ten pilot episodes. The random agent did not collect food in these runs.

These results are preliminary and should not be interpreted as final statistical evidence. The Hamiltonian agent is a deterministic planning baseline, not a learned RL policy.

## Roadmap

1. Shared environment foundation.
2. Baseline agents and evaluation.
3. Tabular Q-Learning.
4. DQN and Double DQN.
5. PPO.
6. NEAT neuroevolution.
7. Hybrid reinforcement learning with safety constraints.
8. Self-play and multi-agent experiments.
9. Rigorous statistical evaluation.
10. Interactive dashboard and public demonstration.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full roadmap and [docs/DECISIONS.md](docs/DECISIONS.md) for architecture decisions.

## Current status

Milestones 1 and 2 have been implemented and verified. The current test suite contains 30 passing tests.

The next development milestone is tabular Q-Learning.

## License

License to be determined.
