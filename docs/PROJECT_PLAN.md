# SnakeMind — Project Plan

## 1. Project identity

**Name:** SnakeMind

**Description:** A comparative reinforcement learning research environment for autonomous Snake agents.

**Central research question:**
Can an agent learn to maximize food collection and survival in a constrained environment, and how does combining learned policies with safety mechanisms affect its performance?

SnakeMind is one shared research environment with interchangeable agents, not a collection of unrelated Snake implementations.

## 2. Locked architectural decisions

- One common Snake game engine.
- A Gymnasium-compatible environment API.
- A shared observation and action interface for all agents.
- Interchangeable agents using the same environment.
- One shared evaluation and experiment framework.
- Reproducible experiments with configurable random seeds.
- Separate training and evaluation.
- Checkpointing and training resumption.
- Metrics saved to CSV and visualized.
- A dashboard and interactive demonstration.
- Portable training workflows across Codespaces, Colab and Kaggle.
- No dependence on paid cloud services for the basic project workflow.

## 3. Planned agents and experiments

1. Random agent — simple baseline.
2. Hamiltonian-cycle agent — deterministic planning baseline.
3. Tabular Q-Learning.
4. Deep Q-Network (DQN).
5. Double DQN.
6. Proximal Policy Optimization (PPO).
7. NEAT — neuroevolution, evaluated separately from conventional gradient-based RL.
8. Hybrid RL — learned policies combined with safety constraints.
9. Self-play and multi-agent experiments.

## 4. Implementation roadmap

### Milestone 1 — Environment foundation
- [x] Implement the shared Snake environment.
- [x] Define observation and action spaces.
- [x] Implement food spawning, movement, rewards and collisions.
- [x] Handle episode termination and truncation correctly.
- [x] Add deterministic seeding.
- [x] Add environment unit tests.
- [x] Run Gymnasium environment validation.
- [x] Add a configurable episode step limit.
- [x] Add a rendering/visualization interface.

### Milestone 2 — Baseline agents
- [ ] Define a common agent interface.
- [ ] Implement random agent.
- [ ] Implement Hamiltonian-cycle agent.
- [ ] Verify both agents against the same environment.
- [ ] Establish baseline evaluation metrics.

### Milestone 3 — Tabular reinforcement learning
- [ ] Design a compact state representation.
- [ ] Implement Q-Learning.
- [ ] Add configurable hyperparameters.
- [ ] Save and load Q-tables.
- [ ] Evaluate against baseline agents.

### Milestone 4 — Deep reinforcement learning
- [ ] Implement DQN.
- [ ] Add experience replay.
- [ ] Add target network.
- [ ] Add training checkpoints and resumption.
- [ ] Implement Double DQN.
- [ ] Compare learning stability and performance.

### Milestone 5 — Policy optimization
- [ ] Implement PPO.
- [ ] Define appropriate observation preprocessing.
- [ ] Compare PPO with DQN and baselines.

### Milestone 6 — Neuroevolution
- [ ] Implement NEAT.
- [ ] Define neural network inputs and outputs.
- [ ] Evaluate population-based learning.
- [ ] Compare NEAT with RL approaches.

### Milestone 7 — Hybrid safety mechanisms
- [ ] Define safety constraints.
- [ ] Implement a safety/action-filtering mechanism.
- [ ] Compare unconstrained and safety-constrained policies.
- [ ] Measure the impact on survival and food collection.

### Milestone 8 — Self-play and multi-agent research
- [ ] Define multi-agent environment requirements.
- [ ] Implement self-play experiments where appropriate.
- [ ] Evaluate multi-agent behavior and limitations.

### Milestone 9 — Rigorous evaluation
- [ ] Evaluate across multiple random seeds.
- [ ] Use consistent board sizes and game rules.
- [ ] Separate training and held-out evaluation.
- [ ] Record episode return, score, survival, length and efficiency.
- [ ] Report means and variability.
- [ ] Compare agents under identical conditions.
- [ ] Preserve experiment configurations and results.

### Milestone 10 — Dashboard and public demonstration
- [ ] Build an interactive dashboard.
- [ ] Visualize Snake gameplay.
- [ ] Show agent comparisons and evaluation metrics.
- [ ] Allow users to select agents and experiments.
- [ ] Prepare a reproducible public demo.
- [ ] Document architecture, methodology and findings.

## 5. Evaluation principles

- No invented performance claims or unmeasured improvement percentages.
- Use multiple seeds rather than relying on one successful run.
- Compare agents under equivalent conditions.
- Preserve configurations, checkpoints and experiment results.
- Distinguish training performance from evaluation performance.
- Report limitations and failed experiments honestly.

## 6. Technology direction

Core:
- Python
- Gymnasium
- NumPy
- pytest

Later milestones:
- PyTorch
- Stable-Baselines3
- Pygame
- TensorBoard
- Matplotlib
- Streamlit
- NEAT implementation/library

Dependencies will be introduced when their milestone requires them.

## 7. Compute and artifact strategy

- GitHub Codespaces: primary development, testing and integration.
- Google Colab: optional GPU training.
- Kaggle Notebooks: alternative training environment.
- Lightning AI: optional additional compute.
- GitHub: source code, documentation and experiment configurations.
- Hugging Face Spaces: potential public demo hosting.

Cloud GPU availability is not guaranteed. Training must support checkpoints, resumption and portable configurations. Important artifacts must not exist only in temporary notebook storage.

## 8. Outstanding design considerations

These are explicitly retained for review and implementation:
- Maximum episode length and truncation semantics.
- Rendering and headless training support.
- Observation representation comparisons.
- Reward design and reward-shaping risks.
- Action masking and safety-filter design.
- Evaluation protocol and statistical uncertainty.
- Training performance and experiment tracking.
- Checkpoint portability and artifact persistence.
- Multi-agent environment design.
- Dashboard deployment and reproducibility.

## 9. Current project status

- [x] GitHub repository created.
- [x] Python 3.14.2 virtual environment created.
- [x] Initial dependencies installed: Gymnasium, NumPy and pytest.
- [x] Dependency versions verified.
- [x] Environment implementation verified.
- [x] Milestone 1 initial tests passed.

Update this section as work is completed. A task is only complete when its implementation and relevant verification are done.


## 10. Implementation notes — November 2026

- Python 3.14.2 is the active project interpreter.
- Gymnasium 1.4.0, NumPy 2.5.5 and pytest 9.2.0 verified.
- Initial Snake environment implemented.
- Relative three-action control implemented.
- Three-channel binary observation implemented.
- Seeded food spawning implemented.
- Collision, board completion and time-limit handling implemented.
- RGB-array rendering implemented.
- Initial unit tests and Gymnasium validation passed.
- Interactive graphical viewer remains outstanding.
- Full-board completion and full rendering validation are being verified.
