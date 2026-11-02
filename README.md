# CS224R: Deep Reinforcement Learning

>  **CS224R: Deep Reinforcement Learning** (Spring 2025) 课程作业，包含模仿学习、模型预测控制、离线强化学习、目标条件强化学习以及元强化学习四个编程作业。

---

## Assignments Summary

| Assignment | Topic | Methods / Algorithms | Environments |
| :--- | :--- | :--- | :--- |
| **Assignment 1** | **Imitation Learning** | Behavior Cloning (BC), DAgger | MuJoCo (`Ant-v4`, `Hopper-v4`) |
| **Assignment 2** | **Online RL & MPC** | MuJoCo MPC (MJPC), Ensemble Actor-Critic (with BC) | MuJoCo Quadruped (`Flat`, `Hill`), Meta-World (`hammer-v2`) |
| **Assignment 3** | **Offline RL & Exploration** | CQL (Conservative Q-Learning), IQL (Implicit Q-Learning), RND | `PointMass` (`Easy`, `Medium`, `Hard`) |
| **Assignment 4** | **Goal-Conditioned RL & Meta-RL** | Goal-Conditioned DQN, Hindsight Experience Replay (HER), $`RL^2`$, DREAM | `BitFlipEnv`, Sawyer Manipulation, Grid / City Navigation |

---

## Assignment 1: Imitation Learning (BC & DAgger)

Implementation of continuous-action imitation learning, evaluating Behavior Cloning (BC) and Dataset Aggregation (DAgger) to address distribution shift (covariate shift) in continuous control tasks.

- **Methods**: Behavior Cloning (BC), DAgger

**Key Implementations**

- `MLPPolicySL`: Gaussian policy network with learnable $`\log \sigma`$ for continuous action prediction.
- Supervised BC objective: $`\min_\theta \mathbb{E}_{(s, a^*) \sim \mathcal{D}} [-\log \pi_\theta(a^*|s)]`$.
- DAgger loop: Iteratively rolls out the learned policy, queries expert $`\pi^*`$ for ground-truth actions on visited states, and aggregates trajectories:

```math
\mathcal{D}_k = \mathcal{D}_{k-1} \cup \{(s, \pi^*(s)) \mid s \sim d^{\pi_{\theta_{k-1}}}\}
```

- **Environments**: MuJoCo `Ant-v4`, `Hopper-v4`
- **Experiments**: Evaluated policy performance across training steps and aggregation iterations against compounding errors.

---

## Assignment 2: Online RL & Model Predictive Control

Exploration of model-based trajectory optimization with MuJoCo MPC alongside sample-efficient online Actor-Critic learning augmented with demonstrations.

- **Methods**: Model Predictive Control (MJPC), Ensemble Actor-Critic (with BC)
- **Key Implementations**:
  - **Part 1 (MuJoCo MPC)**: Real-time predictive control on quadruped locomotion (`Quadruped Flat` & `Hill`) using DeepMind's `mujoco_mpc` framework; configured task cost weights and planning horizons.
  - **Part 2 (Online Actor-Critic & BC)**: Truncated Normal continuous policy; multi-critic ensemble with Clipped Double-Q targets ($`\min(Q_1, Q_2)`$) and soft target updates ($`\tau = 0.005`$); dual replay buffer combining online transitions with demonstrations and auxiliary BC loss.
- **Environments**: MuJoCo Quadruped (`Flat`, `Hill`), Meta-World (`hammer-v2`)

---

## Assignment 3: Offline Reinforcement Learning (CQL & IQL)

Investigation of offline RL methods that learn policies from collected datasets, contrasting conservative penalties with in-support implicit value estimation.

- **Methods**: Conservative Q-Learning (CQL), Implicit Q-Learning (IQL), Random Network Distillation (RND)

**Key Implementations**

- **Conservative Q-Learning (CQL)**: Penalizes out-of-distribution (OOD) action values via a Log-Sum-Exp regularizer to construct a conservative lower bound:

```math
\mathcal{L}_{\text{CQL}}(\theta) = \mathcal{L}_{\text{TD}}(\theta) + \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_{a} \exp(Q_\theta(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q_\theta(s, a)] \right]
```

- **Implicit Q-Learning (IQL)**: Avoids evaluating unseen actions entirely by learning state values $`V(s)`$ via asymmetric expectile regression:

```math
L_2^\tau(u) = |\tau - \mathbb{I}(u < 0)| u^2, \quad u = Q_{\hat{\theta}}(s, a) - V_\psi(s)
```

followed by TD updates on $`Q(s, a)`$ and advantage-weighted policy extraction.

- **Exploration & Data Collection**: Offline dataset generation comparing undirected random exploration against curiosity-driven exploration via Random Network Distillation (RND).
- **Environments**: `PointMass` (`Easy`, `Medium`, `Hard`)
- **Experiments**: Evaluated CQL and IQL across hyperparameter settings (CQL $`\alpha`$, IQL expectile $`\tau`$).

---

## Assignment 4: Goal-Conditioned RL & Meta-Reinforcement Learning

Study of goal-conditioned policies and meta-reinforcement learning across multi-task distributions.

- **Methods**: Goal-Conditioned DQN, Hindsight Experience Replay (HER), $`RL^2`$, DREAM
- **Key Implementations**:
  - **Part 1 (Goal-Conditioned RL & HER)**: Goal-conditioned value network parameterized over concatenated state-goal inputs $`Q(s, a, g)`$; implemented a `NO_HINDSIGHT` baseline alongside HER relabeling with `FINAL`, `FUTURE` ($`k=4`$), and `RANDOM` strategies.
  - **Part 2 (Meta-RL: $`RL^2`$ vs. DREAM)**:
    - **$`RL^2`$**: Recurrent black-box meta-RL ingesting historical transition sequences $`(s, a, r, d)`$ into RNN hidden states to optimize multi-task returns end-to-end.
    - **DREAM**: Decoupled exploration and exploitation architecture; learns latent task representations (Task Identity) via an encoder-decoder to train dedicated information-seeking exploration policies.
- **Environments**: `BitFlipEnv`, Sawyer Robotic Manipulation, Grid / City Navigation

---

## Repository Structure

```text
.
├── hw1/                               # Assignment 1: Imitation Learning
│   ├── CS224R_2025_Homework_1.pdf
│   └── hw1/
│       ├── cs224r/                    # BC & DAgger policies, agents, and pipelines
│       └── data/                      # Training logs
├── hw2/                               # Assignment 2: Online RL & MPC
│   ├── CS224R_2025_Homework_2.pdf
│   └── hw2/
│       ├── ac/                        # Actor-Critic & BC
│       └── mujoco_mpc/                # MuJoCo MPC trajectory optimization
├── hw3/                               # Assignment 3: Offline RL & Exploration
│   ├── CS224r_Homework3.pdf
│   └── hw3/
│       ├── cs224r/                    # CQL, IQL, and RND models
│       └── data/                      # Training logs & plots
└── hw4/                               # Assignment 4: Goal-Conditioned RL & Meta-RL
    ├── CS224R_2025_Homework_4.pdf
    └── hw4/
        ├── goal_conditioned_rl/       # Goal-conditioned RL with HER
        └── meta_rl/                   # Meta-RL (RL^2 & DREAM)
```
