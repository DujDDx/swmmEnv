# SWMMEnv

<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README_CN.md">中文文档</a>
</p>

**Multi-agent reinforcement learning environment for SWMM stormwater simulation**, built on [PettingZoo](https://pettingzoo.farama.org/) and [PySWMM](https://github.com/pyswmm/pyswmm).

---

## Overview

SWMMEnv wraps EPA SWMM hydraulic simulations as a PettingZoo-compatible multi-agent RL environment. Each pump, gate, or weir in your SWMM model becomes a controllable RL agent, enabling training of policies for:

- **Flood mitigation** — minimize flooding across coupled drainage networks
- **Water level regulation** — maintain target levels in storage nodes
- **Energy-efficient pumping** — balance flood control against energy cost
- **Real-time stormwater control** — adaptive policy under rainfall uncertainty

## Features

- **PettingZoo ParallelEnv API** — drop-in compatible with MARLlib, RLlib, Tianshou
- **Config-driven** — swap SWMM models without changing code
- **Flexible agent types** — pumps, gates, weirs with continuous control `[0, 1]`
- **Time synchronization** — decouples RL decision interval from SWMM routing step
- **State normalization** — z-score or min-max normalization for stable training
- **Custom reward functions** — pluggable reward with flooding, level-deviation, and energy components
- **Parallel worker support** — isolated `.inp` copies for safe multi-worker training

## Installation

```bash
pip install swmmEnv
```

For MARL training extras:

```bash
pip install "swmmEnv[marl]"
```

For development:

```bash
pip install "swmmEnv[dev]"
```

## Quick Start

```python
from swmmEnv import SWMMParallelEnv, load_config

# Load configuration
config = load_config("config/example.yaml")

# Create environment
env = SWMMParallelEnv(config)

# Reset
observations, info = env.reset()

# Step: each agent outputs a continuous action [0, 1]
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

## Architecture

```
 RL Agent (MAPPO / QMIX / PPO)
          │
          ▼
 ┌────────────────────────┐
 │  SWMMParallelEnv       │  PettingZoo ParallelEnv wrapper
 │  ┌──────────────────┐  │
 │  │  SWMMEnv (MDP)   │  │  Core RL logic: obs, reward, done
 │  │  ┌─────────────┐ │  │
 │  │  │ SWMMEngine  │ │  │  PySWMM simulation interface
 │  │  │ TimeSync    │ │  │  RL/SWMM step alignment
 │  │  │ Normalizer  │ │  │  Observation/reward scaling
 │  │  │ Mapping     │ │  │  Agent ⇄ SWMM element registry
 │  │  │ Reward Fn   │ │  │  Pluggable reward computation
 │  │  └─────────────┘ │  │
 │  └──────────────────┘  │
 └────────────────────────┘
          │
          ▼
   PySWMM (SWMM5 engine)
```

## Configuration

```yaml
# config/example.yaml
inp_file: "data/model.inp"
rain_file: null
obs_raingage: RG1

agents:
  pump_1:
    type: pump
    link_id: P1
    upstream_node: J1
    downstream_node: J2
  gate_1:
    type: weir
    link_id: W1
    upstream_node: J3
    downstream_node: J4

time_sync:
  decision_interval: 300   # RL decides every 5 min
  swmm_step: 10            # SWMM routes every 10 s

normalization:
  obs:
    depth:  {mean: 2.0, std: 1.5}
    flow:   {mean: 0.5, std: 0.3}
    rainfall: {mean: 5.0, std: 10.0}
    setting: {mean: 0.5, std: 0.3}
  reward:
    mean: 0.0
    std: 10.0

reward_fn: "default_reward"
max_steps: 1000
```

## Agent Observation & Action Space

| Agent Type | Observation Dim | Features |
|-----------|----------------|----------|
| pump | 5 | upstream depth, downstream depth, flow, setting, rainfall |
| gate | 4 | upstream depth, downstream depth, setting, rainfall |
| weir | 4 | upstream depth, downstream depth, setting, rainfall |

- **Observation space**: `Box(-∞, +∞)` after z-score normalization
- **Action space**: `Box(0, 1)` continuous control setting

All agents share a **global reward** (coupled drainage system).

## Reward Functions

Four built-in reward functions, configurable via `reward_fn`:

| Name | Description |
|------|-------------|
| `default_reward` | Flooding + level deviation + energy efficiency |
| `flooding_only` | Pure flooding minimization |
| `normalized_flooding` | Flooding scaled to `[-1, 0]` |
| `multi_objective` | Flooding (10x) + level tracking |

Custom rewards — pass any `callable(engine, config) -> float`:

```python
def my_reward(engine, config):
    flooding = engine.get_total_flooding()
    depth = engine.get_node_state("J1")["depth"]
    return -flooding - abs(depth - 1.5)

config["reward_fn"] = my_reward
```

See `swmmEnv/reward/custom_reward.py` for stateful reward class templates.

## MARLlib Integration

```python
from swmmEnv.envs.register_env import make_env

env = make_env(config_path="config/my_scenario.yaml", worker_index=0)
```

Or register with MARLlib registry:

```python
from swmmEnv.envs.register_env import register_with_marllib
register_with_marllib()
# Now use `env: swmm` in MARLlib configs
```

## Project Structure

```
swmmEnv/
├── swmmEnv/
│   ├── __init__.py              # Package entry
│   ├── sim/
│   │   ├── engine.py            # PySWMM simulation wrapper
│   │   ├── time_sync.py         # RL/SWMM step synchronization
│   │   ├── normalizer.py        # Z-score observation normalization
│   │   └── mapping.py           # Agent ⇄ SWMM element registry
│   ├── envs/
│   │   ├── swmm_env/
│   │   │   ├── env.py           # Core MDP environment
│   │   │   └── pettingzoo_env.py # PettingZoo ParallelEnv wrapper
│   │   └── register_env.py      # MARLlib registration helpers
│   ├── reward/
│   │   ├── default_reward.py    # Built-in reward functions
│   │   └── custom_reward.py     # Custom reward templates
│   └── config/
│       ├── loader.py            # YAML config loading & validation
│       └── default_config.yaml  # Default configuration
├── examples/
│   ├── manual_control.py        # Interactive debugging
│   └── train_mappo.py           # MAPPO training example
├── tests/                       # Unit tests
├── pyproject.toml
└── README.md
```

## Requirements

- Python ≥ 3.8
- PySWMM ≥ 2.1.0
- PettingZoo ≥ 1.24.0
- Gymnasium ≥ 0.29.0
- NumPy ≥ 1.24.0
- PyYAML ≥ 6.0

## Citation

```bibtex
@software{swmmEnv2025,
  author = {dujddx},
  title = {SWMMEnv: Multi-agent RL Environment for SWMM Stormwater Simulation},
  url = {https://github.com/DujDDx/swmmEnv},
  version = {0.1.0},
  year = {2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
