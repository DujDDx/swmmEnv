# SWMMEnv

Multi-agent reinforcement learning environment for SWMM (Storm Water Management Model) simulation.

## Overview

SWMMEnv provides a PettingZoo-compatible interface for training multi-agent reinforcement learning (MARL) algorithms on stormwater management simulations. It integrates:

- **PySWMM**: SWMM simulation engine
- **PettingZoo**: Multi-agent RL environment interface
- **MARLlib**: Training framework (MAPPO, QMIX, etc.)

## Features

- Read standard SWMM `.inp` files and rainfall `.dat` files
- Control pump stations, gates, and weirs
- Global reward for coupled stormwater systems
- Config-driven design for different SWMM models
- Time synchronization between RL steps and SWMM simulation steps
- State normalization for stable training

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from swmmEnv import SWMMParallelEnv, load_config

# Load configuration
config = load_config("config/example.yaml")

# Create environment
env = SWMMParallelEnv(config)

# Reset and get initial observations
observations, info = env.reset()

# Take a step
actions = {"pump_1": 0.8, "gate_1": 0.5}
observations, rewards, terminations, truncations, infos = env.step(actions)

# Close environment
env.close()
```

## Configuration

See `config/default_config.yaml` for configuration structure.

## Project Structure

```
swmmEnv/
├── swmmEnv/
│   ├── sim/           # Simulation modules (engine, time_sync, normalizer, mapping)
│   ├── envs/          # RL environments (core MDP + PettingZoo wrapper)
│   ├── reward/        # Reward functions
│   └── config/        # Configuration system
├── tests/             # Unit tests
├── examples/          # Example scripts
└── data/              # Sample SWMM models
```

## License

MIT License