# SWMMEnv
[![python swimming](assets/pythonSwimming.jpg)](https://martinparr.com/)
<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README_CN_simplified.md">中文</a> · <a href="README_JP.md">日本語</a> · <a href="README_CN_traditional.md">繁體中文</a>
</p>

**Multi-agent reinforcement learning environment for SWMM stormwater simulation**, built on [PettingZoo](https://pettingzoo.farama.org/) and [PySWMM](https://github.com/pyswmm/pyswmm).

---

## Overview

SWMMEnv wraps EPA SWMM hydraulic simulations as a PettingZoo-compatible and RLlib-native multi-agent RL environment. Each pump, gate, or weir in your SWMM model becomes a controllable RL agent

## Features

- **PettingZoo ParallelEnv API** 鈥?drop-in compatible with MARLlib, RLlib, Tianshou
- **RLlib MultiAgentEnv API** 鈥?pass class reference directly, no manual adapter needed
- **Config-driven** 鈥?swap SWMM models without changing code
- **Flexible agent types** 鈥?pumps, gates, weirs with continuous control `[0, 1]`
- **Time synchronization** 鈥?decouples RL decision interval from SWMM routing step
- **State normalization** 鈥?z-score or min-max normalization for stable training
- **Custom reward functions** 鈥?pluggable reward with flooding, level-deviation, and energy components
- **Parallel worker support** 鈥?isolated `.inp` copies for safe multi-worker training

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

> `ray[rllib]` is an **optional** dependency. You can import `SWMMMultiAgentEnv` without it, but ray must be installed to instantiate the class.

## Quick Start

### Basic Usage with PettingZoo API

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

### RLlib API (SWMMMultiAgentEnv)

`SWMMMultiAgentEnv` wraps `SWMMParallelEnv` into an RLlib-compatible `MultiAgentEnv`. Pass the class reference directly 鈥?no factory function or manual adapter needed.

**Direct class reference** (RLlib 2.x+):

```python
from swmmEnv import SWMMMultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        SWMMMultiAgentEnv,                          # pass the class directly
        env_config={"config_path": "configs/control.yaml"},
    )
    .multi_agent(
        policies={"shared_policy": None},
        policy_mapping_fn=lambda agent_id: "shared_policy",
    )
)
algo = config.build()
```

**Via register_env** (classic pattern):

```python
from ray.tune.registry import register_env
from swmmEnv.envs.register_env import make_rllib_env

register_env("swmm_env", lambda cfg: make_rllib_env(config_path="configs/control.yaml"))
```

**Direct instantiation** (testing / debugging):

```python
from swmmEnv import SWMMMultiAgentEnv
import numpy as np

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, info = env.reset()
actions = {aid: np.array([0.5]) for aid in env.possible_agents}
obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

### Using Default Configuration

If you don't have an `.inp` file yet, you can load the package's built-in default config to inspect the environment structure:

```python
from swmmEnv import load_config

# Load built-in default configuration (no path = loads default)
config = load_config()
print(config)
```

### Standalone MDP (without PettingZoo)

Use the core `SWMMEnv` directly for testing or custom integration outside MARL frameworks:

```python
from swmmEnv import SWMMEnv, load_config

config = load_config("config/example.yaml")
env = SWMMEnv(config)

obs = env.reset()
actions = {"pump_1": 0.8, "gate_1": 0.5}
obs, reward, done, info = env.step(actions)

print(f"Reward: {reward:.3f}, Done: {done}")
env.close()
```

### Manual Random Control

Run a full episode with random actions to verify the environment works end-to-end:

```python
import numpy as np
from swmmEnv import SWMMParallelEnv, load_config

config = load_config("config/example.yaml")
env = SWMMParallelEnv(config)

observations, _ = env.reset()
total_reward = 0.0
step = 0
done = False

while not done:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    obs, rewards, terms, truncs, infos = env.step(actions)
    reward = list(rewards.values())[0]
    total_reward += reward
    step += 1
    done = any(terms.values())

    if step % 50 == 0:
        env.render()

print(f"Episode finished after {step} steps, total reward: {total_reward:.3f}")
env.close()
```

### Inspecting Environment State

Retrieve detailed hydraulic state during an episode:

```python
from swmmEnv import SWMMParallelEnv, load_config

config = load_config("config/example.yaml")
env = SWMMParallelEnv(config)
obs, _ = env.reset()

# Get full state snapshot
state = env.core_env.get_state()
print("Node states:", state["nodes"])
print("Link states:", state["links"])
print("Rainfall:", state["rainfall"])

# Get environment info for RL frameworks
info = env.get_env_info()
print(f"Agents: {info['num_agents']}, Episode limit: {info['episode_limit']}")

# Inspect individual nodes
engine = env.core_env.engine
for node_id in ["J1", "J2"]:
    node_state = engine.get_node_state(node_id)
    print(f"\nNode {node_id}: depth={node_state['depth']:.2f}m, "
          f"flooding={node_state['flooding']:.4f} m鲁/s, "
          f"inflow={node_state['total_inflow']:.2f} m鲁/s")

env.close()
```

### Interactive Control Mode

Manually input actions for each agent via command line, useful for debugging:

```bash
python examples/manual_control.py --interactive
```

Or load a custom config:

```bash
python examples/manual_control.py --config my_scenario.yaml --interactive
```

## Architecture

```
           RL Agent (MAPPO / QMIX / PPO)
           │
           ▼
┌─────────────────────────────┐
│ SWMMParallelEnv             │ PettingZoo ParallelEnv wrapper
│ ┌───────────────────────┐   │
│ │ SWMMEnv (MDP)         │   │ Core RL logic: obs, reward, done
│ │ ┌─────────────────┐   │   │
│ │ │ SWMMEngine      │   │   │ PySWMM simulation interface
│ │ │ TimeSync        │   │   │ RL/SWMM step alignment
│ │ │ Normalizer      │   │   │ Observation/reward scaling
│ │ │ Mapping         │   │   │ Agent ⇄ SWMM element registry
│ │ │ Reward Fn       │   │   │ Pluggable reward computation
│ │ └─────────────────┘   │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
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

- **Observation space**: `Box(-鈭? +鈭?` after z-score normalization
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

### Passing a Custom Reward Function

There are three ways to pass a custom reward function:

#### 1. Via YAML config (built-in only)

Set the `reward_fn` field in your YAML file to one of the built-in names:

```yaml
reward_fn: "multi_objective"
```

#### 2. Pass a callable directly (recommended for simple logic)

Define a function with signature `(engine, config) -> float` and assign it to `config["reward_fn"]`:

```python
from swmmEnv import SWMMParallelEnv, load_config

def my_reward(engine, config):
    flooding = engine.get_total_flooding()
    depth = engine.get_node_state("J1")["depth"]
    # Your custom logic here
    return -flooding - abs(depth - 1.5)

config = load_config("config.yaml")
config["reward_fn"] = my_reward   # Override with custom function
env = SWMMParallelEnv(config)
```

#### 3. Class-based stateful reward (for cross-step tracking)

When your reward depends on previous states (e.g., penalizing rapid setting changes), inherit from `CustomRewardFunction`:

```python
from swmmEnv.reward.custom_reward import CustomRewardFunction

class StabilityReward(CustomRewardFunction):
    def __init__(self, flood_weight=1.0, stability_weight=0.1):
        super().__init__()
        self.flood_weight = flood_weight
        self.stability_weight = stability_weight
        self.prev_depths = {}

    def __call__(self, engine, config):
        # Flooding penalty
        reward = -self.flood_weight * engine.get_total_flooding()

        # Stability penalty: penalize large depth changes between steps
        for node_id in config.get("obs_nodes", []):
            try:
                depth = engine.get_node_state(node_id)["depth"]
            except (KeyError, ValueError):
                continue
            if node_id in self.prev_depths:
                change = abs(depth - self.prev_depths[node_id])
                reward -= self.stability_weight * change
            self.prev_depths[node_id] = depth
        return reward

    def reset(self):
        """Reset internal state at the start of a new episode."""
        self.prev_depths = {}

# Usage
config["reward_fn"] = StabilityReward(flood_weight=2.0, stability_weight=0.2)
env = SWMMParallelEnv(config)
```

> **Note**: When using a class-based reward with `SWMMParallelEnv`, the environment calls `reset()` on the reward function at the start of each episode if the method exists.

See `swmmEnv/reward/custom_reward.py` for the full template and more examples.

## API Reference

### `SWMMParallelEnv(config)`

The primary PettingZoo-compatible multi-agent environment. Implements `pettingzoo.ParallelEnv`.

```python
from swmmEnv import SWMMParallelEnv

env = SWMMParallelEnv(config)
obs, infos = env.reset()
obs, rewards, terminations, truncations, infos = env.step(actions)
```

**Key methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `(seed, options) -> (observations, infos)` | Start a new episode; returns initial observations |
| `step` | `(actions) -> (obs, rewards, terms, truncs, infos)` | Apply actions and advance the simulation by one decision interval |
| `observation_space` | `(agent) -> Box` | Get observation space for an agent (RLlib-compatible) |
| `action_space` | `(agent) -> Box` | Get action space for an agent (RLlib-compatible) |
| `observe` | `(agent) -> ndarray` | Get current observation for a specific agent |
| `state` | `() -> ndarray` | Get concatenated global observation (for centralized-critic algorithms) |
| `render` | `(mode) -> None` | Print current environment state to console |
| `close` | `() -> None` | Release simulation resources |
| `get_env_info` | `() -> dict` | Get environment metadata for MARLlib |

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `agents` | list | Currently active agent IDs |
| `possible_agents` | list | All agent IDs in the scenario |
| `core_env` | SWMMEnv | Underlying core MDP environment |
| `observation_spaces` | dict | Per-agent observation spaces |
| `action_spaces` | dict | Per-agent action spaces |

### `SWMMMultiAgentEnv(env_config)`

RLlib-compatible `MultiAgentEnv` wrapping `SWMMParallelEnv`. Extends `ray.rllib.env.MultiAgentEnv`.

```python
from swmmEnv import SWMMMultiAgentEnv

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, infos = env.reset()
obs, rewards, terms, truncs, infos = env.step(actions)
```

`env_config` accepts two forms:
- `{"config_path": "/path/to/config.yaml"}` 鈥?RLlib-style; resolves and loads the YAML file
- `{"inp_file": ..., "agents": ..., ...}` 鈥?full config dict passed through directly

**Key methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `(*, seed, options) -> (obs, infos)` | Start a new episode |
| `step` | `(action_dict) -> (obs, rewards, terms, truncs, infos)` | Step the environment; `terms`/`truncs` include `__all__` key |
| `close` | `() -> None` | Release simulation resources |
| `render` | `(mode) -> None` | Print current environment state |

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `possible_agents` | list | All agent IDs (replaces deprecated `get_agent_ids()`) |
| `observation_space` | `spaces.Dict` | Dict mapping agent ID 鈫?`Box` (legacy OldAPIStack) |
| `action_space` | `spaces.Dict` | Dict mapping agent ID 鈫?`Box` (legacy OldAPIStack) |
| `observation_spaces` | dict | Per-agent observation spaces (new API stack) |
| `action_spaces` | dict | Per-agent action spaces (new API stack) |

### `make_rllib_env(config_path)`

Convenience factory function in `swmmEnv.envs.register_env`:

```python
from swmmEnv.envs.register_env import make_rllib_env

env = make_rllib_env("configs/control.yaml")
```

### `SWMMEnv(config)`

Core MDP environment, independent of PettingZoo. Used internally by `SWMMParallelEnv`.

```python
from swmmEnv import SWMMEnv

env = SWMMEnv(config)
obs = env.reset()
obs, reward, done, info = env.step({"pump_1": 0.8, "gate_1": 0.5})
```

**Key methods:** `reset()`, `step(action_dict)`, `get_observation(agent_id)`, `get_reward()`, `get_state()`, `render(mode)`, `close()`.

### `SWMMEngine(inp_file, config, worker_index=0)`

Low-level PySWMM simulation wrapper. Handles simulation lifecycle, state retrieval, and action application.

```python
from swmmEnv import SWMMEngine

engine = SWMMEngine("model.inp", config, worker_index=0)
engine.start()
engine.apply_action("pump_1", 0.8)
engine.step()

# Retrieve state
node = engine.get_node_state("J1")
link = engine.get_link_state("P1")
flooding = engine.get_total_flooding()
rainfall = engine.get_rainfall("RG1")
time = engine.get_current_time()

engine.close()
```

**Key utilities:**

| Method | Description |
|--------|-------------|
| `start()` | Begin simulation and register before/after step callbacks |
| `step()` | Advance simulation by one control interval |
| `reset()` | Reset simulation using hotstart for fast episode reset |
| `close()` | Release simulation and cleanup worker files |
| `apply_action(agent_id, setting)` | Queue a control action in [0, 1] for an agent |
| `get_node_state(node_id)` | Get depth, head, volume, flooding, inflow for a node |
| `get_link_state(link_id)` | Get flow, depth, volume, current_setting for a link |
| `get_rainfall(gage_id)` | Get rainfall intensity (mm/h) from a rain gage |
| `get_total_flooding()` | Get total flooding rate across all nodes (m鲁/s) |
| `get_system_stats()` | Get routing and runoff statistics |
| `is_ended()` | Check if the simulation has reached its end time |
| `get_current_time()` | Get current simulation datetime |
| `save_hotstart(filepath)` | Save current state to a .hsf hotstart file |

### `TimeSync(decision_interval, swmm_step)`

Manages synchronization between RL decision steps and SWMM simulation steps. The `decision_interval` must be divisible by `swmm_step`.

```python
from swmmEnv.sim import TimeSync

ts = TimeSync(decision_interval=300, swmm_step=10)
print(ts.skip_steps)  # 30 SWMM steps per RL step
ts.advance(engine)    # Advances engine by 30 SWMM steps
```

**Key methods:** `advance(engine)`, `reset()`, `should_act(step)`, `get_elapsed_time()`, `get_elapsed_time_minutes()`.

### `StateNormalizer(config)`

Z-score normalization for observations and rewards to stabilize training.

```python
from swmmEnv.sim import StateNormalizer

normalizer = StateNormalizer(config["normalization"])
obs_normalized = normalizer.normalize_obs(raw_obs, obs_names)
reward_normalized = normalizer.normalize_reward(raw_reward)
reward_denorm = normalizer.denormalize_reward(normalized_reward)
```

**Key methods:** `normalize_obs(obs, obs_names)`, `normalize_obs_value(value, name)`, `normalize_reward(reward)`, `denormalize_reward(normalized_reward)`, `min_max_normalize(value, min, max)`, `clip_and_normalize(value, name, clip_range)`, `update_stats(obs, obs_names)`.

### `MappingRegistry(agents_config)`

Agent-to-SWMM-element mapping registry.

```python
from swmmEnv.sim import MappingRegistry

registry = MappingRegistry(config["agents"])
print(registry.get_all_agents())        # ["pump_1", "gate_1"]
print(registry.get_element_type("pump_1"))  # "pump"
print(registry.get_element_id("pump_1"))    # "P1"
print(registry.get_upstream_node("pump_1")) # "J1"
```

**Key methods:** `get_element_id(agent)`, `get_element_type(agent)`, `get_upstream_node(agent)`, `get_downstream_node(agent)`, `get_agent_config(agent)`, `get_all_agents()`, `get_agents_by_type(type)`, `get_all_link_ids()`, `get_all_node_ids()`, `agent_exists(agent)`.

### `load_config(config_path=None, merge_defaults=True)`

Load and validate a YAML configuration file.

```python
from swmmEnv import load_config

# From file
config = load_config("path/to/config.yaml")

# Default config only
config = load_config()

# Without merging defaults (used for custom env setup via make_env)
config = load_config("path/to/config.yaml", merge_defaults=False)
```

### `validate_config(config)`

Validate a configuration dictionary. Raises `ValueError` with descriptive messages if required fields are missing or invalid.

```python
from swmmEnv.config import validate_config

validate_config(config)  # Raises ValueError on invalid config
```


## Training

### RLlib Direct Integration (Recommended 鈥?New API)

The simplest way to use SWMMEnv with RLlib: pass `SWMMMultiAgentEnv` directly to `PPOConfig().environment()`.

```python
from swmmEnv import SWMMMultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

algo = (
    PPOConfig()
    .environment(
        SWMMMultiAgentEnv,
        env_config={"config_path": "configs/control.yaml"},
    )
    .multi_agent(
        policies={"shared_policy": None},
        policy_mapping_fn=lambda agent_id: "shared_policy",
    )
    .training(lr=0.0005, gamma=0.99, train_batch_size=4000)
    .resources(num_gpus=0)
).build()

for i in range(10):
    result = algo.train()
    print(f"Iteration {i}: "
          f"reward_mean={result['episode_reward_mean']:.2f}, "
          f"timesteps={result['timesteps_total']}")
```

### Ray RLlib (Classic register_env Pattern)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from swmmEnv.envs.register_env import make_rllib_env

register_env("swmm_env", lambda cfg: make_rllib_env(config_path="configs/control.yaml"))

algo = (
    PPOConfig()
    .environment("swmm_env")
    .multi_agent(
        policies={"shared_policy": None},
        policy_mapping_fn=lambda agent_id: "shared_policy",
    )
    .training(lr=0.0005, gamma=0.99, train_batch_size=4000)
    .resources(num_gpus=0)
).build()
```

### MARLlib Integration

SWMMEnv integrates with [MARLlib](https://github.com/Replicable-MARL/MARLlib) through a standalone registration mechanism.

**Step 1: Register the environment**

```python
from swmmEnv.envs.register_env import register_with_marllib

register_with_marllib()
# Now available as `env: "swmm"` in MARLlib configs
```

**Step 2: Create environment and train**

```python
from marllib import marl
from swmmEnv.envs.register_env import make_env

# Create environment
env = make_env(config_path="config/example.yaml")

# Build MAPPO model
model = marl.build_model(
    environment=env,
    algorithm=marl.algos.mappo,
    model_preference={
        "core_arch": "mlp",
        "encode_layer": "128-128",
        "hidden_dim": 64,
    }
)

# Start training
mappo = marl.algos.mappo(hyperparam_source="common")
mappo.fit(
    env=env,
    model=model,
    stop={"timesteps_total": 100000},
    lr=0.0005,
    gamma=0.99,
    batch_episode=10,
)
```

**Using `make_env` directly (without permanent registration):**

```python
from swmmEnv.envs.register_env import make_env

# With explicit config path
env = make_env(config_path="config/my_scenario.yaml", worker_index=0)

# With map_name lookup (searches configs/, config/, cwd)
env = make_env(map_name="control", worker_index=0)
```

`make_env` always loads configuration **without merging defaults**, avoiding conflicts from the default config's placeholder agents.

### Parallel Worker Training

SWMMEnv supports parallel environment workers for distributed training. Each worker gets an isolated copy of the `.inp` file to prevent file-lock conflicts:

```python
from swmmEnv.envs.register_env import make_env

# Worker 0 (uses original .inp file)
env_0 = make_env(config_path="config/example.yaml", worker_index=0)

# Worker 1 (gets a temp copy of .inp)
env_1 = make_env(config_path="config/example.yaml", worker_index=1)

# Worker 2 (gets another isolated copy)
env_2 = make_env(config_path="config/example.yaml", worker_index=2)
```

For efficient episode resets across many episodes, the engine uses **hotstart files** to restore simulation state without recreating the `Simulation` object (much faster for RL training loops).

**Key configuration for training:**

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `warmup_steps` | 0-10 | Steps before first RL decision; helps stabilize initial conditions |
| `max_steps` | 500-2000 | Episode length; tune based on storm event duration |
| `hotstart_file` | auto | Created automatically; provides fast resets |
| Normalization | Calibrate from data | Set `mean`/`std` based on historical simulation runs |


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
│   │   │   ├── pettingzoo_env.py # PettingZoo ParallelEnv wrapper
│   │   │   └── rllib_env.py     # RLlib MultiAgentEnv adapter
│   │   └── register_env.py      # MARLlib / RLlib registration helpers
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
├── README_CN.md                 # 简体中文
├── README_JP.md                 # 日本語
├── README_TW.md                 # 繁體中文
└── README.md                    # English
```



## Requirements

- Python >= 3.8
- PySWMM >= 2.1.0
- PettingZoo >= 1.24.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- PyYAML >= 6.0

## Citation

```bibtex
@software{swmmEnv2025,
  author = {dujddx},
  title = {SWMMEnv: Multi-agent RL Environment for SWMM Stormwater Simulation},
  url = {https://github.com/DujDDx/swmmEnv},
  version = {0.1.0},
  year = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Running Examples & Tests

### Example Scripts

The repository includes ready-to-run examples in the `examples/` directory:

```bash
# Random control (default config, 1 episode)
python examples/manual_control.py

# Random control with custom config and 5 episodes
python examples/manual_control.py --config my_config.yaml --episodes 5

# Interactive mode (manually type actions)
python examples/manual_control.py --interactive

# MAPPO training with MARLlib
python examples/train_mappo.py

# MAPPO training with custom config
python examples/train_mappo.py --config my_config.yaml --steps 50000

# RLlib training directly
python examples/train_mappo.py --backend rllib --steps 100000
```

### Running Tests

Run the unit test suite to verify the installation:

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific component
pytest tests/test_engine.py -v
pytest tests/test_mapping.py -v
pytest tests/test_normalizer.py -v
pytest tests/test_time_sync.py -v
pytest tests/test_swmm_env.py -v
pytest tests/test_pettingzoo_env.py -v

# With coverage report
pytest tests/ --cov=swmmEnv --cov-report=term-missing
```

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `ValueError: decision_interval must be divisible by swmm_step` | Invalid `time_sync` config | Ensure `decision_interval % swmm_step == 0` |
| `FileNotFoundError: Configuration file not found` | Wrong config path | Use absolute path or path relative to cwd |
| `PySWMM simulation not started` errors | Missing `.inp` file or incorrect path | Verify `inp_file` exists and path is correct |
| RL training gradients unstable | Poor normalization parameters | Calibrate `mean`/`std` from historical simulation data |
| Environment resets are slow | Hotstart file not used | Worker creates hotstart automatically; check `_initial_hotstart` is set |
| `RuntimeError: Environment must be reset before stepping` | `reset()` not called before `step()` | Always call `env.reset()` first |
| Concurrent simulation crashes | Multiple PySWMM instances on same `.inp` | Set `worker_index > 0` for each worker to get isolated `.inp` copies |
| `ImportError: ray[rllib] is required` when instantiating `SWMMMultiAgentEnv` | ray not installed | Run `pip install "swmmEnv[marl]"` or `pip install ray[rllib]` |
