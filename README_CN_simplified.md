# SWMMEnv · 暴雨洪水管理模型多智能体强化学习环境
[![python swimming](assets/pythonSwimming.jpg)](https://martinparr.com/)
<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README.md">English</a> · <a href="README_JP.md">日本語</a> · <a href="README_CN_traditional.md">繁體中文</a>
</p>

基于 [PettingZoo](https://pettingzoo.farama.org/) 和 [PySWMM](https://github.com/pyswmm/pyswmm) 构建的 **SWMM 暴雨洪水管理模型多智能体强化学习环境**。

---

## 概述

SWMMEnv 将 EPA SWMM 水力模拟封装为 PettingZoo 兼容且 RLlib 原生的多智能体 RL 环境。SWMM 模型中的每个泵站、闸门和堰坝都成为一个可控制的 RL 智能体

## 功能特性

- **PettingZoo ParallelEnv API** — 无缝对接 MARLlib、RLlib、Tianshou 等框架
- **RLlib MultiAgentEnv API** — 直接传类名即可，无需手动编写适配器
- **配置驱动** — 更换 SWMM 模型无需修改代码
- **多种智能体类型** — 泵站、闸门、堰坝，连续控制 `[0, 1]`
- **时间同步** — 将 RL 决策间隔与 SWMM 路由步长解耦
- **状态归一化** — 支持 z-score 和 min-max 归一化，稳定训练
- **可插拔奖励函数** — 内置洪涝、水位偏差、能耗等多目标奖励组件
- **并行 Worker 支持** — 隔离 `.inp` 副本，安全支持多 Worker 并行训练

## 安装

```bash
pip install swmmEnv
```

安装 MARL 训练相关依赖：

```bash
pip install "swmmEnv[marl]"
```

安装开发依赖：

```bash
pip install "swmmEnv[dev]"
```

> `ray[rllib]` 是**可选**依赖。你可以在没有安装 ray 的情况下导入 `SWMMMultiAgentEnv`，但要实例化该类时必须安装 ray。

## 快速开始

### 基本用法（PettingZoo API）

```python
from swmmEnv import SWMMParallelEnv, load_config

# 加载配置
config = load_config("config/example.yaml")

# 创建环境
env = SWMMParallelEnv(config)

# 重置环境
observations, info = env.reset()

# 交互一步：每个智能体输出连续动作 [0, 1]
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### RLlib API（SWMMMultiAgentEnv）

`SWMMMultiAgentEnv` 将 `SWMMParallelEnv` 封装为 RLlib 兼容的 `MultiAgentEnv`。直接传类名即可，无需工厂函数或手动适配器。

**直接传类名**（RLlib 2.x+）：

```python
from swmmEnv import SWMMMultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        SWMMMultiAgentEnv,                          # 直接传类
        env_config={"config_path": "configs/control.yaml"},
    )
    .multi_agent(
        policies={"shared_policy": None},
        policy_mapping_fn=lambda agent_id: "shared_policy",
    )
)
algo = config.build()
```

**通过 register_env**（经典模式）：

```python
from ray.tune.registry import register_env
from swmmEnv.envs.register_env import make_rllib_env

register_env("swmm_env", lambda cfg: make_rllib_env(config_path="configs/control.yaml"))
```

**直接实例化**（测试 / 调试）：

```python
from swmmEnv import SWMMMultiAgentEnv
import numpy as np

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, info = env.reset()
actions = {aid: np.array([0.5]) for aid in env.possible_agents}
obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

### 使用默认配置

如果还没有 `.inp` 文件，可以加载包内置的默认配置来了解环境结构：

```python
from swmmEnv import load_config

# 加载内置默认配置（不传路径 = 加载默认配置）
config = load_config()
print(config)
```

### 独立使用 MDP 环境（不依赖 PettingZoo）

直接使用核心环境 `SWMMEnv` 进行测试或自定义集成：

```python
from swmmEnv import SWMMEnv, load_config

config = load_config("config/example.yaml")
env = SWMMEnv(config)

obs = env.reset()
actions = {"pump_1": 0.8, "gate_1": 0.5}
obs, reward, done, info = env.step(actions)

print(f"奖励: {reward:.3f}, 终止: {done}")
env.close()
```

### 手动随机控制

运行完整 episode 验证环境是否正常工作：

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

print(f"Episode 结束，共 {step} 步，累计奖励: {total_reward:.3f}")
env.close()
```

### 检查环境状态

在 episode 运行过程中获取详细的水力状态：

```python
from swmmEnv import SWMMParallelEnv, load_config

config = load_config("config/example.yaml")
env = SWMMParallelEnv(config)
obs, _ = env.reset()

# 获取完整状态快照
state = env.core_env.get_state()
print("节点状态:", state["nodes"])
print("管道状态:", state["links"])
print("降雨强度:", state["rainfall"])

# 获取环境信息（用于 RL 框架）
info = env.get_env_info()
print(f"智能体数: {info['num_agents']}, Episode 上限: {info['episode_limit']}")

# 查看单个节点
engine = env.core_env.engine
for node_id in ["J1", "J2"]:
    node_state = engine.get_node_state(node_id)
    print(f"
节点 {node_id}: 水深={node_state['depth']:.2f}m, "
          f"洪涝={node_state['flooding']:.4f} m³/s, "
          f"入流={node_state['total_inflow']:.2f} m³/s")

env.close()
```

### 交互式控制模式

通过命令行手动输入每个智能体的动作，适合调试：

```bash
python examples/manual_control.py --interactive
```

或加载自定义配置：

```bash
python examples/manual_control.py --config my_scenario.yaml --interactive
```

## 架构设计

```
 RL 智能体 (MAPPO / QMIX / PPO)
          │
          ▼
 ┌────────────────────────┐
 │  SWMMParallelEnv       │  PettingZoo ParallelEnv 封装层
 │  ┌──────────────────┐  │
 │  │  SWMMEnv (MDP)   │  │  核心 RL 逻辑: 观测/奖励/终止
 │  │  ┌─────────────┐ │  │
 │  │  │ SWMMEngine  │ │  │  PySWMM 仿真接口
 │  │  │ TimeSync    │ │  │  RL/SWMM 步长对齐
 │  │  │ Normalizer  │ │  │  观测/奖励标准化
 │  │  │ Mapping     │ │  │  智能体 ⇄ SWMM 元素注册表
 │  │  │ Reward Fn   │ │  │  可插拔奖励计算
 │  │  └─────────────┘ │  │
 │  └──────────────────┘  │
 └────────────────────────┘
          │
          ▼
   PySWMM (SWMM5 水力引擎)
```

## 配置说明

```yaml
# config/example.yaml
inp_file: "data/model.inp"    # SWMM 模型的 .inp 文件路径
rain_file: null               # 降雨时间序列文件（可选）
obs_raingage: RG1             # 观测雨量计的 ID

agents:                       # 智能体定义
  pump_1:
    type: pump                # 类型: pump / gate / weir
    link_id: P1               # 对应的 SWMM 链接 ID
    upstream_node: J1         # 上游节点
    downstream_node: J2       # 下游节点
  gate_1:
    type: weir
    link_id: W1
    upstream_node: J3
    downstream_node: J4

time_sync:
  decision_interval: 300      # RL 决策间隔（秒），如 300 = 5 分钟
  swmm_step: 10               # SWMM 路由步长（秒）

normalization:                # 归一化参数（z-score）
  obs:
    depth:  {mean: 2.0, std: 1.5}
    flow:   {mean: 0.5, std: 0.3}
    rainfall: {mean: 5.0, std: 10.0}
    setting: {mean: 0.5, std: 0.3}
  reward:
    mean: 0.0
    std: 10.0

reward_fn: "default_reward"   # 奖励函数名称
max_steps: 1000               # 最大 episode 步数
```



### 完整配置参数字段说明

以下是每个配置字段的完整说明：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `inp_file` | string | 是 | SWMM `.inp` 模型文件路径 |
| `rain_file` | string 或 null | 否 | 降雨时序文件路径；为 null 时使用 `.inp` 中定义的降雨 |
| `obs_raingage` | string | 否 | 用于观测降雨强度的雨量计 ID |
| `obs_nodes` | string 列表 | 否 | 需要通过 `get_state()` 获取完整状态的节点 ID |
| `agents` | dict | 是 | 智能体 ID 到智能体配置的映射（参见下方） |
| `time_sync` | dict | 是 | 时间同步参数 |
| `time_sync.decision_interval` | int | 是 | RL 决策间隔（秒），必须能被 `swmm_step` 整除 |
| `time_sync.swmm_step` | int | 是 | SWMM 路由步长（秒） |
| `normalization` | dict | 是 | 观测和奖励的归一化参数 |
| `normalization.obs` | dict | 是 | 各观测变量的 z-score 归一化均值和标准差 |
| `normalization.reward` | dict | 是 | 奖励的 z-score 归一化均值和标准差 |
| `reward_fn` | string/callable/object | 否 | 内置奖励函数名、自定义函数或 `CustomRewardFunction` 实例 |
| `reward_weights` | dict | 否 | 奖励组件权重（`flooding`, `level_deviation`, `energy`） |
| `target_levels` | dict | 否 | 各节点的目标水深，例如 `{J1: 1.5, J2: 1.0}` |
| `max_steps` | int | 否 | 每个 episode 的最大步数（默认：1000） |
| `warmup_steps` | int | 否 | 第一个 RL 决策前推进的模拟步数（默认：0） |
| `hotstart_file` | string 或 null | 否 | 用于快速 episode 重置的 `.hsf` 热启动文件路径 |
| `worker_index` | int | 否 | 并行训练的工作节点索引（默认：0） |

**智能体配置**（`agents` 下的每个条目）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | `pump`（泵站）、`gate`（闸门）或 `weir`（堰坝）之一 |
| `link_id` | string | SWMM 管道 ID，对应被控制的水力元件 |
| `upstream_node` | string | 上游节点 ID（包含在观测中） |
| `downstream_node` | string | 下游节点 ID（包含在观测中） |

**归一化变量**（位于 `normalization.obs` 下）：

每个变量（`depth`、`flow`、`rainfall`、`setting`）应设置 `mean` 和 `std`。观测值通过 z-score 归一化：`(value - mean) / std`。

## 智能体观测与动作空间

| 智能体类型 | 观测维度 | 特征组成 |
|-----------|---------|---------|
| pump（泵站）| 5 | 上游水深、下游水深、流量、当前开度、降雨强度 |
| gate（闸门）| 4 | 上游水深、下游水深、当前开度、降雨强度 |
| weir（堰坝）| 4 | 上游水深、下游水深、当前开度、降雨强度 |

- **观测空间**：`Box(-∞, +∞)`，经 z-score 归一化后
- **动作空间**：`Box(0, 1)`，连续控制开度

所有智能体共享 **全局奖励**（耦合排水系统的物理特性）。

## 奖励函数

内置四种奖励函数，通过 `reward_fn` 配置：

| 名称 | 说明 |
|------|------|
| `default_reward` | 洪涝惩罚 + 水位偏差 + 能耗效率 |
| `flooding_only` | 纯洪涝最小化 |
| `normalized_flooding` | 洪涝缩放到 `[-1, 0]` |
| `multi_objective` | 洪涝 (10x) + 水位跟踪 |

### 传入自定义奖励函数

有三种方式传入自定义奖励函数：

#### 1. 通过 YAML 配置（仅限内置函数）

在 YAML 文件中设置 `reward_fn` 字段为内置函数名：

```yaml
reward_fn: "multi_objective"
```

#### 2. 直接传入 callable（简单逻辑推荐）

定义签名为 `(engine, config) -> float` 的函数，赋给 `config["reward_fn"]`：

```python
from swmmEnv import SWMMParallelEnv, load_config

def my_reward(engine, config):
    flooding = engine.get_total_flooding()
    depth = engine.get_node_state("J1")["depth"]
    # 在这里写你的自定义逻辑
    return -flooding - abs(depth - 1.5)

config = load_config("config.yaml")
config["reward_fn"] = my_reward   # 用自定义函数覆盖
env = SWMMParallelEnv(config)
```

#### 3. 基于类的有状态奖励（跨步跟踪）

当奖励需要依赖历史状态（如惩罚开度剧烈变化）时，继承 `CustomRewardFunction`：

```python
from swmmEnv.reward.custom_reward import CustomRewardFunction

class StabilityReward(CustomRewardFunction):
    def __init__(self, flood_weight=1.0, stability_weight=0.1):
        super().__init__()
        self.flood_weight = flood_weight
        self.stability_weight = stability_weight
        self.prev_depths = {}

    def __call__(self, engine, config):
        # 洪涝惩罚
        reward = -self.flood_weight * engine.get_total_flooding()

        # 稳定性惩罚：惩罚水深的大幅变化
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
        """在新 episode 开始时重置内部状态。"""
        self.prev_depths = {}

# 使用方式
config["reward_fn"] = StabilityReward(flood_weight=2.0, stability_weight=0.2)
env = SWMMParallelEnv(config)
```

> **注意**：使用 `SWMMParallelEnv` 时，环境会在每个 episode 开始时自动调用奖励函数的 `reset()` 方法（如果存在）。

完整模板和更多示例见 `swmmEnv/reward/custom_reward.py`。

## API 参考

### `SWMMParallelEnv(config)`

主要的 PettingZoo 兼容多智能体环境。实现了 `pettingzoo.ParallelEnv`。

```python
from swmmEnv import SWMMParallelEnv

env = SWMMParallelEnv(config)
obs, infos = env.reset()
obs, rewards, terminations, truncations, infos = env.step(actions)
```

**核心方法：**

| 方法 | 签名 | 说明 |
|------|------|------|
| `reset` | `(seed, options) -> (observations, infos)` | 开始新 episode，返回初始观测 |
| `step` | `(actions) -> (obs, rewards, terms, truncs, infos)` | 应用动作并推进模拟一个决策间隔 |
| `observation_space` | `(agent) -> Box` | 获取智能体的观测空间（RLlib 兼容） |
| `action_space` | `(agent) -> Box` | 获取智能体的动作空间（RLlib 兼容） |
| `observe` | `(agent) -> ndarray` | 获取特定智能体的当前观测 |
| `state` | `() -> ndarray` | 获取拼接后的全局观测（用于集中式评论家算法） |
| `render` | `(mode) -> None` | 将当前环境状态打印到控制台 |
| `close` | `() -> None` | 释放模拟资源 |
| `get_env_info` | `() -> dict` | 获取用于 MARLlib 的环境元数据 |

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `agents` | list | 当前活跃的智能体 ID 列表 |
| `possible_agents` | list | 场景中所有智能体 ID |
| `core_env` | SWMMEnv | 底层核心 MDP 环境 |
| `observation_spaces` | dict | 每个智能体的观测空间 |
| `action_spaces` | dict | 每个智能体的动作空间 |

### `SWMMMultiAgentEnv(env_config)`

RLlib 兼容的 `MultiAgentEnv`，封装 `SWMMParallelEnv`。继承自 `ray.rllib.env.MultiAgentEnv`。

```python
from swmmEnv import SWMMMultiAgentEnv

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, infos = env.reset()
obs, rewards, terms, truncs, infos = env.step(actions)
```

`env_config` 接受两种形式：
- `{"config_path": "/path/to/config.yaml"}` — RLlib 风格；解析并加载 YAML 文件
- `{"inp_file": ..., "agents": ..., ...}` — 完整配置字典，直接透传

**核心方法：**

| 方法 | 签名 | 说明 |
|------|------|------|
| `reset` | `(*, seed, options) -> (obs, infos)` | 开始新 episode |
| `step` | `(action_dict) -> (obs, rewards, terms, truncs, infos)` | 推进环境；`terms`/`truncs` 包含 `__all__` 键 |
| `close` | `() -> None` | 释放模拟资源 |
| `render` | `(mode) -> None` | 打印当前环境状态 |

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `possible_agents` | list | 所有智能体 ID（替代已弃用的 `get_agent_ids()`） |
| `observation_space` | `spaces.Dict` | 字典映射 agent ID → `Box`（兼容旧 API） |
| `action_space` | `spaces.Dict` | 字典映射 agent ID → `Box`（兼容旧 API） |
| `observation_spaces` | dict | 每个智能体的观测空间（新 API） |
| `action_spaces` | dict | 每个智能体的动作空间（新 API） |

### `make_rllib_env(config_path)`

`swmmEnv.envs.register_env` 中的便捷工厂函数：

```python
from swmmEnv.envs.register_env import make_rllib_env

env = make_rllib_env("configs/control.yaml")
```

### `SWMMEnv(config)`

核心 MDP 环境，独立于 PettingZoo。被 `SWMMParallelEnv` 内部使用。

```python
from swmmEnv import SWMMEnv

env = SWMMEnv(config)
obs = env.reset()
obs, reward, done, info = env.step({"pump_1": 0.8, "gate_1": 0.5})
```

**核心方法：** `reset()`, `step(action_dict)`, `get_observation(agent_id)`, `get_reward()`, `get_state()`, `render(mode)`, `close()`。

### `SWMMEngine(inp_file, config, worker_index=0)`

底层 PySWMM 模拟封装。处理模拟生命周期、状态检索和动作应用。

```python
from swmmEnv import SWMMEngine

engine = SWMMEngine("model.inp", config, worker_index=0)
engine.start()
engine.apply_action("pump_1", 0.8)
engine.step()

# 检索状态
node = engine.get_node_state("J1")
link = engine.get_link_state("P1")
flooding = engine.get_total_flooding()
rainfall = engine.get_rainfall("RG1")
time = engine.get_current_time()

engine.close()
```

**关键工具方法：**

| 方法 | 说明 |
|------|------|
| `start()` | 开始模拟并注册步进前后的回调函数 |
| `step()` | 推进模拟一个控制间隔 |
| `reset()` | 使用热启动文件快速重置模拟 |
| `close()` | 释放模拟资源并清理工作节点文件 |
| `apply_action(agent_id, setting)` | 为智能体排队一个 `[0, 1]` 范围内的控制动作 |
| `get_node_state(node_id)` | 获取节点的水深、水头、体积、洪涝、入流 |
| `get_link_state(link_id)` | 获取管道的流量、水深、体积、当前开度 |
| `get_rainfall(gage_id)` | 获取雨量计的降雨强度（mm/h） |
| `get_total_flooding()` | 获取所有节点的总洪涝率（m³/s） |
| `get_system_stats()` | 获取路由和径流统计信息 |
| `is_ended()` | 检查模拟是否已到达结束时间 |
| `get_current_time()` | 获取当前模拟时间 |
| `save_hotstart(filepath)` | 将当前状态保存为 `.hsf` 热启动文件 |

### `TimeSync(decision_interval, swmm_step)`

管理 RL 决策步和 SWMM 模拟步之间的同步。`decision_interval` 必须能被 `swmm_step` 整除。

```python
from swmmEnv.sim import TimeSync

ts = TimeSync(decision_interval=300, swmm_step=10)
print(ts.skip_steps)  # 每个 RL 步 = 30 个 SWMM 步
ts.advance(engine)    # 推进引擎 30 个 SWMM 步
```

**核心方法：** `advance(engine)`, `reset()`, `should_act(step)`, `get_elapsed_time()`, `get_elapsed_time_minutes()`。

### `StateNormalizer(config)`

观测和奖励的 z-score 归一化，用于稳定训练。

```python
from swmmEnv.sim import StateNormalizer

normalizer = StateNormalizer(config["normalization"])
obs_normalized = normalizer.normalize_obs(raw_obs, obs_names)
reward_normalized = normalizer.normalize_reward(raw_reward)
reward_denorm = normalizer.denormalize_reward(normalized_reward)
```

**核心方法：** `normalize_obs(obs, obs_names)`, `normalize_obs_value(value, name)`, `normalize_reward(reward)`, `denormalize_reward(normalized_reward)`, `min_max_normalize(value, min, max)`, `clip_and_normalize(value, name, clip_range)`, `update_stats(obs, obs_names)`。

### `MappingRegistry(agents_config)`

智能体到 SWMM 元件的映射注册表。

```python
from swmmEnv.sim import MappingRegistry

registry = MappingRegistry(config["agents"])
print(registry.get_all_agents())        # ["pump_1", "gate_1"]
print(registry.get_element_type("pump_1"))  # "pump"
print(registry.get_element_id("pump_1"))    # "P1"
print(registry.get_upstream_node("pump_1")) # "J1"
```

**核心方法：** `get_element_id(agent)`, `get_element_type(agent)`, `get_upstream_node(agent)`, `get_downstream_node(agent)`, `get_agent_config(agent)`, `get_all_agents()`, `get_agents_by_type(type)`, `get_all_link_ids()`, `get_all_node_ids()`, `agent_exists(agent)`。

### `load_config(config_path=None, merge_defaults=True)`

加载并验证 YAML 配置文件。

```python
from swmmEnv import load_config

# 从文件加载
config = load_config("path/to/config.yaml")

# 仅加载默认配置
config = load_config()

# 不合并默认配置（用于 make_env 的自定义环境设置）
config = load_config("path/to/config.yaml", merge_defaults=False)
```

### `validate_config(config)`

验证配置字典。如果缺少必需字段或字段无效，会抛出 `ValueError` 并附带描述性消息。

```python
from swmmEnv.config import validate_config

validate_config(config)  # 配置无效时抛出 ValueError
```

## 训练

### RLlib 直接集成（推荐 — 新 API）

使用 `SWMMMultiAgentEnv` 最简单的方式：直接传递给 `PPOConfig().environment()`。

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

### Ray RLlib（经典 register_env 模式）

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

### MARLlib 集成（推荐）

SWMMEnv 通过独立的注册机制与 [MARLlib](https://github.com/Replicable-MARL/MARLlib) 集成。

**第一步：注册环境**

```python
from swmmEnv.envs.register_env import register_with_marllib

register_with_marllib()
# 现在可以在 MARLlib 配置中使用 `env: "swmm"`
```

**第二步：创建环境并训练**

```python
from marllib import marl
from swmmEnv.envs.register_env import make_env

# 创建环境
env = make_env(config_path="config/example.yaml")

# 构建 MAPPO 模型
model = marl.build_model(
    environment=env,
    algorithm=marl.algos.mappo,
    model_preference={
        "core_arch": "mlp",
        "encode_layer": "128-128",
        "hidden_dim": 64,
    }
)

# 开始训练
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

**直接使用 `make_env`（无需永久注册）：**

```python
from swmmEnv.envs.register_env import make_env

# 指定配置路径
env = make_env(config_path="config/my_scenario.yaml", worker_index=0)

# 通过 map_name 查找（搜索 configs/, config/, 当前目录）
env = make_env(map_name="control", worker_index=0)
```

`make_env` 始终加载配置时**不合并默认配置**，避免默认配置的占位智能体导致的冲突。

### 并行 Worker 训练

SWMMEnv 支持分布式训练的并行环境工作节点。每个 worker 获得一个隔离的 `.inp` 文件副本，防止文件锁冲突：

```python
from swmmEnv.envs.register_env import make_env

# Worker 0（使用原始 .inp 文件）
env_0 = make_env(config_path="config/example.yaml", worker_index=0)

# Worker 1（获得一个临时 .inp 副本）
env_1 = make_env(config_path="config/example.yaml", worker_index=1)

# Worker 2（获得另一个隔离副本）
env_2 = make_env(config_path="config/example.yaml", worker_index=2)
```

为提升跨 episode 的重置效率，引擎使用**热启动文件**来恢复模拟状态，无需重新创建 `Simulation` 对象（对于 RL 训练循环来说要快得多）。

**训练关键配置：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `warmup_steps` | 0-10 | 第一个 RL 决策前的步数；有助于稳定初始条件 |
| `max_steps` | 500-2000 | Episode 长度；根据暴雨事件时长调整 |
| `hotstart_file` | 自动 | 自动创建，提供快速重置 |
| 归一化参数 | 从数据中校准 | 基于历史模拟运行设置 `mean`/`std` |

## 项目结构

```
swmmEnv/
├── swmmEnv/
│   ├── __init__.py              # 包入口
│   ├── sim/                     # 仿真模块
│   │   ├── engine.py            # PySWMM 仿真封装
│   │   ├── time_sync.py         # RL 与 SWMM 步长同步
│   │   ├── normalizer.py        # 观测/奖励 z-score 归一化
│   │   └── mapping.py           # 智能体 ⇄ SWMM 元素注册
│   ├── envs/                    # 环境模块
│   │   ├── swmm_env/
│   │   │   ├── env.py           # 核心 MDP 环境
│   │   │   ├── pettingzoo_env.py # PettingZoo ParallelEnv 封装
│   │   │   └── rllib_env.py     # RLlib MultiAgentEnv 适配器
│   │   └── register_env.py      # MARLlib / RLlib 注册辅助函数
│   ├── reward/                  # 奖励模块
│   │   ├── default_reward.py    # 内置奖励函数集合
│   │   └── custom_reward.py     # 自定义奖励模板
│   └── config/                  # 配置模块
│       ├── loader.py            # YAML 配置加载与校验
│       └── default_config.yaml  # 默认配置
├── examples/
│   ├── manual_control.py        # 交互式调试脚本
│   └── train_mappo.py           # MAPPO 训练示例
├── tests/                       # 单元测试
├── pyproject.toml
├── README.md                    # 英文文档
├── README_JP.md                 # 日本語ドキュメント
└── README_TW.md                 # 繁體中文文档
```

## 环境要求

- Python ≥ 3.8
- PySWMM ≥ 2.1.0
- PettingZoo ≥ 1.24.0
- Gymnasium ≥ 0.29.0
- NumPy ≥ 1.24.0
- PyYAML ≥ 6.0

## 引用

```bibtex
@software{swmmEnv2025,
  author = {dujddx},
  title = {SWMMEnv: Multi-agent RL Environment for SWMM Stormwater Simulation},
  url = {https://github.com/DujDDx/swmmEnv},
  version = {0.1.0},
  year = {2026}
}
```




## 开源协议

MIT License。详见 [LICENSE](LICENSE) 文件。
## 运行示例与测试

### 示例脚本

仓库包含可直接运行的示例，位于 `examples/` 目录：

```bash
# 随机控制（默认配置，1 个 episode）
python examples/manual_control.py

# 随机控制（自定义配置，5 个 episodes）
python examples/manual_control.py --config my_config.yaml --episodes 5

# 交互模式（手动输入动作）
python examples/manual_control.py --interactive

# 使用 MARLlib 训练 MAPPO
python examples/train_mappo.py

# 使用自定义配置训练 MAPPO
python examples/train_mappo.py --config my_config.yaml --steps 50000

# 直接使用 RLlib 训练
python examples/train_mappo.py --backend rllib --steps 100000
```

### 运行测试

运行单元测试套件验证安装：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定组件的测试
pytest tests/test_engine.py -v
pytest tests/test_mapping.py -v
pytest tests/test_normalizer.py -v
pytest tests/test_time_sync.py -v
pytest tests/test_swmm_env.py -v
pytest tests/test_pettingzoo_env.py -v

# 带覆盖率报告
pytest tests/ --cov=swmmEnv --cov-report=term-missing
```

## 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ValueError: decision_interval must be divisible by swmm_step` | `time_sync` 配置无效 | 确保 `decision_interval % swmm_step == 0` |
| `FileNotFoundError: Configuration file not found` | 配置路径错误 | 使用绝对路径或相对于当前目录的路径 |
| `PySWMM simulation not started` 相关错误 | `.inp` 文件缺失或路径错误 | 验证 `inp_file` 是否存在且路径正确 |
| RL 训练梯度不稳定 | 归一化参数不当 | 基于历史模拟数据校准 `mean`/`std` |
| 环境重置速度慢 | 未使用热启动文件 | Worker 会自动创建热启动；检查 `_initial_hotstart` 是否已设置 |
| `RuntimeError: Environment must be reset before stepping` | `step()` 前未调用 `reset()` | 确保先调用 `env.reset()` |
| 并发模拟崩溃 | 多个 PySWMM 实例使用同一 `.inp` | 为每个 worker 设置 `worker_index > 0` 以获得隔离的 `.inp` 副本 |
| 实例化 `SWMMMultiAgentEnv` 时报 `ImportError: ray[rllib] is required` | 未安装 ray | 运行 `pip install "swmmEnv[marl]"` 或 `pip install ray[rllib]` |
