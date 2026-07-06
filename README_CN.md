# SWMMEnv · 暴雨洪水管理模型多智能体强化学习环境
[![python swimming](assets/pythonSwimming.jpg)](https://martinparr.com/)
<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README.md">English</a>
</p>

基于 [PettingZoo](https://pettingzoo.farama.org/) 和 [PySWMM](https://github.com/pyswmm/pyswmm) 构建的 **SWMM 暴雨洪水管理模型多智能体强化学习环境**。

---

## 概述

SWMMEnv 将 EPA SWMM 水力模拟封装为 PettingZoo 兼容的多智能体 RL 环境。SWMM 模型中的每个泵站、闸门和堰坝都成为一个可控制的 RL 智能体

## 功能特性

- **PettingZoo ParallelEnv API** — 无缝对接 MARLlib、RLlib、Tianshou 等框架
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

## 快速开始

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

## MARLlib 集成

```python
from swmmEnv.envs.register_env import make_env

env = make_env(config_path="config/my_scenario.yaml", worker_index=0)
```

或注册到 MARLlib 环境注册表：

```python
from swmmEnv.envs.register_env import register_with_marllib
register_with_marllib()
# 之后在 MARLlib 配置中使用 `env: swmm`
```

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
│   │   │   └── pettingzoo_env.py # PettingZoo ParallelEnv 封装
│   │   └── register_env.py      # MARLlib 注册辅助函数
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
└── README_CN.md                 # 中文文档（本文件）
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
