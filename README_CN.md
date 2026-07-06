# SWMMEnv · 暴雨洪水管理模型多智能体强化学习环境

<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README.md">English</a>
</p>

基于 [PettingZoo](https://pettingzoo.farama.org/) 和 [PySWMM](https://github.com/pyswmm/pyswmm) 构建的 **SWMM 暴雨洪水管理模型多智能体强化学习环境**。

---

## 概述

SWMMEnv 将 EPA SWMM 水力模拟封装为 PettingZoo 兼容的多智能体 RL 环境。SWMM 模型中的每个泵站、闸门和堰坝都成为一个可控制的 RL 智能体，支持训练以下策略：

- **防洪减灾** — 最小化耦合排水网络的洪水溢流
- **水位调控** — 维持蓄水节点的目标水位
- **节能抽排** — 在防洪与能耗之间取得平衡
- **实时暴雨控制** — 降雨不确定性下的自适应控制策略

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

自定义奖励 — 传入任意 `callable(engine, config) -> float`：

```python
def my_reward(engine, config):
    flooding = engine.get_total_flooding()
    depth = engine.get_node_state("J1")["depth"]
    return -flooding - abs(depth - 1.5)

config["reward_fn"] = my_reward
```

有状态的奖励函数类模板见 `swmmEnv/reward/custom_reward.py`。

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
  year = {2025}
}
```

## 开源协议

MIT License。详见 [LICENSE](LICENSE) 文件。
