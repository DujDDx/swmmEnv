# SWMMEnv · 豪雨洪水管理模型多智能體強化學習環境

[![python swimming](assets/pythonSwimming.jpg)](https://martinparr.com/)
<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README.md">English</a> · <a href="README_CN_simplified.md">简体中文</a> · <a href="README_JP.md">日本語</a>
</p>

基於 [PettingZoo](https://pettingzoo.farama.org/) 與 [PySWMM](https://github.com/pyswmm/pyswmm) 打造的 **SWMM 豪雨洪水管理模型多智能體強化學習環境**。

---

## 概述

SWMMEnv 將 EPA SWMM 水力模擬封裝為 PettingZoo 相容且 RLlib 原生支援的多智能體 RL 環境。SWMM 模型中的每個抽水站、閘門及堰壩都成為一個可控的 RL 智能體。

## 功能特色

- **PettingZoo ParallelEnv API** — 無縫整合 MARLlib、RLlib、Tianshou 等框架
- **RLlib MultiAgentEnv API** — 直接傳入類別名稱即可，無需手動編寫轉接器
- **設定驅動** — 更換 SWMM 模型無需修改程式碼
- **多種智能體類型** — 抽水站、閘門、堰壩，支援連續控制 `[0, 1]`
- **時間同步** — 將 RL 決策間隔與 SWMM 路由步長解耦
- **狀態正規化** — 支援 z-score 與 min-max 正規化，穩定訓練過程
- **可插拔獎勵函數** — 內建洪災、水位偏差、能耗等多目標獎勵元件
- **並行 Worker 支援** — 隔離 `.inp` 副本，安全支援多 Worker 並行訓練

## 安裝方式

```bash
pip install swmmEnv
```

安裝 MARL 訓練相關依賴：

```bash
pip install "swmmEnv[marl]"
```

安裝開發依賴：

```bash
pip install "swmmEnv[dev]"
```

> `ray[rllib]` 為**選擇性**依賴。您可以在未安裝 ray 的情況下匯入 `SWMMMultiAgentEnv`，但實際建立實體時仍需安裝 ray。

## 快速開始

### 基本用法（PettingZoo API）

```python
from swmmEnv import SWMMParallelEnv, load_config

# 載入設定檔
config = load_config("config/example.yaml")

# 建立環境
env = SWMMParallelEnv(config)

# 重設環境
observations, info = env.reset()

# 執行一步：每個智能體輸出連續動作 [0, 1]
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### RLlib API（SWMMMultiAgentEnv）

`SWMMMultiAgentEnv` 將 `SWMMParallelEnv` 封裝為 RLlib 相容的 `MultiAgentEnv`。直接傳入類別名稱即可，無需工廠函式或手動轉接器。

**直接傳入類別名稱**（RLlib 2.x+）：

```python
from swmmEnv import SWMMMultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        SWMMMultiAgentEnv,
        env_config={"config_path": "configs/control.yaml"},
    )
    .multi_agent(
        policies={"shared_policy": None},
        policy_mapping_fn=lambda agent_id: "shared_policy",
    )
)
algo = config.build()
```

**透過 register_env**（傳統方式）：

```python
from ray.tune.registry import register_env
from swmmEnv.envs.register_env import make_rllib_env

register_env("swmm_env", lambda cfg: make_rllib_env(config_path="configs/control.yaml"))
```

**直接實例化**（測試 / 除錯）：

```python
from swmmEnv import SWMMMultiAgentEnv
import numpy as np

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, info = env.reset()
actions = {aid: np.array([0.5]) for aid in env.possible_agents}
obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

### 使用預設設定

如果您還沒有 `.inp` 檔案，可以載入套件內建的預設設定來了解環境結構：

```python
from swmmEnv import load_config

# 載入內建預設設定（不傳路徑 = 載入預設設定）
config = load_config()
print(config)
```

### 單獨使用 MDP 環境（不依賴 PettingZoo）

直接使用核心環境 `SWMMEnv` 進行測試或自訂整合：

```python
from swmmEnv import SWMMEnv, load_config

config = load_config("config/example.yaml")
env = SWMMEnv(config)

obs = env.reset()
actions = {"pump_1": 0.8, "gate_1": 0.5}
obs, reward, done, info = env.step(actions)

print(f"獎勵: {reward:.3f}, 結束: {done}")
env.close()
```

### 手動隨機控制

執行完整的 episode 來確認環境是否正常運作：

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
    obs, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += sum(rewards.values())
    step += 1
    done = all(terminations.values()) or all(truncations.values())

print(f"Episode 執行完畢: {step} 步, 總獎勵: {total_reward:.2f}")
env.close()
```

### 內建預設設定

```yaml
# config/default_config.yaml
inp_file: "examples/swmm_model.inp"   # SWMM 模型檔案
swmm_step: 300                        # SWMM 路由步長（秒）
decision_interval: 900                # RL 決策間隔（秒）
max_steps: 96                         # 最大步數（例如：24小時，900秒間隔）
warmup_steps: 0                       # 暖機步數

agents:
  - name: pump_1                     # 智能體名稱 → SWMM 元件
    type: pump                        # 智能體類型
    control_type: continuous          # 控制類型
  - name: gate_1
    type: gate
    control_type: continuous
  - name: weir_1
    type: weir
    control_type: continuous

reward:
  type: default                       # 獎勵函數類型
  flooding_penalty: 10.0              # 洪災懲罰係數
  level_deviation_penalty: 1.0        # 水位偏差懲罰係數
  energy_penalty: 0.1                 # 能耗懲罰係數

normalization:
  method: zscore                      # 正規化方式 (zscore / minmax)
```

## 設定指南

透過 YAML 設定檔來精細控制 SWMMEnv 的行為。

### agent 設定

`agents` 清單定義了每個 RL 智能體與 SWMM 元件的對應關係：

```yaml
agents:
  - name: pump_1        # RL 智能體識別碼
    type: pump          # SWMM 元件類型（pump / gate / weir / orifice）
    control_type: continuous  # 控制類型（continuous / discrete）
```

支援的 SWMM 元件：`pump`、`gate`（透過定速泵控制模擬閘門）、`weir`、`orifice`。
每個元件皆支援連續控制 `[0, 1]`（0 = 全關、1 = 全開）。

### reward 設定

內建獎勵函數的權重參數：

```yaml
reward:
  type: default
  flooding_penalty: 10.0          # 節點溢流懲罰
  level_deviation_penalty: 1.0    # 偏離目標水位的懲罰
  energy_penalty: 0.1             # 泵浦能耗懲罰
```

### normalization 設定

```yaml
normalization:
  method: zscore   # 或 minmax
```

`zscore` 方式需要事先校正好的 `mean` / `std` 參數。

### time_sync 設定

```yaml
swmm_step: 300          # SWMM 路由步長（秒），需與 PySWMM 的 stepsize 一致
decision_interval: 900  # RL 決策間隔（秒），必須為 swmm_step 的整數倍
max_steps: 96           # Episode 最大長度
warmup_steps: 0         # 第一個 RL 決策前的暖機步數
```

### 環境設定的自訂方式

#### 自訂獎勵函數

繼承 `CustomReward` 類別來打造自訂獎勵函數：

```python
from swmmEnv.reward.default_reward import DefaultReward

class MyReward(DefaultReward):
    def __init__(self, config):
        super().__init__(config)
        # 加入自訂參數

    def calculate(self, env):
        # 自訂獎勵邏輯
        flooding = env.get_flooding()
        efficiency = env.get_energy_efficiency()
        return -flooding * 5.0 + efficiency * 0.5
```

在設定中指定自訂獎勵類別：

```yaml
reward:
  type: custom            # 使用自訂獎勵函數
  custom_class: my_module.MyReward  # 完整類別名稱
  param1: value1          # 自訂參數
```

#### 自訂狀態正規化

```python
from swmmEnv.sim.normalizer import ZScoreNormalizer
import numpy as np

# 從歷史資料計算正規化參數
obs_history = collect_historical_observations()
normalizer = ZScoreNormalizer()
normalizer.calibrate(obs_history)  # 自動計算 mean/std

# 寫入設定
config["normalization"] = {
    "method": "zscore",
    "mean": normalizer.mean.tolist(),
    "std": normalizer.std.tolist(),
}
```

## 訓練指南

### 分散式訓練（RLlib）

`SWMMMultiAgentEnv` 可直接在 RLlib 的遠端 Worker 上運作。Worker 會自動將設定中的 `inp_file` 路徑轉換為隔離副本，確保並行安全。

**重要注意：** 在 RLlib 中使用分散式訓練時，請在設定中指定原始的 `.inp` 檔案路徑即可，Worker 會自動建立隔離副本。具體設定範例請參考 `examples/train_mappo.py`。

### 訓練效能最佳化

為提升跨 episode 的重設效率，引擎使用**熱啟動檔案**來還原模擬狀態，無需重新建立 `Simulation` 物件（對於 RL 訓練迴圈來說效率大幅提升）。

**訓練關鍵參數：**

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `warmup_steps` | 0-10 | 第一個 RL 決策前的步數；有助於穩定初始條件 |
| `max_steps` | 500-2000 | Episode 長度；依豪雨事件長度調整 |
| `hotstart_file` | 自動 | 自動建立，提供快速重設 |
| 正規化參數 | 從資料校正 | 根據歷史模擬執行設定 `mean`/`std` |

## 專案結構

```
swmmEnv/
├── swmmEnv/
│   ├── __init__.py              # 套件入口
│   ├── sim/                     # 模擬模組
│   │   ├── engine.py            # PySWMM 模擬封裝
│   │   ├── time_sync.py         # RL 與 SWMM 步長同步
│   │   ├── normalizer.py        # 觀測/獎勵 z-score 正規化
│   │   └── mapping.py           # 智能體 ⇄ SWMM 元件註冊
│   ├── envs/                    # 環境模組
│   │   ├── swmm_env/
│   │   │   ├── env.py           # 核心 MDP 環境
│   │   │   ├── pettingzoo_env.py # PettingZoo ParallelEnv 封裝
│   │   │   └── rllib_env.py     # RLlib MultiAgentEnv 轉接器
│   │   └── register_env.py      # MARLlib / RLlib 註冊輔助函式
│   ├── reward/                  # 獎勵模組
│   │   ├── default_reward.py    # 內建獎勵函數集合
│   │   └── custom_reward.py     # 自訂獎勵模板
│   └── config/                  # 設定模組
│       ├── loader.py            # YAML 設定載入與驗證
│       └── default_config.yaml  # 預設設定
├── examples/
│   ├── manual_control.py        # 互動式除錯腳本
│   └── train_mappo.py           # MAPPO 訓練範例
├── tests/                       # 單元測試
├── pyproject.toml
├── README.md                    # 英文文件
├── README_CN_simplified.md      # 簡體中文文件
├── README_JP.md                 # 日文文件
└── README_CN_traditional.md     # 繁體中文文件（本文件）
```

## 環境需求

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

## 授權條款

MIT License。詳見 [LICENSE](LICENSE) 檔案。

## 執行範例與測試

### 範例腳本

倉庫包含可直接執行的範例，位於 `examples/` 目錄：

```bash
# 隨機控制（預設設定，1 個 episode）
python examples/manual_control.py

# 隨機控制（自訂設定，5 個 episodes）
python examples/manual_control.py --config my_config.yaml --episodes 5

# 互動模式（手動輸入動作）
python examples/manual_control.py --interactive

# 使用 MARLlib 訓練 MAPPO
python examples/train_mappo.py

# 使用自訂設定訓練 MAPPO
python examples/train_mappo.py --config my_config.yaml --steps 50000

# 直接使用 RLlib 訓練
python examples/train_mappo.py --backend rllib --steps 100000
```

### 執行測試

執行單元測試套件來驗證安裝：

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定元件的測試
pytest tests/test_engine.py -v
pytest tests/test_mapping.py -v
pytest tests/test_normalizer.py -v
pytest tests/test_time_sync.py -v
pytest tests/test_swmm_env.py -v
pytest tests/test_pettingzoo_env.py -v

# 附涵蓋率報告
pytest tests/ --cov=swmmEnv --cov-report=term-missing
```

## 常見問題排除

| 問題 | 可能原因 | 解決方式 |
|------|----------|----------|
| `ValueError: decision_interval must be divisible by swmm_step` | `time_sync` 設定無效 | 確認 `decision_interval % swmm_step == 0` |
| `FileNotFoundError: Configuration file not found` | 設定路徑錯誤 | 使用絕對路徑或相對於目前目錄的路徑 |
| 與 `PySWMM simulation not started` 相關的錯誤 | `.inp` 檔案不存在或路徑錯誤 | 確認 `inp_file` 是否存在且路徑正確 |
| RL 訓練梯度不穩定 | 正規化參數不當 | 依據歷史模擬資料校正 `mean`/`std` |
| 環境重設速度慢 | 未使用熱啟動檔案 | Worker 會自動建立熱啟動；檢查 `_initial_hotstart` 是否已設定 |
| `RuntimeError: Environment must be reset before stepping` | `step()` 前未呼叫 `reset()` | 確保先執行 `env.reset()` |
| 並行模擬崩潰 | 多個 PySWMM 實例使用相同的 `.inp` | 為每個 worker 設定 `worker_index > 0` 以獲得隔離的 `.inp` 副本 |
| 實例化 `SWMMMultiAgentEnv` 時出現 `ImportError: ray[rllib] is required` | 未安裝 ray | 執行 `pip install "swmmEnv[marl]"` 或 `pip install ray[rllib]` |
