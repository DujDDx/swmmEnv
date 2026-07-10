# SWMMEnv · 豪雨水管理モデルのマルチエージェント強化学習環境

[![python swimming](assets/pythonSwimming.jpg)](https://martinparr.com/)
<p align="center">
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/v/swmmEnv?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/swmmEnv"><img src="https://img.shields.io/pypi/pyversions/swmmEnv" alt="Python"></a>
  <a href="https://github.com/DujDDx/swmmEnv/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="README.md">English</a> · <a href="README_CN_simplified.md">简体中文</a> · <a href="README_CN_traditional.md">繁體中文</a>
</p>

[PettingZoo](https://pettingzoo.farama.org/) と [PySWMM](https://github.com/pyswmm/pyswmm) をベースに構築された **SWMM 豪雨水管理モデルのマルチエージェント強化学習環境**。

---

## 概要

SWMMEnv は EPA SWMM の水理シミュレーションを、PettingZoo 互換かつ RLlib ネイティブなマルチエージェント RL 環境としてラップします。SWMM モデル内の各ポンプ、ゲート、堰が制御可能な RL エージェントになります。

## 特徴

- **PettingZoo ParallelEnv API** — MARLlib、RLlib、Tianshou などとシームレスに連携
- **RLlib MultiAgentEnv API** — クラス名を直接渡すだけで OK、手動アダプター不要
- **設定駆動型** — SWMM モデルの変更にコード修正は不要
- **多様なエージェントタイプ** — ポンプ、ゲート、堰、連続制御と離散制御に対応
- **時刻同期** — RL の決定間隔と SWMM のルーティングステップを分離
- **状態正規化** — z-score または min-max 正規化で安定した学習を実現
- **プラグ可能な報酬関数** — 洪水、水位偏差、エネルギー消費などの多目的報酬コンポーネントを内蔵
- **並列ワーカー対応** — 分離された .inp コピーにより、安全なマルチワーカー並列学習をサポート

## インストール

```bash
pip install swmmEnv
```

MARL 学習用の追加依存関係：

```bash
pip install "swmmEnv[marl]"
```

開発用依存関係：

```bash
pip install "swmmEnv[dev]"
```

> `ray[rllib]` は**オプション**の依存関係です。`SWMMMultiAgentEnv` のインポート自体は ray がなくても可能ですが、クラスのインスタンス化には ray のインストールが必要です。

## クイックスタート

### 基本使用法（PettingZoo API）

```python
from swmmEnv import SWMMParallelEnv, load_config

# 設定をロード
config = load_config("config/example.yaml")

# 環境を作成
env = SWMMParallelEnv(config)

# 環境をリセット
observations, info = env.reset()

# 1ステップ：各エージェントが動作を出力（連続値または離散インデックス）
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### RLlib API（SWMMMultiAgentEnv）

`SWMMMultiAgentEnv` は `SWMMParallelEnv` を RLlib 互換の `MultiAgentEnv` としてラップします。クラス名を直接渡すだけで、ファクトリ関数や手動アダプターは不要です。

**クラス名を直接渡す**（RLlib 2.x+）：

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

**register_env を使用**（従来の方法）：

```python
from ray.tune.registry import register_env
from swmmEnv.envs.register_env import make_rllib_env

register_env("swmm_env", lambda cfg: make_rllib_env(config_path="configs/control.yaml"))
```

**直接インスタンス化**（テスト / デバッグ）：

```python
from swmmEnv import SWMMMultiAgentEnv
import numpy as np

env = SWMMMultiAgentEnv({"config_path": "configs/control.yaml"})
obs, info = env.reset()
actions = {aid: np.array([0.5]) for aid in env.possible_agents}
obs, rewards, terms, truncs, infos = env.step(actions)
env.close()
```

### デフォルト設定を使用する

まだ .inp ファイルがない場合は、パッケージに内蔵されたデフォルト設定をロードして環境構造を確認できます：

```python
from swmmEnv import load_config

# 内蔵デフォルト設定をロード（パスを指定しない = デフォルト設定）
config = load_config()
print(config)
```

### MDP 環境を単体で使用する（PettingZoo 非依存）

コア環境 `SWMMEnv` を直接使用してテストやカスタム統合を行う：

```python
from swmmEnv import SWMMEnv, load_config

config = load_config("config/example.yaml")
env = SWMMEnv(config)

obs = env.reset()
actions = {"pump_1": 0.8, "gate_1": 0.5}
obs, reward, done, info = env.step(actions)

print(f"報酬: {reward:.3f}, 終了: {done}")
env.close()
```

### 手動ランダム制御

完全なエピソードを実行して環境が正しく動作することを確認する：

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

print(f"実行エピソード完了: {step} ステップ, 総報酬: {total_reward:.2f}")
env.close()
```

### 内蔵デフォルト設定

```yaml
# config/default_config.yaml
inp_file: "examples/swmm_model.inp"   # SWMM モデルファイル
swmm_step: 300                        # SWMM ルーティングステップ（秒）
decision_interval: 900                # RL 決定間隔（秒）
max_steps: 96                         # 最大ステップ数（例：24時間、900秒間隔）
warmup_steps: 0                       # ウォームアップステップ数

agents:
  - name: pump_1                     # エージェント名 → SWMM 要素
    type: pump                        # エージェントタイプ
    control_type: continuous          # 制御タイプ
  - name: gate_1
    type: gate
    control_type: continuous
  - name: weir_1
    type: weir
    control_type: continuous

reward:
  type: default                       # 報酬関数タイプ
  flooding_penalty: 10.0              # 洪水ペナルティ係数
  level_deviation_penalty: 1.0        # 水位偏差ペナルティ係数
  energy_penalty: 0.1                 # エネルギー消費ペナルティ係数

normalization:
  method: zscore                      # 正規化方式 (zscore / minmax)
```

## 設定ガイド

YAML 設定ファイルを通じて SWMMEnv の動作を細かく制御できます。

### agent 設定

`agents` リストで各 RL エージェントの SWMM 要素との対応関係を定義します：

```yaml
agents:
  - name: pump_1        # RL エージェント識別子
    type: pump          # SWMM 要素タイプ（pump / gate / weir / orifice）
    control_type: continuous  # 制御タイプ（continuous / discrete）
```

サポートされる SWMM 要素：pump、gate（定速ポンプ制御によるゲート模擬）、weir、orifice。
各要素は連続制御 [0, 1]（0 = 全閉、1 = 全開）および離散制御（`action_space` 設定による）をサポートします。

### reward 設定

内蔵報酬関数の重みパラメータ：

```yaml
reward:
  type: default
  flooding_penalty: 10.0          # ノード溢水分のペナルティ
  level_deviation_penalty: 1.0    # 目標水位からの偏差ペナルティ
  energy_penalty: 0.1             # ポンプエネルギー消費ペナルティ
```

## アクション空間設定

アクション空間のタイプは、オプションの `action_space` 設定セクションで制御します。省略した場合、環境は連続 `Box(0, 1, shape=(1,))` をデフォルトとします。

### 連続アクション空間

```yaml
action_space:
  type: continuous
  low: 0.0
  high: 1.0
  shape: [1]
```

各エージェントは `[0, 1]` の浮動小数点数を出力し、SWMM 要素の目標開度として直接適用されます。

### 離散アクション空間

```yaml
action_space:
  type: discrete
  n: 11
```

各エージェントは整数インデックス `0, 1, ..., n-1` を出力し、等間隔の連続値にマッピングされます：

```
インデックス 0 → target_setting 0.0
インデックス 1 → target_setting 0.1
...
インデックス n-1 → target_setting 1.0
```

基礎となる SWMM シミュレーションは常に連続 `[0, 1]` 開度値を受け取ります——離散は同じ連続制御空間への異なる入力インターフェースです。

### normalization 設定

```yaml
normalization:
  method: zscore   # または minmax
```

zscore 方式は事前に較正された mean / std パラメータが必要です。

### time_sync 設定

```yaml
swmm_step: 300          # SWMM ルーティングステップ（秒）、PySWMM の stepsize と一致
decision_interval: 900  # RL 決定間隔（秒）、swmm_step の整数倍である必要あり
max_steps: 96           # エピソード最大長
warmup_steps: 0         # 最初の RL 決定前のウォームアップステップ
```

### 環境設定のカスタマイズ

#### カスタム報酬関数

`CustomReward` クラスを継承したカスタム報酬関数：

```python
from swmmEnv.reward.default_reward import DefaultReward

class MyReward(DefaultReward):
    def __init__(self, config):
        super().__init__(config)
        # カスタムパラメータの追加

    def calculate(self, env):
        # カスタム報酬ロジック
        flooding = env.get_flooding()
        efficiency = env.get_energy_efficiency()
        return -flooding * 5.0 + efficiency * 0.5
```

カスタム報酬クラスを設定で指定します：

```yaml
reward:
  type: custom            # カスタム報酬関数を使用
  custom_class: my_module.MyReward  # 完全修飾クラス名
  param1: value1          # カスタムパラメータ
```

#### カスタム状態正規化

```python
from swmmEnv.sim.normalizer import ZScoreNormalizer
import numpy as np

# 履歴データから正規化パラメータを計算
obs_history = collect_historical_observations()
normalizer = ZScoreNormalizer()
normalizer.calibrate(obs_history)  # mean/std を自動計算

# 設定に書き出し
config["normalization"] = {
    "method": "zscore",
    "mean": normalizer.mean.tolist(),
    "std": normalizer.std.tolist(),
}
```

## トレーニングガイド

### 分散トレーニング（RLlib）

`SWMMMultiAgentEnv` は RLlib のリモートワーカーで直接動作します。ワーカーは設定内の `inp_file` パスを自動的に分離コピーに変換し、並列安全を確保します。

**重要な注意：** RLlib で分散トレーニングを使用する場合、.inp ファイルのパスは設定内のオリジナルパスとして指定してください。ワーカーが自動的に分離コピーを作成します。具体的な設定例については `examples/train_mappo.py` を参照してください。

### トレーニングパフォーマンスの最適化

エピソード間のリセット効率を向上させるため、エンジンは**ホットスタートファイル**を使用してシミュレーション状態を復元し、Simulation オブジェクトを再作成する必要がありません（RL トレーニングループでは大幅に高速化されます）。

**トレーニングの主要設定パラメータ：**

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `warmup_steps` | 0-10 | 最初の RL 決定前のステップ数；初期条件の安定化に有効 |
| `max_steps` | 500-2000 | エピソード長；暴雨イベントの期間に合わせて調整 |
| `hotstart_file` | 自動 | 自動生成され、高速リセットを提供 |
| 正規化パラメータ | データから較正 | 履歴シミュレーション実行に基づいて mean/std を設定 |

## プロジェクト構造

```
swmmEnv/
├── swmmEnv/
│   ├── __init__.py              # パッケージエントリ
│   ├── sim/                     # シミュレーションモジュール
│   │   ├── engine.py            # PySWMM シミュレーションラッパー
│   │   ├── time_sync.py         # RL と SWMM ステップの同期
│   │   ├── normalizer.py        # 観測/報酬の z-score 正規化
│   │   └── mapping.py           # エージェント ⇄ SWMM 要素の登録
│   ├── envs/                    # 環境モジュール
│   │   ├── swmm_env/
│   │   │   ├── env.py           # コア MDP 環境
│   │   │   ├── pettingzoo_env.py # PettingZoo ParallelEnv ラッパー
│   │   │   └── rllib_env.py     # RLlib MultiAgentEnv アダプター
│   │   └── register_env.py      # MARLlib / RLlib 登録ヘルパー
│   ├── reward/                  # 報酬モジュール
│   │   ├── default_reward.py    # 内蔵報酬関数コレクション
│   │   └── custom_reward.py     # カスタム報酬テンプレート
│   └── config/                  # 設定モジュール
│       ├── loader.py            # YAML 設定のロードとバリデーション
│       └── default_config.yaml  # デフォルト設定
├── examples/
│   ├── manual_control.py        # インタラクティブデバッグスクリプト
│   └── train_mappo.py           # MAPPO トレーニング例
├── tests/                       # 単体テスト
├── pyproject.toml
├── README.md                    # 英語ドキュメント
├── README_CN_simplified.md      # 簡体字中国語ドキュメント
├── README_CN_traditional.md     # 繁体字中国語ドキュメント
└── README_JP.md                 # 日本語ドキュメント（本ファイル）
```

## 環境要件

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

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 実行例とテスト

### サンプルスクリプト

リポジトリには `examples/` ディレクトリに実行可能なサンプルが含まれています：

```bash
# ランダム制御（デフォルト設定、1 エピソード）
python examples/manual_control.py

# ランダム制御（カスタム設定、5 エピソード）
python examples/manual_control.py --config my_config.yaml --episodes 5

# インタラクティブモード（手動でアクションを入力）
python examples/manual_control.py --interactive

# MARLlib を使用した MAPPO トレーニング
python examples/train_mappo.py

# カスタム設定で MAPPO トレーニング
python examples/train_mappo.py --config my_config.yaml --steps 50000

# RLlib を直接使用したトレーニング
python examples/train_mappo.py --backend rllib --steps 100000
```

### テストの実行

単体テストスイートを実行してインストールを確認：

```bash
# 全テストを実行
pytest tests/ -v

# 特定コンポーネントのテストを実行
pytest tests/test_engine.py -v
pytest tests/test_mapping.py -v
pytest tests/test_normalizer.py -v
pytest tests/test_time_sync.py -v
pytest tests/test_swmm_env.py -v
pytest tests/test_pettingzoo_env.py -v

# カバレッジレポート付き
pytest tests/ --cov=swmmEnv --cov-report=term-missing
```

## トラブルシューティング

| 問題 | 考えられる原因 | 解決策 |
|------|---------------|--------|
| `ValueError: decision_interval must be divisible by swmm_step` | time_sync 設定が無効 | decision_interval % swmm_step == 0 を確認 |
| `FileNotFoundError: Configuration file not found` | 設定パスが間違っている | 絶対パスまたはカレントディレクトリからの相対パスを使用 |
| PySWMM simulation not started 関連エラー | .inp ファイルがないかパスが間違っている | inp_file が存在し、パスが正しいことを確認 |
| RL トレーニングの勾配が不安定 | 正規化パラメータが不適切 | 履歴シミュレーションデータに基づいて mean/std を較正 |
| 環境リセットが遅い | ホットスタートファイル未使用 | ワーカーが自動的にホットスタートを作成；_initial_hotstart の設定を確認 |
| RuntimeError: Environment must be reset before stepping | step() の前に reset() を呼んでいない | 必ず最初に env.reset() を呼び出す |
| 並列シミュレーションがクラッシュ | 複数の PySWMM インスタンスが同じ .inp を使用 | 各ワーカーに worker_index &gt; 0 を設定して分離された .inp コピーを使用 |
| SWMMMultiAgentEnv インスタンス化時の ImportError: ray[rllib] is required | ray が未インストール | pip install "swmmEnv[marl]" または pip install ray[rllib] を実行 |
