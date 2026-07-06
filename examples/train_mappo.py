"""
MARLlib training example for SWMMEnv.

This script demonstrates how to train a multi-agent policy
using MARLlib with the SWMMEnv.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_mappo(config_path: str = None, num_steps: int = 100000):
    """
    Train MAPPO on SWMMEnv using MARLlib.

    Args:
        config_path: Path to SWMMEnv configuration file
        num_steps: Number of training timesteps
    """
    try:
        from marllib import marl
        import ray
    except ImportError:
        print("MARLlib is required for this example.")
        print("Install with: pip install marllib ray[rllib]")
        return

    # Import environment registration
    from swmmEnv.envs.register_env import make_env, register_with_marllib

    # Register environment with MARLlib
    register_with_marllib()

    # Create environment
    if config_path:
        env = make_env(
            environment_name="swmm",
            map_name="custom",
            config_path=config_path
        )
    else:
        env = make_env(environment_name="swmm", map_name="control")

    # Get environment info
    env_info = env.get_env_info()
    print("\nEnvironment info:")
    print(f"  Number of agents: {env_info['num_agents']}")
    print(f"  Episode limit: {env_info['episode_limit']}")

    # Build model configuration
    model = marl.build_model(
        environment=env,
        algorithm=marl.algos.mappo,
        model_preference={
            "core_arch": "mlp",  # MLP for non-sequential control
            "encode_layer": "128-128",
            "hidden_dim": 64,
        }
    )

    # Initialize MAPPO algorithm
    mappo = marl.algos.mappo(
        hyperparam_source="common"
    )

    # Training configuration
    train_config = {
        "lr": 0.0005,
        "gamma": 0.99,
        "batch_episode": 10,
        "num_sgd_iter": 5,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "clip_param": 0.3,
    }

    print("\nStarting training...")
    print(f"  Algorithm: MAPPO")
    print(f"  Target timesteps: {num_steps}")

    # Start training
    mappo.fit(
        env=env,
        model=model,
        stop={"timesteps_total": num_steps},
        **train_config
    )

    print("\nTraining completed!")


def train_with_rllib(config_path: str = None, num_steps: int = 100000):
    """
    Train using Ray RLlib directly (alternative to MARLlib).

    This shows direct RLlib integration without MARLlib's abstraction.
    """
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env import ParallelPettingZooEnv
        import ray
    except ImportError:
        print("Ray RLlib is required for this example.")
        print("Install with: pip install ray[rllib]")
        return

    from swmmEnv import SWMMParallelEnv, load_config

    # Load config
    config = load_config(config_path)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create environment creator
    def env_creator(env_config):
        env_config_inner = config.copy()
        env_config_inner.update(env_config)
        return SWMMParallelEnv(env_config_inner)

    # Register environment with Ray
    from ray.tune.registry import register_env
    register_env("swmm_env", env_creator)

    # Configure PPO for multi-agent
    config = (
        PPOConfig()
        .environment("swmm_env")
        .multi_agent(
            policies={
                "shared_policy": None,  # Will be auto-generated
            },
            policy_mapping_fn=lambda agent_id: "shared_policy",
        )
        .training(
            lr=0.0005,
            gamma=0.99,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .resources(
            num_gpus=0,  # Set to 1 if GPU available
        )
    )

    # Build algorithm
    algo = config.build()

    print("\nStarting training with Ray RLlib...")

    # Training loop
    for i in range(num_steps // 4000):
        result = algo.train()
        print(f"Iteration {i}: "
              f"timesteps={result['timesteps_total']}, "
              f"episode_reward_mean={result['episode_reward_mean']:.2f}")

    print("\nTraining completed!")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MARL on SWMMEnv")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to SWMMEnv configuration file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help='Number of training timesteps'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['marllib', 'rllib'],
        default='marllib',
        help='Training backend to use'
    )

    args = parser.parse_args()

    if args.backend == 'marllib':
        train_mappo(args.config, args.steps)
    else:
        train_with_rllib(args.config, args.steps)