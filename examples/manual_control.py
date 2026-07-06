"""
Manual control example for SWMMEnv.

This script demonstrates how to manually control the SWMM environment
without MARLlib, useful for testing and debugging.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv import SWMMParallelEnv, load_config


def run_manual_control(config_path: str = None, num_episodes: int = 1):
    """
    Run manual control demonstration.

    Args:
        config_path: Path to configuration file
        num_episodes: Number of episodes to run
    """
    # Load configuration
    if config_path is None:
        config = load_config(None)
        print("Using default configuration")
    else:
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")

    # Create environment
    env = SWMMParallelEnv(config)

    print(f"\nEnvironment created:")
    print(f"  Agents: {env.possible_agents}")
    print(f"  Max steps: {config.get('max_steps', 1000)}")

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*60)

        # Reset environment
        observations, info = env.reset()

        print(f"\nInitial observations:")
        for agent, obs in observations.items():
            print(f"  {agent}: {obs}")

        # Run episode with random actions
        step = 0
        total_reward = 0.0
        done = False

        while not done:
            # Generate random actions
            actions = {}
            for agent in env.agents:
                # Random action between 0 and 1
                import numpy as np
                actions[agent] = np.array([np.random.random()], dtype=np.float32)

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Track reward (all agents get same reward)
            step_reward = list(rewards.values())[0]
            total_reward += step_reward
            step += 1

            # Check done
            done = any(terminations.values())

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}: reward={step_reward:.3f}, "
                      f"cumulative={total_reward:.3f}")

            # Render every 50 steps
            if step % 50 == 0:
                env.render()

        print(f"\nEpisode finished after {step} steps")
        print(f"Total reward: {total_reward:.3f}")

        # Get final state
        state = env.core_env.get_state()
        print(f"\nFinal state:")
        print(f"  Total flooding: {state.get('rainfall', 0):.1f} mm/h")
        print(f"  Rainfall: {env.core_env.engine.get_total_flooding():.3f} m³/s")

    # Close environment
    env.close()
    print("\nEnvironment closed.")


def interactive_control(config_path: str = None):
    """
    Interactive control mode.

    Allows user to input actions manually.
    """
    config = load_config(config_path)
    env = SWMMParallelEnv(config)

    print("\n" + "="*60)
    print("Interactive SWMM Control")
    print("="*60)
    print(f"Agents: {env.possible_agents}")
    print("\nCommands:")
    print("  Enter action value (0.0-1.0) for each agent")
    print("  'q' - Quit")
    print("  'r' - Reset episode")
    print("  's' - Show state")
    print("="*60)

    observations, _ = env.reset()
    step = 0
    total_reward = 0.0

    while True:
        print(f"\n--- Step {step} ---")

        # Get actions from user
        actions = {}
        for agent in env.agents:
            while True:
                try:
                    value = input(f"  {agent} [0.0-1.0]: ").strip()

                    if value.lower() == 'q':
                        env.close()
                        print("Goodbye!")
                        return

                    if value.lower() == 'r':
                        observations, _ = env.reset()
                        step = 0
                        total_reward = 0.0
                        print("Episode reset!")
                        break

                    if value.lower() == 's':
                        env.render()
                        continue

                    action = float(value)
                    action = max(0.0, min(1.0, action))
                    import numpy as np
                    actions[agent] = np.array([action], dtype=np.float32)
                    break

                except ValueError:
                    print("  Invalid input. Enter a number between 0.0 and 1.0")

        if not actions:
            continue

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        step_reward = list(rewards.values())[0]
        total_reward += step_reward
        step += 1

        print(f"  Reward: {step_reward:.3f} (cumulative: {total_reward:.3f})")

        # Check termination
        if any(terminations.values()):
            print("\nEpisode ended!")
            print(f"Total steps: {step}")
            print(f"Total reward: {total_reward:.3f}")

            response = input("\nStart new episode? (y/n): ").strip().lower()
            if response == 'y':
                observations, _ = env.reset()
                step = 0
                total_reward = 0.0
            else:
                break

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SWMMEnv manual control")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_control(args.config)
    else:
        run_manual_control(args.config, args.episodes)