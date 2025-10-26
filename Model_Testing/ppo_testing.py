import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Ensure we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from Model_Helpers.environments import make_env, make_shaped_env


def test_ppo(model_path, model_type="baseline", n_episodes=5, record_video=True, frame_stack=4, device="auto"):
    """
    Evaluate a trained PPO agent in the Crafter environment.

    Args:
        model_path (str): Path to trained PPO model (.zip file)
        model_type (str): 'baseline' or 'shaped' environment
        n_episodes (int): Number of test episodes
        record_video (bool): Whether to record a video of the agent
        frame_stack (int): Number of frames to stack
        device (str): Device ('auto', 'cuda', or 'cpu')
    """
    # Setup test log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = Path(f"./Testing/PPO/{model_type}/{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Test directory: {test_dir}")

    # Create test environment
    env_fn = make_shaped_env if model_type == "shaped" else make_env
    env = env_fn(log_dir=str(test_dir), save_video=record_video, save_episode=True)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    env = VecTransposeImage(env)

    # Load trained model
    print(f"✓ Loading PPO model from: {model_path}")
    model = PPO.load(model_path, env=env, device=device)

    # Run evaluation
    total_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f} | Steps = {steps}")

    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n✓ Mean reward over {n_episodes} episodes: {mean_reward:.2f}")

    env.close()
    print("✓ Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent on Crafter")

    parser.add_argument("--model_path", type=str, required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "shaped"],
                        help="Environment type: baseline or shaped")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--record_video", action="store_true", help="Record video of test episodes")
    parser.add_argument("--frame_stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for inference")

    args = parser.parse_args()

    test_ppo(
        model_path=args.model_path,
        model_type=args.model_type,
        n_episodes=args.n_episodes,
        record_video=args.record_video,
        frame_stack=args.frame_stack,
        device=args.device
    )
