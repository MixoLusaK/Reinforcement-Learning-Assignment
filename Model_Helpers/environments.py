import os
import gym
import crafter
from gym.envs.registration import register
from shimmy import GymV21CompatibilityV0
from .image_preprocessing import GrayscaleNormalizeWrapper

try:
    from .belief_reward_shaping import BeliefRewardWrapper
except ImportError:
    BeliefRewardWrapper = None

register(
    id='CrafterPartial-v1',
    entry_point='crafter:Env',
)


def make_env(log_dir='./Training/Logs/jsons/',
             save_video=False,
             save_episode=False):
    """
    Create standard Crafter environment without reward shaping.
    Args:
        log_dir: Directory to save logs
        save_video: Whether to save videos
        save_episode: Whether to save episode data
    Returns:
        Wrapped Crafter environment compatible with Gymnasium/SB3
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create base environment
    env = gym.make("CrafterPartial-v1")

    # Apply Recorder wrapper
    env = crafter.Recorder(
        env,
        log_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=save_episode
    )

    env = GymV21CompatibilityV0(env=env)

    return env


def make_shaped_env(log_dir='./Training/Logs/jsons_shaped/',
                    lambda_param=1000,
                    health_weight=0.1,
                    clip_belief_reward=True,
                    use_clusters=True,
                    save_video=False,
                    save_episode=False):
    """
    Create Crafter environment WITH reward shaping (for training improved model).
    Args:
        log_dir: Directory to save logs
        lambda_param: Lambda parameter for belief reward scaling
        health_weight: Weight for health-based reward shaping
        clip_belief_reward: Whether to clip belief rewards
        use_clusters: Whether to use achievement clusters
        save_video: Whether to save videos
        save_episode: Whether to save episode data
    Returns:
        Wrapped Crafter environment with reward shaping, compatible with Gymnasium/SB3
    """
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("CrafterPartial-v1")

    env = crafter.Recorder(
        env,
        log_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=save_episode
    )

    env = GymV21CompatibilityV0(env=env)
    if BeliefRewardWrapper is not None:
        env = BeliefRewardWrapper(
            env,
            lambda_param=lambda_param,
            health_weight=health_weight,
            clip_belief_reward=clip_belief_reward,
            use_clusters=use_clusters
        )
    else:
        print("WARNING: BeliefRewardWrapper not available, using standard environment")

    return env


def make_preprocessed_shaped_env(log_dir='./Training/Logs/jsons_processed/', save_video=False,
                    save_episode=False):
    """
    Used for: DQN + Reward Shaping + Preprocessing (Iteration 3)
    """
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make("CrafterPartial-v1")

    env = crafter.Recorder(
        env,
        log_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=save_episode
    )
    env = GymV21CompatibilityV0(env=env)
    env = BeliefRewardWrapper(env)

    # Improvement 2: Preprocessing
    env = GrayscaleNormalizeWrapper(env)

    return env