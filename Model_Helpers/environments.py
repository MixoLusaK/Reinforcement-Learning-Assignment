import os
import gymnasium as gym
import crafter
from gymnasium.envs.registration import register
from shimmy import GymV21CompatibilityV0

# Optional reward shaping wrapper
try:
    from .belief_reward_shaping import BeliefRewardWrapper
except ImportError:
    BeliefRewardWrapper = None

# Register Crafter environment
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

    # Create base environment directly from the local crafter package
    # Avoid using `gym.make` (gymnasium) because the local `crafter.Env`
    # may not subclass `gymnasium.Env` and would trigger a type check error.
    env = crafter.Env()

    # Apply Recorder wrapper
    env = crafter.Recorder(
        env,
        log_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=save_episode
    )

    # Wrap for Gymnasium / SB3 compatibility
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

    # Create base environment directly from the local crafter package
    env = crafter.Env()

    # Apply Recorder wrapper
    env = crafter.Recorder(
        env,
        log_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=save_episode
    )

    # IMPORTANT: Apply GymV21CompatibilityV0 BEFORE BeliefRewardWrapper
    env = GymV21CompatibilityV0(env=env)

    # Now apply reward shaping wrapper
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