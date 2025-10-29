import os
import gym
import crafter
from gym.envs.registration import register
from shimmy import GymV21CompatibilityV0
from .image_preprocessing import GrayscaleNormalizeWrapper
from .frame_stacking import FrameStackWrapper

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
    """ Create standard Crafter environment without reward shaping."""
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
    return env


def make_shaped_env(log_dir='./Training/Logs/jsons_shaped/',
                    lambda_param=1000,
                    health_weight=0.1,
                    clip_belief_reward=True,
                    use_clusters=True,
                    save_video=False,
                    save_episode=False):
    """ Create Crafter environment WITH reward shaping (Improvement 1)."""
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
    """Used for: DQN + Reward Shaping + Preprocessing (Improvement 2)"""
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


def make_framestack_env(log_dir='./Training/Logs/jsons_framestack/',
                        num_stack=4,
                        lazy=True,
                        save_video=False,
                        save_episode=False):
    """
    Improvement 1 + 2 + 3: Reward Shaping + Preprocessing + Frame Stacking.
    Temporal: Yes (4 frames of history)
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

    # Improvement 1: Reward Shaping
    if BeliefRewardWrapper is not None:
        env = BeliefRewardWrapper(env)

    # Improvement 2: Image Preprocessing
    env = GrayscaleNormalizeWrapper(env)

    # Improvement 3: Frame Stacking
    env = FrameStackWrapper(env, num_stack=num_stack, stack_axis=-1, lazy=lazy)

    return env