#image_preprocessing.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GrayscaleNormalizeWrapper(gym.ObservationWrapper):
    """
    Combines grayscale conversion and normalization into one wrapper.

    Transforms:
    - Input: (64, 64, 3) uint8 in [0, 255]
    - Output: (64, 64, 1) float32 in [0.0, 1.0]

    This is the standard preprocessing used in many DQN implementations.
    """

    def __init__(self, env):
        super().__init__(env)

        # Get original shape (should be 64x64x3 for Crafter)
        obs_shape = self.observation_space.shape

        # New shape: (height, width, 1) with float32
        new_shape = (obs_shape[0], obs_shape[1], 1)

        # Update observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )

        print(f"[PREPROCESSING] Grayscale + Normalize wrapper applied")
        print(f"  Original shape: {obs_shape} (uint8)")
        print(f"  New shape: {new_shape} (float32)")
        print(f"  Memory reduction: ~3x")

    def observation(self, obs):
        """
        Convert RGB observation to grayscale and normalize.

        Uses standard luminosity method:
        Gray = 0.299*R + 0.587*G + 0.114*B
        """
        # Convert to grayscale using luminosity method
        grayscale = np.dot(obs[..., :3], [0.299, 0.587, 0.114])

        # Normalize to [0, 1] range
        normalized = grayscale.astype(np.float32) / 255.0

        # Add channel dimension back: (64, 64) -> (64, 64, 1)
        return np.expand_dims(normalized, axis=-1)
