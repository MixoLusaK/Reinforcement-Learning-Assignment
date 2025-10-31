import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class FrameStackWrapper(gym.ObservationWrapper):
    """Stacks the last N frames together to give the agent temporal context."""

    def __init__(self, env, num_stack=4, stack_axis=-1, lazy=False):
        """
        Args:
            env: The environment to wrap
            num_stack: Number of frames to stack (default: 4)
            stack_axis: Axis along which to stack frames
                       -1 or 2: Stack along channels (H, W, C*N) - Default
                       0: Stack along first dimension (N, H, W, C)
            lazy: If True, use lazy frame stacking for memory efficiency
                  (useful with large replay buffers)
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.stack_axis = stack_axis
        self.lazy = lazy

        # Initialize frame buffer (deque for efficient append/pop)
        self.frames = deque(maxlen=num_stack)

        # Update observation space based on stacking
        old_space = self.observation_space
        old_shape = old_space.shape

        if stack_axis == -1 or stack_axis == len(old_shape) - 1:
            # Stack along channel dimension: (H, W, C) -> (H, W, C*N)
            new_shape = old_shape[:-1] + (old_shape[-1] * num_stack,)
        elif stack_axis == 0:
            # Stack along first dimension: (H, W, C) -> (N, H, W, C)
            new_shape = (num_stack,) + old_shape
        else:
            raise ValueError(f"stack_axis must be 0 or -1, got {stack_axis}")

        self.observation_space = spaces.Box(
            low=np.repeat(old_space.low, num_stack, axis=stack_axis),
            high=np.repeat(old_space.high, num_stack, axis=stack_axis),
            shape=new_shape,
            dtype=old_space.dtype
        )

    def observation(self, obs):
        """
        Stack frames together.
        """
        # Add current frame to buffer
        self.frames.append(obs)

        if self.lazy:
            # Return LazyFrames object (doesn't create array until needed)
            return LazyFrames(list(self.frames), self.stack_axis)
        else:
            # Immediately create stacked array
            if self.stack_axis == -1 or self.stack_axis == 2:
                # Stack along channels: (H, W, C*N)
                return np.concatenate(list(self.frames), axis=-1)
            else:
                # Stack along first dimension: (N, H, W, C)
                return np.stack(list(self.frames), axis=0)

    def reset(self, **kwargs):
        """
        Reset environment and fill frame buffer with initial observation.
        """
        obs, info = self.env.reset(**kwargs)

        # Fill buffer with copies of initial frame
        # This prevents the agent from seeing "blank" frames at episode start
        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self.observation(obs), info


class LazyFrames:
    """
    Lazy frame stacking - only creates numpy array when needed.
    Stores references to frames instead of copying them.
    """

    def __init__(self, frames, stack_axis=-1):
        """
        Args:
            frames: List of frames to stack
            stack_axis: Axis along which to stack when converting to array
        """
        self._frames = frames
        self.stack_axis = stack_axis
        self._out = None  # Cache for the stacked array

    def _force(self):
        """Create the actual stacked array (only called when needed)."""
        if self._out is None:
            if self.stack_axis == -1 or self.stack_axis == 2:
                self._out = np.concatenate(self._frames, axis=-1)
            else:
                self._out = np.stack(self._frames, axis=0)
        return self._out

    def __array__(self, dtype=None):
        """Allow numpy operations on LazyFrames."""
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._force()[i]

    @property
    def shape(self):
        return self._force().shape

    @property
    def dtype(self):
        return self._frames[0].dtype

