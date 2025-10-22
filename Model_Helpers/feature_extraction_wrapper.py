import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CrafterCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Crafter environment.
    Reduces 4-frame stacked 64x64 RGB images to compact features.

    Input: (batch, 12, 64, 64) - 4 frames × 3 channels
    Output: (batch, features_dim) - compressed feature vector
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # features_dim is the output size of the feature extractor
        super().__init__(observation_space, features_dim)

        # Calculate input channels based on observation space
        # If using frame stacking, this will be n_frames * n_channels
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # First conv block: 64x64 -> 15x15
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Second conv block: 15x15 -> 6x6
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Third conv block: 6x6 -> 4x4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Linear layer to project to desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        Args:
            observations: Tensor of shape (batch, channels, height, width)
        Returns:
            features: Tensor of shape (batch, features_dim)
        """
        return self.linear(self.cnn(observations))