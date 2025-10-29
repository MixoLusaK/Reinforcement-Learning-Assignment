#dependencies

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class SingleChannelCNN(BaseFeaturesExtractor):
    """
    Custom CNN architecture for single-channel (grayscale) images.
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Extract input dimensions
        n_input_channels = observation_space.shape[2]  # Should be 1 for grayscale

        print(f"[CNN] Building custom CNN for single-channel images")
        print(f"[CNN] Input channels: {n_input_channels}")
        print(f"[CNN] Input shape: {observation_space.shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Transpose to (batch, channels, height, width) format
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            # Reshape from (1, H, W, C) to (1, C, H, W)
            sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        print(f"[CNN] Flattened features: {n_flatten}")

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, height, width, channels)
        # Need: (batch, channels, height, width)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
