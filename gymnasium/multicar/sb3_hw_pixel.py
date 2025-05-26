import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import json
import highway_env  # noqa: F401
from utils import record_videos, show_videos
from rl_agents.agents.common.factory import agent_factory
from tqdm import trange
import imageio
from datetime import datetime
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, DQN, SAC, TD3, PPO
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def train(model_path="highway_ppo/model", len_train=2e4, load_model=None):    
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2
    }
    env = gym.make('highway-v0', config=config)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)

    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        tensorboard_log="highway_ppo_cnn/",)
    model.learn(1000)

if __name__ == "__main__":
    train()
