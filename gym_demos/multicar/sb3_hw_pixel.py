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


def train(model_path="highway_ppo/model_cnn", len_train=2e4, load_model=None, name_tag=None):    
    env = get_env()
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    if load_model:
        model = PPO.load(load_model, env=env)
    else:
        model = PPO(
            "CnnPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=1,
            tensorboard_log="highway_ppo_cnn/",)
        
    model.learn(len_train)
    model.save(model_path)
    env.close()

def get_env():
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
            "duration": 100,
        },
        "policy_frequency": 2
    }
    env = gym.make('highway-v0', config=config, render_mode="rgb_array")
    return env

def record_race(model_path="highway_ppo/model_cnn", eva_episodes=5, name_tag=""):
    env = get_env()
    model = PPO.load(model_path, env=env) 
    (obs, info), done = env.reset(), False
    env.unwrapped.config["duration"]= 100
    # Run episode
    res = {"rewards": [], "steps": [], "speeds": []}
    for i in range(eva_episodes):
        res_speed = 0.0
        res_steps = 0.0
        obs, info = env.reset()
        done = truncated = False
        frames = []
        total_reward = 0
        early_stop_tag = "e"
        for step in trange(env.unwrapped.config["duration"], desc="Running..."):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
            # print(f"Info: {info}, obs: {obs.shape}")
            # print(f"Speed: {info['speed']}, ")
            total_reward += reward
            res_steps += 1
            res_speed += info["speed"]

            frame = env.render()
            frames.append(frame)
            if done or truncated:
                print("early stop")
                early_stop_tag = "f"
                break
        imageio.mimsave(f'videos/cnn_{name_tag}_{early_stop_tag}_{total_reward:.2f}.mp4', frames, fps=10)
        frames = []
        res["rewards"].append(total_reward)
        res["steps"].append(step)
        res["speeds"].append(info["speed"]/ res_steps)
    env.close()
    return res

def multi_run(model_path="highway_ppo/model_cnn", num_episodes=5, train_mode=True, name_tag=""):
    
    rewards = []
    for i in range(num_episodes):
        if train_mode:
            train(model_path=model_path, name_tag=name_tag)
        reward = record_race(model_path=model_path, name_tag=name_tag)
        rewards.append(reward)
    
    # visualization 
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute the average of each sublist
    avg_rewards = [np.mean(sublist["rewards"]) for sublist in rewards]
    avg_steps = [np.mean(sublist["steps"]) for sublist in rewards]
    avg_speeds = [np.mean(sublist["speeds"]) for sublist in rewards]

    print("Average Rewards:", avg_rewards)
    print("Average Steps:", avg_steps)
    print("Average Speeds:", avg_speeds)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, marker='o', label='Avg Reward')
    plt.plot(avg_steps, marker='s', label='Avg Steps')
    plt.plot(avg_speeds, marker='^', label='Avg Speed')

    plt.title('Training Metrics Trend')
    plt.xlabel('Episode')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'videos/training_metrics_{name_tag}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    model_path="highway_ppo/model_cnn"
    #record_race(model_path=model_path)
    name_tag = datetime.now().strftime("%H_%M_%S")
    multi_run(model_path="highway_ppo/model_cnn", num_episodes=5, train_mode=True, name_tag=name_tag)
