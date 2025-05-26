import gymnasium as gym
import highway_env
from utils import record_videos, show_videos
import pygame
import numpy as np

from rl_agents.agents.common.factory import agent_factory
from tqdm import trange

from human_actions import register_input_v2
pygame.init()

def human_race():
    action = np.array([1])
    env = gym.make('highway-v0', render_mode='human')

    obs, info = env.reset()
    done = truncated = False

    while not (done or truncated):
        register_input_v2(action)
        obs, reward, done, truncated, info = env.step(action[0])


    env.close()
    pygame.quit()

def record_race():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env = record_videos(env)
    (obs, info), done = env.reset(), False

    # Make agent
    agent_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
        "env_preprocessors": [{"method":"simplify"}],
        "budget": 50,
        "gamma": 0.7,
    }
    agent = agent_factory(env, agent_config)

    # Run episode
    for step in trange(env.unwrapped.config["duration"], desc="Running..."):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()
    show_videos()

if __name__ == "__main__":
    record_race()
    pass