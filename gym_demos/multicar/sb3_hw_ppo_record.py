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

def train(model_path="highway_ppo/model"):
    n_cpu = 1
    batch_size = 64
    #env = make_vec_env("highway-fast-v0", n_envs=n_cpu)
    env = make_vec_env(
        "highway-fast-v0",
        n_envs=n_cpu,
        env_kwargs={
            "config": {
                "duration": 100  # customize the episode length here
            }
        }
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=1,
        tensorboard_log="highway_ppo/",
    )
    # Train the agent
    model.learn(total_timesteps=int(500)) # 2e4
    # Save the agent
    model.save(model_path)


def record_race(model_path="highway_ppo/model"):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    model = PPO.load(model_path, env=env)    

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
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)        
        
    env.close()
    show_videos()


if __name__ == "__main__":
    model_path="highway_ppo/model"

    if True:
        train(model_path=model_path)   
    record_race(model_path=model_path)
