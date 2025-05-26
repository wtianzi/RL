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

def train(model_path="highway_ppo/model", len_train=2e4, load_model=None):
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
    if load_model:
        model = PPO.load(load_model, env=env)
    else:
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
    model.learn(total_timesteps=int(len_train)) # 2e4
    # Save the agent
    model.save(model_path)

def record_race(model_path="highway_ppo/model"):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")    
    model = PPO.load(model_path, env=env)    

    (obs, info), done = env.reset(), False

    # Make agent
    agent_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
        "env_preprocessors": [{"method":"simplify"}],
        "budget": 50,
        "gamma": 0.7,
    }
    agent = agent_factory(env, agent_config)
    env.unwrapped.config["duration"]= 500
    # Run episode
    res_rewards = []
    for i in range(5):
        obs, info = env.reset()
        done = truncated = False
        frames = []
        total_reward = 0
        early_stop_tag = "e"
        for step in trange(env.unwrapped.config["duration"], desc="Running..."):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)  
            total_reward += reward      
            frames.append(env.render())
            if done or truncated:
                print("early stop")
                early_stop_tag = "f"
                break
        filename = datetime.now().strftime("%H_%M_%S")
        imageio.mimsave(f'videos/{filename}_{early_stop_tag}_{total_reward}.mp4', frames, fps=10)
        frames = []
        res_rewards.append(total_reward) 
    env.close()
    return res_rewards


if __name__ == "__main__":
    model_path="highway_ppo/model"
    # analysze the model
    # draw the res_rewards

    rewards = []
    if True:
        for i in range(2):
            t_model_path = f"{model_path}"
            train(model_path=t_model_path, len_train=2e4, load_model=t_model_path)   
            rewards.append(record_race(model_path=t_model_path))
    
    # visualization 
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute the average of each sublist
    averages = [np.mean(sublist) for sublist in rewards]

    # Plotting
    plt.plot(averages, marker='o')
    plt.title('Average Reward Trend')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()
