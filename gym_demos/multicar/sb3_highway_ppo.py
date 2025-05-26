import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import json
import highway_env  # noqa: F401


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 1
        batch_size = 64
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu)
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
        model.save("highway_ppo/model")
    
    env = gym.make("highway-fast-v0")
    model = PPO.load("highway_ppo/model", env=env)        

    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()