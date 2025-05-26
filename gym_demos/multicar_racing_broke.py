# refer to https://github.com/igilitschenski/multi_car_racing/tree/master
import gym
import gym_multi_car_racing
from human_actions import register_input
import numpy as np
import pygame
pygame.init()
env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

obs = env.reset()
done = False
total_reward = 0
print("action shape", env.action_space)
exit(0)


action = np.array([0.0, 0.0, 0.0])
while not done:
  # The actions have to be of the format (num_agents,3)
  # The action format for each car is as in the CarRacing-v0 environment.
  register_input(action)
  #action = my_policy(obs)

  # Similarly, the structure of this is the same as in CarRacing-v0 with an
  # additional dimension for the different agents, i.e.
  # obs is of shape (num_agents, 96, 96, 3)
  # reward is of shape (num_agents,)
  # done is a bool and info is not used (an empty dict).
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()
  if done:
    break

print("individual scores:", total_reward)
