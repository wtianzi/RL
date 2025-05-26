import gymnasium as gym
import pygame
import numpy as np
import random
from human_actions import register_input
pygame.init()

# 假设多车用4个单车环境模拟，方便演示
NUM_ROBOTS = 3
NUM_CARS = NUM_ROBOTS + 1

# 创建多个CarRacing环境实例

env_human = gym.make("CarRacing-v3", render_mode="human")
envs = [gym.make("CarRacing-v3", render_mode="rgb_array") for _ in range(NUM_ROBOTS)]
envs.append(env_human)

# 初始化所有环境
observations = []
for i, env in enumerate(envs):
    obs, _ = env.reset()
    observations.append(obs)

# 玩家车动作初始化
player_action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

def get_robot_action():
    # 简单随机动作示范
    steering = random.uniform(-1, 1)
    gas = random.choice([0.0, 0.5, 1.0])
    brake = 0.0
    return np.array([steering, gas, brake])

quit = False
clock = pygame.time.Clock()

while True:
    # 处理pygame事件（只针对玩家车）
    register_input(player_action)

    # 机器人动作生成
    robot_actions = [get_robot_action() for _ in range(NUM_ROBOTS)]

    # 所有动作列表，机器人在前，玩家车在最后
    all_actions = robot_actions + [player_action]

    # 分别step每个环境，采集结果
    for i in range(NUM_CARS):
        obs, reward, terminated, truncated, info = envs[i].step(all_actions[i])
        observations[i] = obs
        if terminated or truncated:
            envs[i].reset()

    # 渲染玩家车环境窗口（pygame会自动管理）
    envs[-1].render()

    clock.tick(30)  # 控制帧率

# 关闭所有环境和pygame
for env in envs:
    env.close()
pygame.quit()
