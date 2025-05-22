import gymnasium as gym
import pygame
import numpy as np
pygame.init()
if False:
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

env = gym.make("CarRacing-v3", render_mode="human")
mode = 0 
if mode == 0:
    # reset with colour scheme change
    obs, _ = env.reset()
elif mode == 1:
    obs, _ = env.reset(options={"randomize": True})
elif mode == 2:
    # reset with no colour scheme change
    obs, _ = env.reset(options={"randomize": False})
print("observation shape", obs.shape)
print("action space", env.action_space)
#exit(0)

done = False 
total_reward = 0.0
steps = 0
restart = False

a = np.array([0.0, 0.0, 0.0])

def register_input():
    global quit, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0

        if event.type == pygame.QUIT:
            quit = True

quit = False
while not quit:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        register_input()
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or restart or quit:
            break
env.close()
pygame.quit()