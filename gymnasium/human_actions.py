
import pygame
import numpy as np

#a = np.array([0.0, 0.0, 0.0])

def register_input(a):
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

def register_input_v2(a):
    global quit, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a = 0
            if event.key == pygame.K_RIGHT:
                a = 2
            if event.key == pygame.K_UP:
                a = 3
            if event.key == pygame.K_DOWN:
                a = 4
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a = 1
            if event.key == pygame.K_RIGHT:
                a = 1
            if event.key == pygame.K_UP:
                a = 1
            if event.key == pygame.K_DOWN:
                a = 1

        if event.type == pygame.QUIT:
            quit = True