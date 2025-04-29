# env_wrapper.py
import gym
import numpy as np
import cv2
from gym import spaces
from collections import deque

class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=4):
        super(CarRacingWrapper, self).__init__(env)
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_stack, 96, 96), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess(obs)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info

    def preprocess(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (96, 96))
        return obs
