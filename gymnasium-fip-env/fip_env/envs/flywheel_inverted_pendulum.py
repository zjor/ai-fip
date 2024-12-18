import copy
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from numpy import dtype
from numpy.ma.core import shape
from numpy import pi, sin, cos

RENDER_FPS = 24


class Actions(Enum):
    nop = 0
    cw = 1
    ccw = 2


class FlywheelInvertedPendulumEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 24}

    def __init__(self, render_mode=None):
        self.theta_threshold = pi / 24  # rod angle limit exceeding which the episode terminates
        self.max_torque = 2.0  # maximal torque applied to the wheel
        self.max_wheel_w = 16.0  # maximal angular velocity of the wheel
        self.max_rod_w = 4.0  # maximal angular velocity of the pendulum

        # physical model parameters
        self.g: float = 9.81  # gravity
        self.m1: float = 0.9  # mass of the rod
        self.l: float = 1.0  # length of the rod
        self.m2: float = 3.0  # mass of the wheel
        self.r: float = 0.6  # radius of the flywheel
        self.b: float = 0.5  # friction coefficient between the rod and the wheel
        self.dt: float = 1.0 / RENDER_FPS  # timestep

        # state variables
        self.theta: float = 0.0  # angle of the rod
        self.theta_dot: float = 0.0  # angular velocity of the rod
        self.phi: float = 0.0  # angle of the wheel (we don't care about it)
        self.phi_dot: float = 0.0  # angular velocity of the wheel

        self.window_size = 512

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # observation limits
        high = np.array([
            self.theta_threshold,  # angle of the rod
            self.max_rod_w,  # angular velocity of the rod
            np.inf,  # angle of the wheel (we don't care)
            self.max_wheel_w,  # angular velocity of the wheel
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        return (
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward, terminated, False, {}
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_to_window()

    def _render_to_surface(self) -> pygame.Surface:
        scale = 25.0
        [offset_x, offset_y] = [self.window_size / 2, self.window_size / 2]

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        l = self.l * scale
        pygame.draw.line(
            canvas,
            (255, 255, 255),
            (offset_x, offset_y),
            (l * cos(self.theta) + offset_x, l * sin(self.theta) + offset_y), 2)

        return pygame.transform.flip(canvas, False, True)

    def _render_frame(self):
        canvas = self._render_to_surface()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def _render_to_window(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = self._render_to_surface()
        self.window.blit(canvas, canvas.get_rect(topleft=(0, 0)))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])
