import copy
from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from numpy import dtype
from numpy.ma.core import shape

PADDING = 32
RACKET_SIZE = 96

g = 9.81  # gravity


def integrate_rk4(state, step, t, dt, dydx_func):
    """
    Fourth-order Runge-Kutta method.
    Source: https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/
    :param step:
    :param state:
    :param t:
    :param dt:
    :param dydx_func:
    :return:
    """
    k1 = dydx_func(state, step, t, dt)
    k2 = dydx_func([v + d * dt / 2 for v, d in zip(state, k1)], step, t, dt)
    k3 = dydx_func([v + d * dt / 2 for v, d in zip(state, k2)], step, t, dt)
    k4 = dydx_func([v + d * dt for v, d in zip(state, k3)], step, t, dt)
    return [v + (k1_ + 2 * k2_ + 2 * k3_ + k4_) * dt / 6 for v, k1_, k2_, k3_, k4_ in zip(state, k1, k2, k3, k4)]


def dydx_ball(state: np.ndarray[float], _step: int, _t: float, _dt: float) -> np.ndarray[float]:
    """
    :param state:
    :param _step:
    :param _t:
    :param _dt:
    :return: derivatives of all state variables of a free-falling ball
    """
    [_x, _y, vx, vy] = state
    return np.array([vx, vy, 0, -g], dtype=float)


def draw_radial_gradient_circle(surface, center, radius, inner_color, outer_color):
    """
    Draws a radial gradient circle on the given surface.

    :param surface: Pygame surface to draw on
    :param center: Tuple (x, y) for the circle center
    :param radius: Radius of the circle
    :param inner_color: Inner color as an (R, G, B) tuple
    :param outer_color: Outer color as an (R, G, B) tuple
    """
    for r in range(radius, 0, -1):
        # Interpolate color
        t = r / radius  # Ratio (0 to 1)
        color = (
            int(inner_color[0] * t + outer_color[0] * (1 - t)),
            int(inner_color[1] * t + outer_color[1] * (1 - t)),
            int(inner_color[2] * t + outer_color[2] * (1 - t))
        )
        # Draw circle with the interpolated color
        pygame.draw.circle(surface, color, center, r)


class Actions(Enum):
    up = 0
    down = 1
    nop = 2


class BallCatcherEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 24}

    def __init__(self, render_mode=None, size=512):
        self.size = size
        self.window_size = size + PADDING * 2

        self.dt = 1.0 / self.metadata['render_fps']

        self._step: int = 0  # current step
        self._time: float = 0  # elapsed time
        self._last_state: np.ndarray[float] = None  # the state from the last timestep
        self._state: np.ndarray[float] = None  # ball [x, y, vx, vy]

        self._agent_location: float = 0  # racket location

        self.observation_space = gym.spaces.Dict(
            {
                "ball_location": spaces.Box(0, size, shape(2, ), dtype=float),
                "last_ball_location": spaces.Box(0, size, shape(2, ), dtype=float),
                "agent": spaces.Box(0, size, shape(1, ), dtype=float),
            }
        )

        # We have 3 actions: "up", "down", "nop"
        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {
            Actions.up.value: 1,
            Actions.down.value: -1,
            Actions.nop.value: 0,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self):
        [x, y, _, _] = self._state
        [last_x, last_y, _, _] = self._last_state
        return {
            "ball_location": np.array([x, y], dtype=float),
            "last_ball_location": np.array([last_x, last_y], dtype=float),
            "agent": self._agent_location
        }

    # TODO: understand the meaning of this method
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # TODO: calc location & velocity to hit the wall and stay within boundaries
        x = 0
        y = self.size / 2
        vx = 65
        vy = 35
        self._state = np.array([x, y, vx, vy], dtype=float)
        self._last_state = copy.deepcopy(self._state)

        self._agent_location = self.size / 2
        return self._get_observation(), self._get_info()

    def step(self, action):
        self._last_state = copy.deepcopy(self._state)
        self._state = integrate_rk4(self._state, self._step, self._time, self.dt, dydx_ball)
        self._step += 1
        self._time += self.dt

        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size)

        [x, y, _, _] = self._state
        terminated = x >= self.size

        delta = 0
        if terminated:
            delta = abs(y - self._agent_location)
            if delta >= RACKET_SIZE / 2:
                reward = -100
            else:
                reward = 100 * (1 - 2 * delta / RACKET_SIZE)
        else:
            reward = 0 if action == Actions.nop.value else -0.01

        print(f"Terminated: {terminated}; Reward: {reward:.2f}; Delta: {delta}")

        if self.render_mode == "human":
            self._render_interactive()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_to_surface(self) -> pygame.Surface:
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        [ox, oy] = [PADDING, PADDING]
        [x, y, _, _] = self._state
        draw_radial_gradient_circle(
            canvas,
            (x + ox, y + oy),
            16,
            (185, 67, 102),
            (204, 146, 155)
        )

        pygame.draw.rect(
            canvas,
            (144, 238, 144),
            pygame.Rect(
                (self.size + ox, self._agent_location - RACKET_SIZE / 2 + oy),
                (16, RACKET_SIZE))
        )
        pygame.draw.rect(
            canvas,
            (204, 146, 155),
            pygame.Rect(
                (self.size + ox, self._agent_location - 2 + oy),
                (16, 4))
        )

        return pygame.transform.flip(canvas, False, True)

    def _render_frame(self):
        canvas = self._render_to_surface()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def _render_interactive(self) -> None:
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
