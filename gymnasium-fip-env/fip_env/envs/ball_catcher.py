from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from numpy import dtype
from numpy.ma.core import shape

PADDING = 32
RACKET_SIZE = 96


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
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, size=512):
        self.size = size
        self.window_size = size + PADDING * 2

        self.dt = 1.0 / self.metadata['render_fps']
        self.g = 9.81

        self._ball_location = np.array([0, self.size / 2], dtype=np.float32)
        self._ball_velocity = np.array([0, self.size / 2], dtype=np.float32)
        self._agent_location = np.array([self.size, self.size / 2], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "ball_location": spaces.Box(0, size, shape(2, ), dtype=np.float32),
                "ball_prev_location": spaces.Box(0, size, shape(2, ), dtype=np.float32),
                "agent": spaces.Box(0, size, shape(1, ), dtype=np.float32),
            }
        )

        # We have 3 actions: "up", "down", "nop"
        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {
            Actions.up.value: np.array([0, 1]),
            Actions.down.value: np.array([0, -1]),
            Actions.nop.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self):
        return {
            "ball_location": self._ball_location,
            "ball_prev_location": self._ball_location,  # TODO: calc ball location using velocity and a timestep
            "agent": self._agent_location
        }

    # TODO: understand the meaning of this method
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # TODO: calc location & velocity to hit the wall and stay within boundaries
        self._ball_location = np.array([0, self.size / 2], dtype=np.float32)
        self._ball_velocity = np.array([50, 30], dtype=np.float32)
        self._agent_location = np.array([
            self.size,
            self.size / 2],
            dtype=np.float32)
        return self._get_observation(), self._get_info()

    def step(self, action):
        [x, y] = self._ball_location
        [vx, vy] = self._ball_velocity

        # TODO: use RK-4 integrator
        x += vx * self.dt
        y += vy * self.dt
        vy -= self.g * self.dt
        self._ball_location = np.array([x, y], dtype=np.float32)
        self._ball_velocity = np.array([vx, vy], dtype=np.float32)

        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size)
        # TODO: calc if terminated => detect collision or miss
        terminated = False
        # TODO: calc reward, if collided, calc distance from the center of an agent
        reward = 0

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
        [x, y] = self._ball_location
        draw_radial_gradient_circle(
            canvas,
            (PADDING + x, PADDING + y),
            16,
            (185, 67, 102),
            (204, 146, 155)
        )
        [x, y] = self._agent_location
        pygame.draw.rect(
            canvas,
            (144, 238, 144),
            pygame.Rect((x - 8, y - RACKET_SIZE / 2), (16, RACKET_SIZE))
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
