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


class BallCatcherEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, size=512):
        self.size = size
        self.window_size = size + PADDING * 2

        self._ball_location = np.array([0, self.size / 2], dtype=np.float32)
        self._ball_prev_location = np.array([0, self.size / 2], dtype=np.float32)
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._ball_location = np.array([0, self.size / 2], dtype=np.float32)
        self._ball_prev_location = np.array([0, self.size / 2], dtype=np.float32)
        self._agent_location = np.array([
            self.size,
            self.size / 2],
            dtype=np.float32)

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
        return canvas

    def _render_frame(self):
        canvas = self._render_to_surface()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
