import gymnasium as gym
from gymnasium import spaces


class BallCatcher(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, size=512):
        self.size = size
        self.window_size = size
        """
        TODO: declare observation space
        - ball location
        - ball previous location (agent should derive the concept of velocity)
        - agent's bat location
        """

        # TODO: declare action space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # TODO: self._ball_location

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        ...
