import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from numpy import pi, sin, cos

from fip_env.commons.graphics import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_PINK, COLOR_GREEN)
from fip_env.commons.physics import integrate_rk4, normalize_angle

RENDER_FPS = 24


class FlywheelInvertedPendulumEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 24}

    def __init__(self,
                 kick_probability=0.0,
                 kick_strength=0.2,
                 max_steps=400,
                 theta_threshold=pi / 12,
                 dtheta_threshold=10,
                 dphi_threshold=10.0,
                 max_torque=50,
                 render_mode=None,
                 verbose_termination=False):

        # params for simulating an external disturbance
        # added to theta_dot
        self.kick_probability = kick_probability
        self.kick_strength = kick_strength

        self.max_steps = max_steps

        self.theta_threshold = theta_threshold  # rod angle limit exceeding which the episode terminates
        self.dtheta_threshold = dtheta_threshold  # max angular velocity of the rod
        self.dphi_threshold = dphi_threshold  # max angular velocity of the wheel
        self.max_torque = max_torque  # maximal torque applied to the wheel
        self.verbose_termination = verbose_termination

        # physical model parameters
        self.g: float = 9.81  # gravity
        self.m1: float = 0.1  # mass of the rod
        self.l: float = 2.0  # length of the rod
        self.m2: float = 0.5  # mass of the wheel
        self.r: float = 0.75  # radius of the flywheel
        self.b: float = 0.0  # friction coefficient between the rod and the wheel
        self.dt: float = 1.0 / RENDER_FPS  # timestep

        # state variables
        self.theta: float = 0.0  # angle of the rod
        self.theta_dot: float = 0.0  # angular velocity of the rod
        self.phi: float = 0.0  # angle of the wheel (we don't care about it)
        self.phi_dot: float = 0.0  # angular velocity of the wheel

        self._step: int = 0  # current episode timestamp
        self._t: float = 0.0  # episode time
        self._current_action: float = 0.0  # applied torque
        self._last_action: float = 0.0

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
            1.0,  # cos(theta)
            1.0,  # sin(theta)
            self.dtheta_threshold,  # angular velocity of the rod
            self.dphi_threshold,  # angular velocity of the wheel
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._step = 0
        self._t = 0.0

        rand = self.np_random
        deviation = pi / 8
        self.theta = pi - rand.uniform(low=-deviation, high=deviation, size=(1,))[0]
        self.theta_dot = rand.uniform(low=-2, high=2, size=(1,))[0]

        self.phi = rand.uniform(low=-pi, high=pi, size=(1,))[0]
        self.phi_dot = rand.uniform(low=-2, high=2, size=(1,))[0]

        return self._get_obs(), {}

    def _derivate(self, state: np.ndarray[float], _step: int, _t: float, _dt: float) -> np.ndarray[float]:
        """
        :param state:
        :param _step:
        :param _t:
        :param _dt:
        :return: derivatives of all state variables of the flywheel inverted pendulum
        """
        [_th, _dth, _phi, _dphi] = state
        [l, m1, m2, r, b, g] = [self.l, self.m1, self.m2, self.r, self.b, self.g]
        friction = - b * (_dphi - _dth)
        force = self._current_action + friction
        J = m2 * r ** 2
        ddth = (force - (0.5 * m1 + m2) * l * g * sin(-_th)) / (m1 * l ** 2 / 3 + m2 * l ** 2 + J)
        ddphi = force / J
        return np.array([_dth, ddth, _dphi, ddphi], dtype=float)

    def _get_obs(self):
        return np.array([
            cos(self.theta),
            sin(self.theta),
            self.theta_dot,
            self.phi_dot], dtype=np.float32)

    def _terminated(self) -> bool:
        # if abs(self.phi_dot) > self.dphi_threshold:
        #     if self.verbose_termination:
        #         print(f"|phi_dot| > {self.dphi_threshold}")
        #     return True
        #
        # if abs(self.theta_dot) > self.dtheta_threshold:
        #     if self.verbose_termination:
        #         print(f"|theta_dot| > {self.dtheta_threshold}")
        #     return True
        #
        # _n = 6
        # if abs(self.theta) > _n * pi:
        #     if self.verbose_termination:
        #         print(f"|theta| > {_n} * PI")
        #     return True
        #
        # angle_threshold = pi # impossible condition
        # if abs(normalize_angle(self.theta)) > angle_threshold :
        #     if self.verbose_termination:
        #         print(f"|normalized theta| > {angle_threshold}")
        #     return True

        return False

    def _reward(self):
        n_theta = normalize_angle(self.theta)
        termination_penalty = 0.0  # 100.0 if self._terminated() else 0.0
        return -(
                n_theta ** 2 +
                0.25 * self.theta_dot ** 2 +
                0.01 * self.phi_dot ** 2
        ) - termination_penalty

    def step(self, action: float):
        self._last_action = self._current_action
        self._current_action = np.clip(action, -self.max_torque, self.max_torque)[0]

        state = [self.theta, self.theta_dot, self.phi, self.phi_dot]
        state = integrate_rk4(state, self._step, self._t, self.dt, self._derivate)

        if np.random.random() < self.kick_probability:
            # apply kick to the pole's angular velocity
            state[1] += np.random.uniform(-self.kick_strength, self.kick_strength)

        self._step += 1
        self._t += self.dt

        self.theta = state[0]
        self.theta_dot = state[1]
        self.phi = state[2]
        self.phi_dot = state[3]

        truncated = self._step > self.max_steps

        if self.render_mode == "human":
            self._render_to_window()

        return self._get_obs(), self._reward(), self._terminated(), truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_to_window()

    def _render_to_surface(self) -> pygame.Surface:
        scale = 50.0
        [offset_x, offset_y, offset_angle] = [self.window_size / 2, self.window_size / 2, pi / 2]
        theta = self.theta + offset_angle
        phi = self.phi + offset_angle
        origin = np.array([offset_x, offset_y], dtype=np.float32)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(COLOR_BLACK)

        l = self.l * scale
        r = self.r * scale
        rod_end = np.array([l * cos(theta), l * sin(theta)])
        pygame.draw.line(
            canvas,
            COLOR_BLUE,
            origin.tolist(),
            (rod_end + origin).tolist(), width=3)

        # red tick
        pygame.draw.line(
            canvas,
            COLOR_PINK,
            (np.array([0.5 * r * cos(phi), 0.5 * r * sin(phi)]) + rod_end + origin).tolist(),
            (np.array([r * cos(phi), r * sin(phi)]) + rod_end + origin).tolist(),
            width=4
        )

        # green tick
        phi2 = phi + pi
        pygame.draw.line(
            canvas,
            COLOR_GREEN,
            (np.array([0.5 * r * cos(phi2), 0.5 * r * sin(phi2)]) + rod_end + origin).tolist(),
            (np.array([r * cos(phi2), r * sin(phi2)]) + rod_end + origin).tolist(),
            width=4
        )

        pygame.draw.circle(canvas, COLOR_BLUE, (rod_end + origin).tolist(), r, width=3)

        flipped = pygame.transform.flip(canvas, False, True)

        # print time, step and state parameters
        font = pygame.font.SysFont('Courier New', 20)
        text_surface = font.render(f"Time: {self._t:>6.1f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topleft=(10, 10))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"Step: {self._step:>6}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topleft=(10, 35))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"Reward: {self._reward():>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topleft=(10, 60))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"Θ: {self.theta:>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topright=(self.window_size - 10, 10))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"Ω: {self.theta_dot:>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topright=(self.window_size - 10, 35))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"Ω: {self.theta_dot:>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topright=(self.window_size - 10, 35))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"dφ: {self.phi_dot:>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topright=(self.window_size - 10, 60))
        flipped.blit(text_surface, text_rect)

        text_surface = font.render(f"u: {self._current_action:>6.2f}", True, COLOR_BLUE)
        text_rect = text_surface.get_rect(topright=(self.window_size - 10, 85))
        flipped.blit(text_surface, text_rect)

        return flipped

    def _render_frame(self):
        canvas = self._render_to_surface()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def _render_to_window(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Flywheel Inverted Pendulum')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = self._render_to_surface()
        self.window.blit(canvas, canvas.get_rect(topleft=(0, 0)))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])
