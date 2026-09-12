"""LQR design and state helpers for the actuated flywheel pendulum."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are


STATE_COST = np.diag([120.0, 2.0, 0.015])
CONTROL_COST = np.array([[0.8]])
SWING_UP_TORQUE_LIMIT_NM = 0.5
ENERGY_GAIN = 60.0
CATCH_CONE_DEG = 30.0
ENERGY_TOLERANCE_FRACTION = 0.05
LQR_FALLBACK_ANGLE_DEG = 45.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def reduced_state(data: mujoco.MjData) -> np.ndarray:
    """Return [pendulum angle, pendulum rate, absolute wheel rate]."""
    pivot_rate = data.joint("pivot").qvel[0]
    relative_wheel_rate = data.joint("wheel-hinge").qvel[0]
    return np.array(
        [
            wrap_angle(data.joint("pivot").qpos[0]),
            pivot_rate,
            pivot_rate + relative_wheel_rate,
        ]
    )


def linearize(
    model: mujoco.MjModel, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference the continuous reduced dynamics at upright equilibrium."""
    scratch = mujoco.MjData(model)
    actuator_id = model.actuator("wheel-torque").id

    def dynamics(state: np.ndarray, torque: float) -> np.ndarray:
        mujoco.mj_resetData(model, scratch)
        scratch.joint("pivot").qpos[0] = state[0]
        scratch.joint("wheel-hinge").qpos[0] = 0.0
        scratch.joint("pivot").qvel[0] = state[1]
        scratch.joint("wheel-hinge").qvel[0] = state[2] - state[1]
        scratch.ctrl[actuator_id] = torque
        mujoco.mj_forward(model, scratch)
        return np.array(
            [state[1], scratch.qacc[0], scratch.qacc[0] + scratch.qacc[1]]
        )

    equilibrium = np.zeros(3)
    basis = np.eye(3)
    a = np.column_stack(
        [
            (
                dynamics(equilibrium + basis[i] * epsilon, 0.0)
                - dynamics(equilibrium - basis[i] * epsilon, 0.0)
            )
            / (2.0 * epsilon)
            for i in range(3)
        ]
    )
    b = (
        dynamics(equilibrium, epsilon) - dynamics(equilibrium, -epsilon)
    )[:, np.newaxis] / (2.0 * epsilon)
    return a, b


def lqr_gain(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the continuous-time LQR gain and the linearized A/B matrices."""
    a, b = linearize(model)
    solution = solve_continuous_are(a, b, STATE_COST, CONTROL_COST)
    gain = np.linalg.solve(CONTROL_COST, b.T @ solution).reshape(3)
    return gain, a, b


def control_torque(
    model: mujoco.MjModel, data: mujoco.MjData, gain: np.ndarray
) -> tuple[float, float]:
    """Return requested and range-limited wheel torque for the current state."""
    requested = float(-gain @ reduced_state(data))
    actuator_id = model.actuator("wheel-torque").id
    lower, upper = model.actuator_ctrlrange[actuator_id]
    applied = float(np.clip(requested, lower, upper))
    return requested, applied


def pendulum_energy_parameters(model: mujoco.MjModel) -> tuple[float, float]:
    """Return pendulum inertia and gravity coefficient for energy shaping."""
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, scratch, mass_matrix)

    # If the wheel has zero absolute rate, qvel = [theta_rate, -theta_rate].
    carrier_motion = np.array([1.0, -1.0])
    inertia = float(carrier_motion @ mass_matrix @ carrier_motion)

    scratch.joint("pivot").qpos[0] = math.pi / 2.0
    mujoco.mj_forward(model, scratch)
    gravity_coefficient = float(-scratch.qfrc_bias[0])
    return inertia, gravity_coefficient


@dataclass(frozen=True)
class ControlOutput:
    mode: str
    requested_torque_nm: float
    applied_torque_nm: float
    energy_error_j: float


class SwingUpLQRController:
    """Energy-shaping swing-up with an LQR catch and fallback."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.gain, _, _ = lqr_gain(model)
        self.inertia, self.gravity_coefficient = pendulum_energy_parameters(model)
        self.actuator_id = model.actuator("wheel-torque").id
        self.mode = "swing-up"
        self.catch_count = 0

    def reset(self) -> None:
        self.mode = "swing-up"
        self.catch_count = 0

    def update(self, data: mujoco.MjData) -> ControlOutput:
        state = reduced_state(data)
        angle, angle_rate, _ = state
        energy = (
            0.5 * self.inertia * angle_rate**2
            + self.gravity_coefficient * (math.cos(angle) - 1.0)
        )
        energy_error = -energy

        if self.mode == "lqr" and abs(angle) > math.radians(LQR_FALLBACK_ANGLE_DEG):
            self.mode = "swing-up"

        if self.mode == "swing-up":
            catch_energy_tolerance = (
                ENERGY_TOLERANCE_FRACTION * 2.0 * self.gravity_coefficient
            )
            if (
                abs(angle) < math.radians(CATCH_CONE_DEG)
                and abs(energy_error) < catch_energy_tolerance
            ):
                self.mode = "lqr"
                self.catch_count += 1

        if self.mode == "lqr":
            requested = float(-self.gain @ state)
            lower, upper = self.model.actuator_ctrlrange[self.actuator_id]
        else:
            # Motor torque reacts on the carrier with the opposite sign.
            requested = -ENERGY_GAIN * energy_error * angle_rate
            lower, upper = -SWING_UP_TORQUE_LIMIT_NM, SWING_UP_TORQUE_LIMIT_NM

        applied = float(np.clip(requested, lower, upper))
        return ControlOutput(self.mode, requested, applied, energy_error)
