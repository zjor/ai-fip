"""LQR design and state helpers for the actuated flywheel pendulum."""

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are


STATE_COST = np.diag([120.0, 2.0, 0.015])
CONTROL_COST = np.array([[0.8]])


def reduced_state(data: mujoco.MjData) -> np.ndarray:
    """Return [pendulum angle, pendulum rate, absolute wheel rate]."""
    pivot_rate = data.joint("pivot").qvel[0]
    relative_wheel_rate = data.joint("wheel-hinge").qvel[0]
    return np.array(
        [data.joint("pivot").qpos[0], pivot_rate, pivot_rate + relative_wheel_rate]
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
