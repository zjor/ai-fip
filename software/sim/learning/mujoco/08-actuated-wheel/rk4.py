"""Analytical reaction-wheel pendulum, stepped with fourth-order Runge-Kutta.

The state is [pivot angle, pivot rate, absolute wheel rate]. A positive motor
torque accelerates the wheel and produces an opposite reaction on the carrier.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Parameters:
    carrier_inertia: float
    wheel_inertia: float
    gravity_coefficient: float
    pivot_damping: float
    wheel_hinge_damping: float


def parameters_from_model(model: mujoco.MjModel) -> Parameters:
    """Read the compiled scene's physical constants, without using its dynamics."""
    if model.nv != 2:
        raise ValueError("expected exactly the pivot and wheel-hinge joints")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_matrix)

    # For zero absolute wheel speed, qvel = [pivot_rate, -pivot_rate].
    carrier_motion = np.array([1.0, -1.0])
    carrier_inertia = float(carrier_motion @ mass_matrix @ carrier_motion)
    wheel_inertia = float(mass_matrix[1, 1])

    data.joint("pivot").qpos[0] = np.pi / 2.0
    mujoco.mj_forward(model, data)
    gravity_coefficient = float(-data.qfrc_bias[0])

    return Parameters(
        carrier_inertia=carrier_inertia,
        wheel_inertia=wheel_inertia,
        gravity_coefficient=gravity_coefficient,
        pivot_damping=float(model.dof_damping[0]),
        wheel_hinge_damping=float(model.dof_damping[1]),
    )


def derivative(state: np.ndarray, torque: float, params: Parameters) -> np.ndarray:
    angle, angle_rate, wheel_rate = state
    relative_wheel_rate = wheel_rate - angle_rate
    wheel_net_torque = torque - params.wheel_hinge_damping * relative_wheel_rate
    return np.array(
        [
            angle_rate,
            (
                params.gravity_coefficient * np.sin(angle)
                - params.pivot_damping * angle_rate
                - wheel_net_torque
            )
            / params.carrier_inertia,
            wheel_net_torque / params.wheel_inertia,
        ]
    )


def step(state: np.ndarray, torque: float, dt: float, params: Parameters) -> np.ndarray:
    """Advance one interval with torque held constant throughout the interval."""
    k1 = derivative(state, torque, params)
    k2 = derivative(state + 0.5 * dt * k1, torque, params)
    k3 = derivative(state + 0.5 * dt * k2, torque, params)
    k4 = derivative(state + dt * k3, torque, params)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
