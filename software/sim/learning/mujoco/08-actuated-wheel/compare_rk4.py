"""Reproducibly compare MuJoCo and analytical RK4 trajectories."""

import math
from pathlib import Path

import mujoco
import numpy as np

from controller import lqr_gain, reduced_state
from rk4 import parameters_from_model, step


SCENE = Path(__file__).with_name("scene.xml")
MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
PARAMS = parameters_from_model(MODEL)
GAIN, _, _ = lqr_gain(MODEL)
ACTUATOR_ID = MODEL.actuator("wheel-torque").id
DT = float(MODEL.opt.timestep)
EULER_LIMITS = {
    # Maximum angle (deg), pivot rate (rad/s), wheel rate (rad/s), torque (Nm).
    "free fall": (0.8, 0.02, 0.05, 0.0),
    "fixed pulse": (0.2, 0.02, 0.05, 0.0),
    "LQR": (0.2, 0.02, 0.2, 0.02),
}
RK4_LIMITS = {name: (1e-5, 1e-5, 1e-5, 1e-5) for name in EULER_LIMITS}


def compare(
    name: str, angle_deg: float, duration_s: float, limits: tuple[float, ...]
) -> None:
    data = mujoco.MjData(MODEL)
    mujoco.mj_resetData(MODEL, data)
    data.joint("pivot").qpos[0] = math.radians(angle_deg)
    mujoco.mj_forward(MODEL, data)
    analytical = reduced_state(data)
    max_errors = np.zeros(3)
    max_torque_difference = 0.0

    for index in range(round(duration_s / DT)):
        time_s = index * DT
        if name == "free fall":
            mujoco_torque = analytical_torque = 0.0
        elif name == "fixed pulse":
            mujoco_torque = analytical_torque = 0.2 if time_s < 0.1 else 0.0
        elif name == "LQR":
            mujoco_torque = float(np.clip(-GAIN @ reduced_state(data), -1.7, 1.7))
            analytical_torque = float(np.clip(-GAIN @ analytical, -1.7, 1.7))
        else:
            raise ValueError(f"unknown scenario: {name}")

        data.ctrl[ACTUATOR_ID] = mujoco_torque
        mujoco.mj_step(MODEL, data)
        analytical = step(analytical, analytical_torque, DT, PARAMS)
        error = reduced_state(data) - analytical
        error[0] = (error[0] + math.pi) % (2.0 * math.pi) - math.pi
        max_errors = np.maximum(max_errors, np.abs(error))
        max_torque_difference = max(
            max_torque_difference, abs(mujoco_torque - analytical_torque)
        )

    print(
        f"{name:11} {duration_s:.1f}s  "
        f"max |Δθ|={math.degrees(max_errors[0]):.8f} deg  "
        f"|Δθ̇|={max_errors[1]:.8f} rad/s  "
        f"|Δω|={max_errors[2]:.8f} rad/s  "
        f"|Δτ|={max_torque_difference:.8f} Nm"
    )
    observed = (math.degrees(max_errors[0]), *max_errors[1:], max_torque_difference)
    if any(value > limit for value, limit in zip(observed, limits)):
        raise AssertionError(f"{name}: observed {observed} exceeds {limits}")


def main() -> None:
    print(f"MuJoCo timestep: {DT:.4f} s")
    print(f"analytical parameters: {PARAMS}")
    for integrator, limits in (
        (mujoco.mjtIntegrator.mjINT_EULER, EULER_LIMITS),
        (mujoco.mjtIntegrator.mjINT_RK4, RK4_LIMITS),
    ):
        MODEL.opt.integrator = integrator
        print(f"MuJoCo integrator: {integrator.name}")
        compare("free fall", angle_deg=10.0, duration_s=0.7, limits=limits["free fall"])
        compare("fixed pulse", angle_deg=0.0, duration_s=0.5, limits=limits["fixed pulse"])
        compare("LQR", angle_deg=5.0, duration_s=2.0, limits=limits["LQR"])
    print("trajectory checks: PASS")


if __name__ == "__main__":
    main()
