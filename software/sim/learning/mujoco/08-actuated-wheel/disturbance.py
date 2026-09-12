"""Seeded random force pulses applied tangentially at the wheel centre."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class KickStatus:
    force_n: float = 0.0
    started: bool = False
    number: int = 0


class RandomTangentialKicks:
    """Schedule reproducible random kicks while stabilization is active."""

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.next_start_s: float | None = None
        self.active_until_s = 0.0
        self.force_n = 0.0
        self.count = 0

    def reset(self) -> None:
        self.next_start_s = None
        self.active_until_s = 0.0
        self.force_n = 0.0
        self.count = 0

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, enabled: bool
    ) -> KickStatus:
        data.qfrc_applied[:] = 0.0
        if not enabled:
            return KickStatus(number=self.count)

        if self.next_start_s is None:
            self.next_start_s = data.time + float(self.rng.uniform(1.0, 1.5))

        started = False
        if data.time >= self.active_until_s and self.force_n != 0.0:
            self.force_n = 0.0
            self.next_start_s = data.time + float(self.rng.uniform(2.5, 3.5))

        if self.force_n == 0.0 and data.time >= self.next_start_s:
            direction = float(self.rng.choice((-1.0, 1.0)))
            self.force_n = direction * float(self.rng.uniform(2.0, 4.0))
            self.active_until_s = data.time + 0.04
            self.count += 1
            started = True

        if self.force_n != 0.0:
            pendulum_rotation = data.body("pendulum").xmat.reshape(3, 3)
            rod_direction = pendulum_rotation[:, 2]
            tangent = pendulum_rotation[:, 0]
            np.testing.assert_allclose(tangent @ rod_direction, 0.0, atol=1e-12)
            force = self.force_n * tangent
            point = data.body("wheel").xpos.copy()
            body_id = model.body("wheel").id
            mujoco.mj_applyFT(
                model,
                data,
                force,
                np.zeros(3),
                point,
                body_id,
                data.qfrc_applied,
            )

        return KickStatus(self.force_n, started, self.count)
