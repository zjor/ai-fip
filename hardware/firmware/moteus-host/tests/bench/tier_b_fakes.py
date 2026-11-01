"""Fakes for Tier B session tests: a physics clock, a disc-integrating controller and a conf stream."""

import math
from types import SimpleNamespace

import moteus


INERTIA = 3.54e-4
KT = 0.0244
SNAPSHOT = ("servo.timeout_mode 12\nservopos.position_min -1\nservopos.position_max 1\n"
            "servo.max_velocity 500\nservo.flux_brake_min_voltage 34\nmotor.Kv 330\n")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    async def wait(self, deadline):
        self.now = max(self.now, deadline)


class FakeController:
    """Integrates disc speed from the last command: velocity mode ramps at accel_limit,
    torque mode accelerates by torque/(2*pi*I), stopped decays by friction.

    Torque-speed extras: with no_load_rev_s the torque tapers from 80 % of no-load to zero
    (a voltage limit) and friction acts; VOLTAGE reports bus_v, or sag_to_v while
    |q| > sag_above_a, or sag_to_v once the commanded phase has run sag_after_s (independent
    of q); stuck reports velocity 0 for any command."""

    def __init__(self, clock, *, velocity=0.0, friction_nm=0.004, fail_spin_up=False,
                 drop_stop_replies=0, drop_query_every=0, mode_11_after_s=None, noise=0.0,
                 enable_s=0.0, bus_v=24.0, no_load_rev_s=None, sag_above_a=None, sag_to_v=None,
                 stuck=False, sag_after_s=None):
        self.bus_v = bus_v
        self.no_load_rev_s = no_load_rev_s
        self.sag_above_a = sag_above_a
        self.sag_to_v = sag_to_v
        self.sag_after_s = sag_after_s
        self.stuck = stuck
        self.clock = clock
        self.enable_s = enable_s         # firmware enable sequence: modes 2, 3 before position mode 10
        self.enabled_at = None
        self.noise = noise               # alternating ± jitter on the reported velocity
        self.reads = 0
        self.velocity = velocity
        self.friction = friction_nm
        self.fail_spin_up = fail_spin_up
        self.drop_stop_replies = drop_stop_replies
        self.drop_query_every = drop_query_every
        self.mode_11_after_s = mode_11_after_s
        self.queries = 0
        self.mode = 0
        self.command = None
        self.commands = []
        self.stops = 0
        self.last_t = clock.now

    def _integrate(self):
        dt = self.clock.now - self.last_t
        self.last_t = self.clock.now
        if dt <= 0.0:
            return
        if self.mode == 10 and self.command is not None:
            torque = self.command.get("feedforward_torque", 0.0)
            if math.isnan(self.command["position"]) and torque and self.no_load_rev_s is not None:
                taper = min(1.0, max(0.0, (self.no_load_rev_s - abs(self.velocity)) / (0.2 * self.no_load_rev_s)))
                available = torque * taper
                if self.velocity == 0.0 and abs(available) <= self.friction:
                    return
                friction = math.copysign(self.friction, self.velocity if self.velocity else available)
                self.velocity += (available - friction) / (2 * math.pi * INERTIA) * dt
            elif math.isnan(self.command["position"]) and torque:
                self.velocity += torque / (2 * math.pi * INERTIA) * dt
            else:
                # a failing spin-up reaches only 30 % of the target (supply current-limited)
                target = self.command["velocity"] * (0.3 if self.fail_spin_up else 1.0)
                step = self.command["accel_limit"] * dt
                self.velocity += max(-step, min(step, target - self.velocity))
        else:
            decay = self.friction / (2 * math.pi * INERTIA) * dt
            self.velocity = math.copysign(max(abs(self.velocity) - decay, 0.0), self.velocity)

    def result(self):
        self.clock.now += 0.001
        if self.mode == 10 and self.mode_11_after_s is not None and self.clock.now >= self.mode_11_after_s:
            self.mode = 11          # firmware watchdog fired: torque off, commands ignored until a stop
        self._integrate()
        if self.stuck:
            self.velocity = 0.0
        mode = self.mode
        if mode == 10 and self.enabled_at is not None and self.clock.now - self.enabled_at < self.enable_s:
            mode = 2 if self.clock.now - self.enabled_at < self.enable_s / 4 else 3
        q = 0.0
        if self.mode == 10 and self.command is not None:
            q = self.command.get("feedforward_torque", 0.0) / KT or 0.5
        self.reads += 1
        reported = self.velocity + (self.noise if self.reads % 2 else -self.noise)
        voltage = self.bus_v
        if self.sag_above_a is not None and abs(q) > self.sag_above_a:
            voltage = self.sag_to_v
        if (self.sag_after_s is not None and self.mode == 10 and self.enabled_at is not None
                and self.clock.now - self.enabled_at >= self.sag_after_s):
            voltage = self.sag_to_v
        values = dict(MODE=mode, FAULT=0, POSITION=1.0, VELOCITY=reported,
                      TORQUE=q * KT, Q_CURRENT=q, D_CURRENT=0.0, VOLTAGE=voltage, POWER=0.0,
                      TEMPERATURE=25.0, HOME_STATE=1, TRAJECTORY_COMPLETE=1)
        return SimpleNamespace(values={getattr(moteus.Register, k): v for k, v in values.items()})

    async def set_stop(self, **kwargs):
        self.stops += 1
        if self.drop_stop_replies > 0 and self.command is not None:
            # lose the reply to the stop that ends a commanded phase (the cleanup stop)
            self.drop_stop_replies -= 1
            raise TimeoutError("stop reply lost")
        reply = self.result()
        self.mode = 0
        self.command = None
        self.enabled_at = None
        return reply

    async def query(self):
        self.queries += 1
        if self.drop_query_every and self.queries % self.drop_query_every == 0:
            raise TimeoutError("query reply lost")
        return self.result()

    async def set_position(self, **kwargs):
        self.commands.append(kwargs)
        if self.mode == 11:
            return self.result()    # ignored by the firmware until a stop
        if self.mode == 0:
            self.enabled_at = self.clock.now
        self.command = kwargs
        self.mode = 10
        return self.result()


class FakeStream:
    def __init__(self):
        self.settings = {b"servo.timeout_mode": b"12", b"servopos.position_min": b"-1",
                         b"servopos.position_max": b"1", b"servo.max_velocity": b"500",
                         b"servo.flux_brake_min_voltage": b"34"}
        self.writes = []
        self.flushed = False

    async def flush_read(self):
        self.flushed = True

    async def command(self, command, allow_any_response=False):
        self.writes.append(command)
        words = command.split()
        if words[:2] == [b"conf", b"write"]:
            raise AssertionError("conf write is forbidden")
        if words[:2] == [b"conf", b"set"]:
            self.settings[words[2]] = words[3]
            return b""
        if words[:2] == [b"conf", b"get"]:
            if not allow_any_response:
                raise TimeoutError("conf get never produces OK")
            return self.settings[words[2]]
        raise AssertionError(f"unexpected stream command {command!r}")

