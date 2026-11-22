"""Small wheel spin on the assembled pendulum: ramp up, hold, ramp down, stop.

Runs on the Raspberry Pi with the pi3hat (needs root). The pendulum must hang
down and be free to swing; it only feels the reaction torque
I_wheel * wheel_acceleration (about 0.04 Nm at the defaults, roughly a 2 deg
tilt against ~1.1 Nm of gravity torque).

Profile (velocity mode, position = nan):
  ramp up    0 -> --speed rev/s at --accel rev/s^2
  hold       --hold s at --speed
  ramp down  --speed -> 0 at --accel
  settle     0 rev/s for 0.3 s, then set_stop()

Safety:
  - moteus maximum_torque --max-torque (default 0.05 Nm), watchdog 0.1 s;
  - abort + set_stop() on |velocity| > --abort-speed, any fault, a body
    rotation > --abort-tilt deg from the start (quaternion angle; Euler roll is
    unusable because the board sits near pitch 90 deg), Ctrl-C or any exception;
  - wheel position is not checked: the moteus position bounds are nan on this build.

The IMU is read in a separate pi3hat cycle: moteus-pi3hat 1.1.0 fails when a
moteus command and request_attitude=True share one cycle (see README).

Usage (on the Pi):
  sudo .venv/bin/python wheel_nudge.py --dry-run     # read-only: state, bounds, plan
  sudo .venv/bin/python wheel_nudge.py               # run, log to --out
"""
import argparse, asyncio, csv, math, sys, time

import moteus
import moteus_pi3hat

PERIOD_S = 0.01
WATCHDOG_S = 0.1
SETTLE_S = 0.3
FIELDS = ["t_s", "phase", "velocity_cmd_rev_s", "mode", "fault", "position_rev",
          "velocity_rev_s", "torque_nm", "q_current_a", "voltage_v",
          "imu_roll_deg", "imu_pitch_deg", "tilt_deg"]


class Abort(Exception):
    pass


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def profile(args):
    """Return (phase, end_time_s, velocity(t_in_phase)) tuples."""
    ramp = args.speed / args.accel
    return [
        ("ramp_up", ramp, lambda t: args.accel * t),
        ("hold", args.hold, lambda t: args.speed),
        ("ramp_down", ramp, lambda t: args.speed - args.accel * t),
        ("settle", SETTLE_S, lambda t: 0.0),
    ]


def planned_travel_rev(args):
    return args.speed * (args.speed / args.accel + args.hold)


async def read_imu(transport):
    """Return (roll_deg, pitch_deg, quaternion (w, x, y, z)) or NaNs/None."""
    results = await transport.cycle([], request_attitude=True)
    imu = next((r for r in results if isinstance(r, moteus_pi3hat.CanAttitudeWrapper)), None)
    if imu is None:
        return math.nan, math.nan, None
    q = imu.attitude
    return (math.degrees(imu.euler_rad.roll), math.degrees(imu.euler_rad.pitch),
            (q.w, q.x, q.y, q.z))


def rotation_deg(q0, q1):
    """Angle of the rotation between two attitudes. No gimbal lock, unlike Euler
    angles: the board is mounted with pitch near 90 deg, where roll jumps."""
    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


async def main(args):
    transport = moteus_pi3hat.Pi3HatRouter(servo_bus_map={args.bus: [args.id]})
    qr = moteus.QueryResolution()
    qr.q_current = moteus.F32
    controller = moteus.Controller(id=args.id, transport=transport, query_resolution=qr)
    R = moteus.Register

    # --- preflight, read-only -------------------------------------------------
    state = await asyncio.wait_for(controller.query(), 0.5)
    v = state.values
    roll0, pitch0, q0 = await read_imu(transport)
    print(f"moteus: mode {v[R.MODE]}, fault {v[R.FAULT]}, position {v[R.POSITION]:.3f} rev, "
          f"velocity {v[R.VELOCITY]:.3f} rev/s, {v[R.VOLTAGE]:.1f} V, {v[R.TEMPERATURE]:.0f} C")
    print(f"imu: roll {roll0:.1f} deg, pitch {pitch0:.1f} deg")
    travel = planned_travel_rev(args)
    print(f"planned travel {travel:.2f} rev")
    ramp = args.speed / args.accel
    print(f"plan: 0 -> {args.speed} rev/s in {ramp:.2f} s, hold {args.hold} s, "
          f"-> 0 in {ramp:.2f} s, settle {SETTLE_S} s; max torque {args.max_torque} Nm, "
          f"abort at |v| > {args.abort_speed} rev/s or tilt > {args.abort_tilt} deg")

    if v[R.FAULT] != 0 or abs(v[R.VELOCITY]) > 0.1:
        print("not stopped or faulted; refusing to run")
        return 1
    if q0 is None:
        print("no IMU attitude; refusing to run")
        return 1
    if args.dry_run:
        print("dry run: no command sent")
        return 0

    # --- run ------------------------------------------------------------------
    rows, start = [], time.perf_counter()
    try:
        for phase, duration, vel in profile(args):
            t0 = time.perf_counter()
            while (tp := time.perf_counter() - t0) < duration:
                cmd = vel(tp)
                r = await controller.set_position(
                    position=math.nan, velocity=cmd, accel_limit=args.accel,
                    maximum_torque=args.max_torque, watchdog_timeout=WATCHDOG_S,
                    query=True)
                roll, pitch, q = await read_imu(transport)
                rv = r.values
                rows.append([round(time.perf_counter() - start, 4), phase, round(cmd, 4),
                             rv[R.MODE], rv[R.FAULT], rv[R.POSITION], rv[R.VELOCITY],
                             rv[R.TORQUE], rv[R.Q_CURRENT], rv[R.VOLTAGE], roll, pitch,
                             rotation_deg(q0, q) if q else math.nan])
                if rv[R.FAULT] != 0:
                    raise Abort(f"fault {rv[R.FAULT]}")
                if abs(rv[R.VELOCITY]) > args.abort_speed:
                    raise Abort(f"velocity {rv[R.VELOCITY]:.2f} rev/s")
                tilt = rows[-1][-1]
                if not tilt <= args.abort_tilt:  # also aborts on a missing IMU reading
                    raise Abort(f"tilt {tilt:.1f} deg")
                await asyncio.sleep(max(0.0, PERIOD_S - (time.perf_counter() - start - rows[-1][0])))
        result = 0
    except Abort as e:
        print(f"ABORT: {e}")
        result = 1
    finally:
        await controller.set_stop()

    await asyncio.sleep(0.5)
    final = (await controller.query()).values
    print(f"stopped: mode {final[R.MODE]}, fault {final[R.FAULT]}, "
          f"velocity {final[R.VELOCITY]:.3f} rev/s")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"log: {args.out} ({len(rows)} rows)")
    if rows:
        peak_v = max(abs(r[6]) for r in rows)
        peak_t = max(abs(r[7]) for r in rows)
        peak_tilt = max(r[12] for r in rows)
        print(f"peak |velocity| {peak_v:.2f} rev/s, peak |torque| {peak_t:.3f} Nm, "
              f"peak tilt {peak_tilt:.1f} deg")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bus", type=int, default=1)
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--speed", type=float, default=2.0, help="hold speed, rev/s")
    p.add_argument("--accel", type=float, default=2.0, help="ramp rate, rev/s^2")
    p.add_argument("--hold", type=float, default=2.0, help="hold time, s")
    p.add_argument("--max-torque", type=float, default=0.05, help="Nm")
    p.add_argument("--abort-speed", type=float, default=2.5, help="rev/s")
    p.add_argument("--abort-tilt", type=float, default=10.0, help="deg, body rotation from start")
    p.add_argument("--out", default=f"/tmp/wheel_nudge-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    p.add_argument("--dry-run", action="store_true")
    sys.exit(asyncio.run(main(p.parse_args())))
