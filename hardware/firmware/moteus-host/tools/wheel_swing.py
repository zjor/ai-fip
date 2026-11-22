"""Make the hanging pendulum swing: drive the wheel velocity as a sine at the
pendulum's natural frequency, then stop the wheel and log the free swing.

Runs on the Raspberry Pi with the pi3hat (needs root). The pendulum must hang
down, be free to swing, and have clear space for the --abort-tilt angle.

Wheel velocity command (velocity mode, position = nan):
  v(t) = -A * env(t) * sin(2 pi f t),  env ramps 0 -> 1 over the first cycle
for --cycles whole cycles, so v ends at 0. The reaction torque on the body is
about I_wheel * A * 2 pi * 2 pi f (A in rev/s), ~0.033 Nm at the defaults, and
moteus caps it at --max-torque. Wheel position is not checked: the moteus
position bounds (servopos.position_min/max) are nan on this build.
After the drive: 0.3 s at 0 rev/s, set_stop(), then --coast s of query-only
logging while the pendulum swings freely.

Default f = 1.42 Hz: measured free swing of the hanging pendulum on
2026-11-22 (gyro z, period 0.704 s, wheel axis 13.5 cm from the pivot). The
CAD model in software/sim/phase0.toml predicts 0.94 Hz; it does not match this
build.

Safety: maximum_torque cap, 0.1 s watchdog, abort + set_stop() on fault,
|velocity| > --abort-speed, body rotation > --abort-tilt deg from the start
(quaternion angle), Ctrl-C or any exception.

Usage (on the Pi, next to wheel_nudge.py):
  sudo .venv/bin/python wheel_swing.py --dry-run
  sudo .venv/bin/python wheel_swing.py [--amp 0.2] [--freq 1.42] [--cycles 8]
"""
import argparse, asyncio, csv, math, sys, time

import moteus
import moteus_pi3hat

from wheel_nudge import Abort, PERIOD_S, WATCHDOG_S, read_imu, rotation_deg

SETTLE_S = 0.3
FIELDS = ["t_s", "phase", "velocity_cmd_rev_s", "mode", "fault", "position_rev",
          "velocity_rev_s", "torque_nm", "q_current_a", "tilt_deg",
          "rate_x_dps", "rate_y_dps", "rate_z_dps"]


async def read_imu_rates(transport):
    results = await transport.cycle([], request_attitude=True)
    imu = next((r for r in results if isinstance(r, moteus_pi3hat.CanAttitudeWrapper)), None)
    if imu is None:
        return None, (math.nan,) * 3
    q, r = imu.attitude, imu.rate_dps
    return (q.w, q.x, q.y, q.z), (r.x, r.y, r.z)


def velocity_cmd(args, t):
    period = 1.0 / args.freq
    env = min(1.0, t / period)
    return -args.amp * env * math.sin(2 * math.pi * args.freq * t)


async def main(args):
    transport = moteus_pi3hat.Pi3HatRouter(servo_bus_map={args.bus: [args.id]})
    qr = moteus.QueryResolution()
    qr.q_current = moteus.F32
    controller = moteus.Controller(id=args.id, transport=transport, query_resolution=qr)
    R = moteus.Register

    state = await asyncio.wait_for(controller.query(), 0.5)
    v = state.values
    _, _, q0 = await read_imu(transport)
    drive_s = args.cycles / args.freq
    torque_est = 0.0029 * args.amp * 2 * math.pi * 2 * math.pi * args.freq
    print(f"moteus: mode {v[R.MODE]}, fault {v[R.FAULT]}, position {v[R.POSITION]:.3f} rev, "
          f"velocity {v[R.VELOCITY]:.3f} rev/s, {v[R.VOLTAGE]:.1f} V, {v[R.TEMPERATURE]:.0f} C")
    print(f"plan: wheel velocity -{args.amp} * sin(2 pi {args.freq} t), {args.cycles} cycles "
          f"({drive_s:.1f} s), reaction torque ~{torque_est:.3f} Nm (cap {args.max_torque}), "
          f"then stop and "
          f"log {args.coast} s of free swing; abort at tilt > {args.abort_tilt} deg")

    if v[R.FAULT] != 0 or abs(v[R.VELOCITY]) > 0.1:
        print("not stopped or faulted; refusing to run")
        return 1
    if q0 is None:
        print("no IMU attitude; refusing to run")
        return 1
    if args.dry_run:
        print("dry run: no command sent")
        return 0

    rows, start = [], time.perf_counter()

    async def log(phase, cmd, rv):
        q, rates = await read_imu_rates(transport)
        tilt = rotation_deg(q0, q) if q else math.nan
        rows.append([round(time.perf_counter() - start, 4), phase, round(cmd, 4),
                     rv[R.MODE], rv[R.FAULT], rv[R.POSITION], rv[R.VELOCITY],
                     rv[R.TORQUE], rv[R.Q_CURRENT], tilt, *rates])
        return tilt

    async def pace():
        await asyncio.sleep(max(0.0, PERIOD_S - (time.perf_counter() - start - rows[-1][0])))

    result = 0
    try:
        for phase, duration, cmd_fn in (("drive", drive_s, lambda t: velocity_cmd(args, t)),
                                        ("settle", SETTLE_S, lambda t: 0.0)):
            t0 = time.perf_counter()
            while (tp := time.perf_counter() - t0) < duration:
                cmd = cmd_fn(tp)
                r = await controller.set_position(
                    position=math.nan, velocity=cmd, maximum_torque=args.max_torque,
                    watchdog_timeout=WATCHDOG_S, query=True)
                rv = r.values
                tilt = await log(phase, cmd, rv)
                if rv[R.FAULT] != 0:
                    raise Abort(f"fault {rv[R.FAULT]}")
                if abs(rv[R.VELOCITY]) > args.abort_speed:
                    raise Abort(f"velocity {rv[R.VELOCITY]:.2f} rev/s")
                if not tilt <= args.abort_tilt:
                    raise Abort(f"tilt {tilt:.1f} deg")
                await pace()
    except Abort as e:
        print(f"ABORT: {e}")
        result = 1
    finally:
        await controller.set_stop()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.coast:
        r = await controller.query()
        await log("coast", 0.0, r.values)
        await pace()

    final = (await controller.query()).values
    print(f"stopped: mode {final[R.MODE]}, fault {final[R.FAULT]}, "
          f"velocity {final[R.VELOCITY]:.3f} rev/s, position {final[R.POSITION]:.3f} rev")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    drive = [r for r in rows if r[1] != "coast"]
    print(f"log: {args.out} ({len(rows)} rows)")
    if drive:
        print(f"drive: peak |velocity| {max(abs(r[6]) for r in drive):.2f} rev/s, "
              f"peak |torque| {max(abs(r[7]) for r in drive):.3f} Nm, "
              f"peak tilt {max(r[9] for r in drive):.1f} deg; "
              f"peak tilt overall {max(r[9] for r in rows):.1f} deg")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bus", type=int, default=1)
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--amp", type=float, default=0.2, help="wheel velocity amplitude, rev/s")
    p.add_argument("--freq", type=float, default=1.42, help="drive frequency, Hz")
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--coast", type=float, default=4.0, help="free-swing logging after stop, s")
    p.add_argument("--max-torque", type=float, default=0.05, help="Nm")
    p.add_argument("--abort-speed", type=float, default=1.0, help="rev/s")
    p.add_argument("--abort-tilt", type=float, default=30.0, help="deg, body rotation from start")
    p.add_argument("--out", default=f"/tmp/wheel_swing-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    p.add_argument("--dry-run", action="store_true")
    sys.exit(asyncio.run(main(p.parse_args())))
