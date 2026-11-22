"""Read-only link check: query the moteus and the pi3hat IMU over the pi3hat.

Runs on the Raspberry Pi with the pi3hat; needs root (the pi3hat library
accesses SPI/GPIO registers directly). Commands no motion.

Each step runs two pi3hat cycles: one moteus query, then one IMU attitude
read. They are separate because moteus-pi3hat 1.1.0 (with moteus 1.1.3) raises
AttributeError ('CanAttitudeWrapper' has no 'channel') when a moteus query and
request_attitude=True share one cycle. Prints the moteus state and IMU angles
for the first and last step, then the step-time statistics. Exits with code 1
if the moteus misses any reply.

Usage (on the Pi, from a venv with `moteus` and `moteus-pi3hat`):
  sudo .venv/bin/python tools/pi3hat_ping.py [--bus 1] [--id 1] [--count 100]
"""
import argparse, asyncio, math, statistics, sys, time

import moteus
import moteus_pi3hat


def format_moteus(result):
    v = result.values
    R = moteus.Register
    return (f"mode {v[R.MODE]}, position {v[R.POSITION]:.4f} rev, "
            f"velocity {v[R.VELOCITY]:.4f} rev/s, voltage {v[R.VOLTAGE]:.1f} V, "
            f"temperature {v[R.TEMPERATURE]:.0f} C, fault {v[R.FAULT]}")


def format_imu(attitude):
    e = attitude.euler_rad
    return (f"roll {math.degrees(e.roll):.1f} deg, pitch {math.degrees(e.pitch):.1f} deg, "
            f"rate x {attitude.rate_dps.x:.2f} dps, y {attitude.rate_dps.y:.2f} dps")


async def one_step(transport, controller, timeout_s):
    results = await asyncio.wait_for(
        transport.cycle([controller.make_query()]), timeout_s)
    servo = next((r for r in results if getattr(r, "id", None) == controller.id), None)
    results = await asyncio.wait_for(
        transport.cycle([], request_attitude=True), timeout_s)
    imu = next((r for r in results if isinstance(r, moteus_pi3hat.CanAttitudeWrapper)), None)
    return servo, imu


async def main(args):
    transport = moteus_pi3hat.Pi3HatRouter(servo_bus_map={args.bus: [args.id]})
    controller = moteus.Controller(id=args.id, transport=transport)
    print(f"pi3hat bus {args.bus} (JC{args.bus}), moteus id {args.id}, {args.count} steps")

    durations, missing, first, last = [], 0, None, None
    for _ in range(args.count):
        start = time.perf_counter()
        try:
            servo, imu = await one_step(transport, controller, args.timeout)
        except asyncio.TimeoutError:
            servo, imu = None, None
        durations.append(time.perf_counter() - start)
        if servo is None:
            missing += 1
        else:
            last = (servo, imu)
            first = first or last

    if first is None:
        print(f"no reply from moteus id {args.id} on bus {args.bus}; "
              "check the cable port (JC1-JC5), controller power and CAN ID")
        return 1

    for label, (servo, imu) in (("first", first), ("last", last)):
        print(f"{label}: moteus {format_moteus(servo)}")
        print(f"{label}: imu    {format_imu(imu) if imu else 'no attitude'}")
    ms = sorted(d * 1e3 for d in durations)
    print(f"step time ms (query + attitude): min {ms[0]:.2f}, median {statistics.median(ms):.2f}, "
          f"max {ms[-1]:.2f}; missing moteus replies {missing}/{args.count}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bus", type=int, default=1, help="pi3hat CAN bus, 1-5 (= JC1-JC5)")
    p.add_argument("--id", type=int, default=1, help="moteus CAN ID")
    p.add_argument("--count", type=int, default=100, help="number of steps")
    p.add_argument("--timeout", type=float, default=0.5, help="per-cycle timeout, s")
    sys.exit(asyncio.run(main(p.parse_args())))
