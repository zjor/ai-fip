# Phase 0 feasibility check

The Phase 0 gate answers one question: does a non-empty, robust hardware design
window exist before parts are ordered and detailed mechanics are designed? It is
an analytical impossibility check. It does not replace the constrained nonlinear
simulation required by Phase 2.

## Gate status

**NOT READY.** The initial calculation is reproducible, but several decisive
inputs are still planning ranges or unknown:

- component-level mass, placement, and inertia;
- mj5208 continuous torque at zero speed, limited by motor temperature rather
  than only controller current;
- cogging torque before and after moteus compensation;
- required flywheel momentum for swing-up;
- closed-loop tolerance to sensor noise, sample rate, and latency.

The calculation lives in [`software/sim/phase0.toml`](../../software/sim/phase0.toml)
and can be run from `software/sim`:

```shell
poetry run phase0
poetry run phase0-control --pessimistic --sweep
poetry run python -m unittest discover -s tests
```

## Inputs and provenance

The current mechanical nominal values reproduce the estimates already used in
the [motor selection criteria](../hardware/hardware.md#motor-selection-criteria):
total moving mass 0.55 kg and center of mass 0.18 m from the pivot. The range
endpoints and the pivot/flywheel inertias are explicitly planning assumptions;
they have not been obtained from CAD or measurements.

The mj5208 values come from the manufacturer: 330 Kv, 1.7 Nm motor peak torque,
7500 rpm maximum speed, and 193 g mass. Controller current limits are taken from
the moteus project specifications: c1 has 20 A peak and 5 A uncooled continuous
phase current; r4.11 has 100 A peak and 12 A uncooled continuous phase current.

Sources:

- [mjbots mj5208 product page](https://mjbots.com/products/mj5208)
- [mjbots moteus-c1 product page](https://mjbots.com/products/moteus-c1)
- [moteus hardware specifications](https://github.com/mjbots/moteus#specifications)

The torque constant is derived from Kv:

$$
K_t = \frac{60}{2\pi K_v} \approx 0.0289\ \mathrm{Nm/A}
$$

This ideal SI conversion must eventually be checked against moteus calibration
and measured output torque. Phase-current conventions and magnetic saturation
can make a naive current-times-$K_t$ estimate inaccurate.

## Mechanical calculations

Replace the aggregate mass estimate with one row per component. For each part,
record mass $m_i$, center-of-mass distance $r_i$, and inertia around its own
center $I_{i,cm}$. Then calculate:

$$
G = \sum_i m_i g r_i
$$

$$
I_p = \sum_i \left(I_{i,cm} + m_i r_i^2\right)
$$

The gravitational torque at angle $\theta$ and the down-to-up potential-energy
change are:

$$
\tau_g(\theta) = G\sin\theta
$$

$$
\Delta E = 2G
$$

At the nominal aggregate estimate, $G=0.971$ Nm and gravity produces 0.332 Nm
at 20 degrees. With the pessimistic independent endpoints, $G=1.339$ Nm and the
20-degree torque is 0.458 Nm.

## Peak-torque gate

For the target recovery angle, require:

$$
\tau_{peak} \ge kG\sin\theta_{target}, \qquad k \ge 1.5
$$

At the initial pessimistic corner:

| Actuator | Estimated peak | Margin at 20 degrees | Safe angle at 1.5x | Result |
|---|---:|---:|---:|:---:|
| mj5208 + c1 | 0.578 Nm | 1.26x | 16.7 degrees | FAIL |
| mj5208 + r4.11 | 1.700 Nm | 3.71x | 57.8 degrees | PASS |

These are analytical current-limit results, not proof of transient or thermal
performance. The c1 result is particularly sensitive to mass and center-of-mass
placement, so the pessimistic endpoints must be replaced by a coherent CAD
configuration rather than relaxed to obtain a pass.

## Thermal gate

The controller's uncooled continuous-current rating gives only a proxy:

$$
\tau_{driver,continuous} = K_t I_{driver,continuous}
$$

This gives approximately 0.145 Nm for c1 and 0.347 Nm for r4.11. It says whether
the controller can carry the current, not whether the mj5208 can dissipate its
$I^2R$ heat at near-zero speed. The thermal gate therefore remains unknown for
both configurations until supported by a manufacturer curve, a thermal model
with winding resistance, or a controlled bench test.

## Flywheel momentum gate

The available angular momentum is:

$$
H_{max} = I_w\omega_{max}
$$

At the conservative 12 V bus assumption, 330 Kv gives a 3960 rpm ideal no-load
speed. The planning flywheel-inertia range of 0.0015--0.0045 kg m2 corresponds to
approximately 0.62--1.87 Nms. This is capacity only; it is not a pass criterion
until the constrained nonlinear swing-up simulation produces $H_{required}$ and
shows that torque remains available along the trajectory.

## Preliminary sensor timing budget

The linearized upright instability has growth rate and characteristic time:

$$
\lambda = \sqrt{\frac{G}{I_p}}, \qquad t_c = \frac{1}{\lambda}
$$

The nominal planning point gives about 160 ms. The fastest independent parameter
corner gives about 116 ms. As preliminary engineering rules, the calculator
allocates no more than 10% of that time to total loop delay and no more than 5%
to one sample period. That suggests:

- total sensing-to-actuation delay no greater than about 12 ms;
- control/sample rate of at least about 173 Hz.

These analytical bounds are supplemented by a provisional sampled nonlinear
stabilization check in `app.phase0_control`. Its state is pendulum angle,
pendulum rate, and absolute wheel speed. It uses continuous LQR gains, sampled
measurements, delayed commands, a conservative linear torque-speed envelope,
torque saturation, and RK4 plant integration. A run passes when a 10-degree
initial displacement remains within 2 degrees and 100 rpm throughout the final
second of a five-second run.

For the pessimistic independent parameter corner and r4.11, five deterministic
noise trials per point currently give:

| Sweep (other values at reference) | Largest failing point | Nearest passing point |
|---|---:|---:|
| Sample rate, 5 ms delay | 50 Hz (0/5) | 75 Hz (5/5) |
| Delay, 500 Hz sample rate | 15 ms (0/5) | 10 ms (5/5) |
| Angle/rate noise standard deviation | 2 deg / 20 deg/s (0/5) | 1 deg / 10 deg/s (5/5) |

The reference scenario is 500 Hz, 5 ms delay, 0.1-degree angle noise, 1 deg/s
rate noise, and 5 rpm wheel-speed noise. The noise sweep changes angle and rate
noise together at a fixed 1:10 ratio; it does not imply that real sensor errors
have that relationship. These boundaries depend on the provisional inertias,
state availability, LQR weights, and simple noise model. They must be repeated
with the selected sensor, estimator, coherent CAD configurations, jitter, and
quantization before the sensor budget passes.

## Gate exit conditions

- Replace aggregate mechanical ranges with component-level CAD/BOM values.
- Calculate coherent nominal and pessimistic builds rather than independent
  combinations that cannot physically coexist.
- Establish mj5208 zero-speed continuous torque and cogging bounds.
- Add a real or conservatively fitted torque-speed envelope.
- Compute required swing-up momentum in the constrained nonlinear model.
- Repeat the provisional sampled-LQR sweep with the selected sensor/estimator,
  coherent CAD cases,
  quantization, and timing jitter.
- Confirm at least one configuration passes every bound with margin.

Only after all items pass should Phase 1 authorize purchasing a motor/driver and
fixing the mechanical geometry.
