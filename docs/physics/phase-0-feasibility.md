# Phase 0 feasibility check

The Phase 0 gate answers one question: does a non-empty, robust hardware design
window exist before parts are ordered and detailed mechanics are designed? It is
an analytical check plus a constrained swing-up simulation. It does not replace
the full honest simulation required by Phase 2 (sensor model, friction, cogging,
estimator).

## Gate status

**Torque budget: DONE. Driver decision: moteus r4.11. Gate overall: NOT READY.**

Done on 2026-09-04:

- peak, continuous and momentum budgets computed with the moteus firmware
  conventions and the measured mj5208 constants;
- swing-up simulated with the real torque-speed envelope; required momentum
  known;
- battery placement decided (pivot axis);
- driver decided: **moteus r4.11**.

Still open before the gate is passed:

- component-level mass, placement and inertia from CAD (all mechanical values
  below are planning ranges);
- bench confirmation of the mj5208 thermal model (stall current vs. winding
  temperature);
- cogging torque before and after moteus anti-cogging;
- sensor/estimator selection and a repeat of the sampled-LQR sweep with it.

Reproduce from `software/sim`:

```shell
poetry run phase0                                # analytical report
poetry run phase0-swingup --sweep                # momentum gate, hardware peak torque
poetry run phase0-swingup --sweep --torque-limit 0.5   # momentum gate, torque-capped pumping
poetry run phase0-control --pessimistic --sweep  # sensor timing sweep
poetry run python -m unittest discover -s tests
```

Inputs live in [`software/sim/phase0.toml`](../../software/sim/phase0.toml)
with provenance per block.

## Design decisions feeding this check (2026-09-04)

| Decision | Value | Consequence |
|---|---|---|
| Swing-up | required | momentum gate is mandatory |
| Disturbance to survive | finger poke, target 10°, preferred 20° | larger pokes are allowed to fall, flip and swing back up |
| Mechanics | full redesign allowed | mass model stays parametric until CAD |
| Battery | 4S LiPo on the moving part | bus 12 V (loaded, near empty) to 16.8 V; 14.8 V nominal |

## Inputs and provenance

Mechanical nominal values reproduce the estimates used in the
[motor selection criteria](../hardware/hardware.md#motor-selection-criteria):
total moving mass 0.55 kg, centre of mass 0.18 m. Ranges and inertias are
planning assumptions, not CAD.

Motor and controller constants (verified 2026-09-04):

| Quantity | Value | Source |
|---|---|---|
| mj5208 Kv (nominal) | 330 rpm/V | [mjbots product page](https://mjbots.com/products/mj5208) |
| mj5208 Kv (measured devkit motor) | 304 rpm/V | [mjbots blog 2025-04-24](https://blog.mjbots.com/2025/04/24/improving-motor-constant-calibration-in-moteus/) |
| Phase resistance, line-to-centre | 0.047 Ω | same blog |
| Phase inductance | 28.6 µH | same blog |
| Poles | 14 | moteus documentation example config for mj5208 |
| Motor peak torque / power / speed / mass | 1.7 Nm / 600 W / 7500 rpm / 193 g | product page |
| moteus-c1 | 20 A peak, 5 A (14 A cooled) continuous, 10–51 V, 8.9 g, $69 | [product page](https://mjbots.com/products/moteus-c1) |
| moteus r4.11 | 100 A peak, 12 A (32 A cooled) continuous, 10–44 V, 14.2 g, 46×53 mm, $94 | [product page](https://mjbots.com/products/moteus-r4-11) |

### Torque constant: moteus convention

The moteus firmware (`fw/bldc_servo.cc`) computes reported and limited torque as

$$
K_t = \frac{3}{2}\cdot\frac{1}{\sqrt{3}}\cdot\frac{60}{2\pi}\cdot\frac{1}{K_v} \approx \frac{8.27}{K_v}
$$

with $K_v$ measured as peak-to-peak line-to-line voltage per rpm. For the mj5208
this gives **0.0251 Nm/A** (Kv 330) or 0.0272 Nm/A (measured Kv 304). The
earlier estimate 60/(2π·Kv) = 0.029 Nm/A overstated torque by 15 %; all numbers
below use 0.0251 (conservative). Consequences:

- c1 peak: 20 A → **0.50 Nm** (was 0.58);
- r4.11 peak: motor-limited at **1.7 Nm**, reached at 62–68 A.

### Torque-speed envelope

moteus limits the phase voltage to $V_{eff} = 0.5\,V_{bus}\,r_{max}(1 - m)$ with
modulation margin $m = 0.15$ and $r_{max} \approx 0.94$ (unsure: depends on
PWM rate), and the q-axis current at rotor frequency $f$ must satisfy the
voltage circle

$$
(v_{per\,Hz} f + R\,i_q)^2 + (\pi\,p\,L\,i_q\,f)^2 \le V_{eff}^2,
\qquad v_{per\,Hz} = \frac{60}{\sqrt{3}\,K_v}.
$$

This is the same quadratic the firmware solves for its base velocity. Results
for the mj5208 (Kv 330):

| Bus | No-load speed | Knee, c1 (0.50 Nm) | Knee, r4.11 (1.7 Nm) |
|---:|---:|---:|---:|
| 12.0 V | 2740 rpm | 2150 rpm | 830 rpm |
| 14.8 V | 3380 rpm | 2780 rpm | 1370 rpm |
| 16.8 V | 3840 rpm | 3220 rpm | 1750 rpm |

The hobby rule $\omega = K_v V_{bus}$ (3960 rpm at 12 V) overstates the usable
speed by ~45 %; the 2026-08-10 log figure of 4900 rpm at 4S is not reachable
under moteus. Momentum capacity is computed with the 2740 rpm figure.

## Mechanical calculations

For component-level CAD values, record mass $m_i$, centre-of-mass distance $r_i$
and own inertia $I_{i,cm}$ per part, then

$$
G = \sum_i m_i g r_i, \qquad I_p = \sum_i (I_{i,cm} + m_i r_i^2),
\qquad \tau_g(\theta) = G\sin\theta, \qquad \Delta E_{swing\text{-}up} = 2G.
$$

Nominal: $G = 0.971$ Nm, 0.169 Nm at 10°, 0.332 Nm at 20°, 1.94 J.
Pessimistic corner (0.65 kg, 0.21 m): $G = 1.339$ Nm, 0.233 Nm at 10°,
0.458 Nm at 20°, 2.68 J.

### Battery placement

A 0.12 kg pack (4S, ~1000 mAh; mass unsure ±20 %) on the nominal design:

| Placement | G | τ_g at 20° | Swing-up energy | I_p | t_c |
|---|---:|---:|---:|---:|---:|
| on the pivot axis | 0.971 Nm | 0.332 Nm | 1.94 J | 0.0250 kg m² | 160 ms |
| at the top end (0.25 m) | 1.265 Nm | 0.433 Nm | 2.53 J | 0.0325 kg m² | 160 ms |

At the top end the pack raises every torque and energy requirement by 30 % and
buys nothing: the time constant is unchanged because for a point mass
$g/r \approx G/I_p$ at this geometry. **Decision: battery on the pivot axis.**
Mounting it slightly *below* the pivot turns it into a counterweight and lowers
$G$ further; that is a mechanical option for Phase 1, not a requirement.

## Peak-torque gate

Requirement: $\tau_{peak} \ge k\,G\sin\theta$, $k = 1.5$, at the pessimistic
gravity corner.

| Actuator | Peak | Margin at 10° (target) | Margin at 20° (preferred) | Safe angle at 1.5× | Result |
|---|---:|---:|---:|---:|:---:|
| mj5208 + c1 | 0.50 Nm | 2.16× | 1.09× | 14.4° | PASS target, FAIL preferred |
| mj5208 + r4.11 | 1.70 Nm | 7.31× | 3.71× | 57.8° | PASS both |

## Continuous (thermal) gate

Requirement: ≥ 0.20 Nm sustained at stall without airflow (balance hold with
friction, cogging and wheel-speed regulation).

| Actuator | Driver proxy $K_t I_{cont}$ | Motor copper-loss model | Result |
|---|---:|---:|:---:|
| mj5208 + c1 (bare) | 0.125 Nm | limited by driver | FAIL |
| mj5208 + r4.11 (bare) | 0.301 Nm | 0.30–0.42 Nm | PASS (model) |

The motor model is $P = 1.5\,I^2 R$ with $R = 0.047\ \Omega$, a 60 K allowed
winding rise and an assumed winding-to-ambient resistance of 3–6 K/W; it gives
a continuous current of 12–17 A. That is a model, not a measurement, and the
bench test (stall current vs. winding temperature with the real wheel mounted
as a heat sink) is a Phase 1 task. The r4.11's 12 A bare rating happens to sit
at the low end of the same range, so the driver and motor limits are matched.

## Flywheel momentum gate (swing-up)

`phase0-swingup` simulates the reaction-wheel pendulum from hanging with
energy pumping (torque sign follows the pendulum rate, scaled by the energy
error) and hands over to the sampled LQR inside a 30° cone when the energy is
within 5 % of the upright value. Torque is limited by the envelope above at the
current wheel speed. A run passes when the pendulum is caught, settles to
≤ 2° and ≤ 100 rpm, and the wheel never exceeds 70 % of no-load speed (so torque
is still available for the catch).

Key result: the required momentum $H = I_w\,\omega_{w,max}$ is a torque
integral and therefore **independent of the wheel inertia**; the wheel inertia
only sets how fast the wheel spins to store it.

| Strategy | Design | Swings | Time | H required | Max wheel rpm at I_w = 0.0015 / 0.003 / 0.0045 kg m² (12 V) |
|---|---|---:|---:|---:|---|
| Torque-capped pumping ≤ 0.5 Nm (c1, or r4.11 with software cap) | nominal | 1 | 1.1 s | 0.28 Nms | 1800 (0.66) / 900 (0.33) / 600 (0.22) |
| same | pessimistic | 2 | 1.3 s | 0.19 Nms | 1220 (0.45) / 610 (0.22) / 410 (0.15) |
| Direct lift at 1.7 Nm (r4.11, no cap) | nominal | 0 | 0.4 s | 0.44 Nms | 2620 (0.96, FAIL) / 1410 (0.51) / 910 (0.33) |
| same | pessimistic | 0 | 0.3 s | 0.45 Nms | 2840 (0.91, FAIL) / 1410 (0.51) / 910 (0.33) |

RMS phase current during swing-up is 5–11 A for about a second, thermally
negligible. Design window from this gate:

$$
I_w \ge \frac{H}{0.7\,\omega_{nl}} =
\begin{cases}
0.0014\ \mathrm{kg\,m^2} & \text{torque-capped pumping} \\
0.0022\ \mathrm{kg\,m^2} & \text{direct lift at 1.7 Nm}
\end{cases}
\quad\text{at 12 V.}
$$

Recommendation: design for **I_w ≈ 0.003 kg m²** (both strategies pass at 51 %
or less of no-load speed at the empty-battery voltage). The 2025 wheel had
0.0008 kg m² and would not pass.

Caveats: no friction, no cogging, no sensor noise in the swing-up run; the
energy-pumping gain and catch cone are hand-tuned; the pendulum model treats the
wheel as a point mass in $I_p$.

## Preliminary sensor timing budget

Unchanged from the first calculation. Linearized growth rate
$\lambda = \sqrt{G/I_p}$ gives a time constant of 160 ms nominal, 116 ms at the
fastest corner. Allocating ≤ 10 % to total loop delay and ≤ 5 % to one sample
period gives **delay ≤ 12 ms and sample rate ≥ 173 Hz**. The sampled nonlinear
check (`phase0-control --pessimistic --sweep`, r4.11) fails at 50 Hz and passes
at 75 Hz; fails at 15 ms delay and passes at 10 ms; fails at 2° angle noise with
20 °/s rate noise and passes at 1° / 10 °/s. These must be repeated with the
selected sensor and estimator.

## Driver decision: moteus r4.11

| Gate | c1 | r4.11 |
|---|:---:|:---:|
| Peak, finger poke 10° | PASS 2.2× | PASS 7.3× |
| Peak, preferred 20° | FAIL 1.1× | PASS 3.7× |
| Continuous ≥ 0.2 Nm, bare board | FAIL 0.125 Nm | PASS 0.30 Nm |
| Swing-up momentum | PASS (pumping only) | PASS (pumping or direct lift) |
| Bus 4S | ok | ok |
| Mass / size | 8.9 g, 38×38 mm | 14.2 g, 46×53 mm |
| Price | $69 | $94 |

The c1 meets the minimum requirement only, and only with a cooled board for
continuous torque. The r4.11 removes the driver as a constraint for $25 and
5 g; the remaining limits are the motor's thermal capacity and the mechanics.
**Decision 2026-09-04: mj5208 + moteus r4.11.**

## Gate exit conditions

- Replace the planning ranges in `phase0.toml` with a component-level mass
  model from CAD (motor 193 g, r4.11 14.2 g, battery ~120 g on the axis, rod,
  wheel of I_w ≈ 0.003 kg m²).
- Bench: mj5208 stall current vs. winding temperature; cogging before/after
  anti-cogging; torque constant via lever-and-scale.
- Select the angle sensor/estimator and repeat the sampled-LQR sweep with its
  noise, quantization and latency.
- Confirm at least one coherent configuration passes every gate with margin.

Peak, continuous and momentum budgets pass for the r4.11 across the whole
planning range, so ordering the motor and driver is no longer blocked by the
torque budget.
