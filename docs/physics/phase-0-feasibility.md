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

Done on 2026-11-09 (PNDLM-14): the motor and controller inputs of
`phase0.toml` are measured values (PNDLM-8, 9, 11, 12) or a bounded
assumption (PNDLM-13); see [Measured inputs](#measured-inputs-pndlm-14).

Still open before the gate is passed:

- weighed parts: the CAD mass model is in place, RPi/pi3hat and axle hardware
  are still catalogue estimates;
- the motor thermal limit stays an assumption (PNDLM-13, no sensor);
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

### Measured inputs (PNDLM-14)

**Update 2026-11-09.** The motor and controller block of
[`phase0.toml`](../../software/sim/phase0.toml) now uses the bench results.
Raw runs stay in the local `logs/` of `hardware/firmware/moteus-host`; the
derived records are committed in `docs/hardware/measurements/`.

| Input | Value in `phase0.toml` | Uncertainty | Source | Replaces |
|---|---|---|---|---|
| K_t | 0.0244 Nm/A | ±3 % (1 sd); measured 1–8 A only | PNDLM-9, 7 runs 20261026T095749Z…175936Z, [torque-constant.json](../hardware/measurements/2026-10-26-torque-constant.json) | 8.27/Kv = 0.0251 (Kv 330) |
| Kv | 339 rpm/V, derived as 8.27/K_t | as K_t | PNDLM-12 fits every no-load and base speed with it | 330 (product page); firmware `motor.Kv` 296 is not physical |
| r_max (`max_voltage_ratio`) | 1.00 | ±0.03 | PNDLM-12, 5 runs 20261104T111019Z…113421Z, [torque-speed.json](../hardware/measurements/2026-11-04-torque-speed.json) | 0.94 (UNSURE) |
| No-load speed | 248.3 rpm/V of loaded bus (result, not an input) | ±1.0 rpm/V | PNDLM-12, 24 steps, intercept 6 ± 12 rpm | 228 rpm/V |
| R, line-to-centre | 0.0467 Ω | — | controller calibration dump, PNDLM-12 runs | 0.047 Ω (mjbots blog) |
| L_q | 44.8 µH | — | same dump | 28.6 µH (mjbots blog) |
| Rotating losses | a 4.43 mNm, b 37.8 µNm·s, c 9.7 nNm·s²; a 3–7 mNm for the assembly | a ±0.07, b ±1.4 (se) | PNDLM-11, [rotating-losses.json](../hardware/measurements/2026-11-01-rotating-losses.json) | none (no losses) |
| Rotor + hub inertia | +0.000024 kg m² added to `flywheel_inertia_kg_m2` | ±0.00001 | PNDLM-11, same record | not counted |
| Actuator delay | sweep default 5 ms covers it | ≤ 4.4 ms rise, laptop + fdcanusb | PNDLM-8, PNDLM-9 (see [timing](#preliminary-sensor-timing-budget)) | — |
| Thermal limit | R_th 3 / 4.5 / 6 K/W, rise ≤ 60 K, hot R +0.393 %/K, hot K_t −0.12 %/K | ASSUMPTION, break-even 8.9 K/W | PNDLM-13, [continuous gate](#continuous-thermal-gate) | R_th model without hot R and hot K_t |

Not measured, and kept as named inputs:

- motor peak torque 1.7 Nm, 7500 rpm (product page): K_t above 8 A is an
  extrapolation, 1.7 Nm needs about 70 A;
- thermal resistance, thermal time constant and magnet grade (PNDLM-13);
- K_t at speed: the PNDLM-11 plateau cross-check leaves a 1.2 mNm residual at
  2500–3000 rpm, K_t was measured on a held rotor;
- breakaway friction below ~1 rev/s (PNDLM-11);
- cogging torque before and after anti-cogging;
- the loop delay of the final RPi + pi3hat host (only the laptop + fdcanusb
  path is measured);
- weighed part masses, battery mass.

Effect on the checks (re-run 2026-11-09, `poetry run phase0`,
`phase0-swingup --sweep`, `phase0-control --pessimistic --sweep`):

- peak: c1 0.488 Nm (was 0.501); r4.11 stays motor-limited at 1.7 Nm, margins
  7.28× / 3.70× unchanged;
- continuous, r4.11: driver proxy 0.293 Nm (was 0.301); motor model with hot R
  and hot K_t 0.24 / 0.28 / 0.34 Nm (was 0.30 / 0.34 / 0.42); PASS
  (assumption);
- no-load speed at 12 V: 2994 rpm (was 2740), so the wheel stores 9 % more
  momentum; r4.11 knee 856 rpm at 12 V, lower than before at 14.8 V and
  16.8 V because L is 57 % higher;
- swing-up: torque-capped pumping passes 24/24; uncapped direct lift fails 1
  of 24 rows (was 3): pessimistic design, lightest wheel 0.0024 kg m², 12 V,
  wheel at 0.72 of no-load;
- sampled LQR: unchanged, passes 50–1000 Hz, 15 ms delay, 2° / 20 °/s noise;
  fails at 20 ms.

The tables below keep their 2026-09-04 planning values unless an update note
says otherwise.

### Planning constants (2026-09-04)

**Update 2026-09-04 (evening):** the `[design]` block of `phase0.toml` now comes
from the OpenSCAD mass model (`hardware/cad`, `make masses`): m_t 0.776 kg,
l_c 0.146 m, G 1.11 Nm nominal (1.345 Nm pessimistic), I_p 0.029 kg m²,
I_w 0.0027 kg m² with 8 tuning bolts. The tables below were computed with the
earlier planning ranges (0.55 kg, 0.18 m, G 0.971 nominal); re-run
`poetry run phase0` for current numbers. Conclusions are unchanged: the r4.11
passes peak (7.3× / 3.7×), continuous and torque-capped swing-up (24/24 rows).
One new caveat: the *uncapped* direct lift at 12 V drives the wheel to 71–79 %
of no-load speed at I_w 0.0024–0.0027, above the 70 % limit; fit 10 bolts
(I_w 0.0030) or cap swing-up torque at 0.5 Nm in software.

The mechanical nominal values used in the tables below reproduce the estimates
from the [motor selection criteria](../hardware/hardware.md#motor-selection-criteria):
total moving mass 0.55 kg, centre of mass 0.18 m.

Motor and controller constants (verified 2026-09-04, superseded by
[Measured inputs](#measured-inputs-pndlm-14) for Kv, K_t, R, L and r_max):

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

**Update 2026-11-09 (PNDLM-14).** With the measured K_t 0.0244 (Kv 339),
r_max 1.00, R 0.0467 Ω and L_q 44.8 µH:

| Bus | No-load speed | Knee, c1 (0.488 Nm) | Knee, r4.11 (1.7 Nm) |
|---:|---:|---:|---:|
| 12.0 V | 2994 rpm | 2311 rpm | 856 rpm |
| 14.8 V | 3693 rpm | 2964 rpm | 1327 rpm |
| 16.8 V | 4192 rpm | 3430 rpm | 1650 rpm |

The no-load column matches the measured 248.3 rpm/V. The hobby rule (Kv 339:
4067 rpm at 12 V) still overstates it by 36 %. The knee at 1.7 Nm is an
extrapolation: K_t is measured to 8 A only.

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
| mj5208 + c1 | 0.488 Nm | 2.09× | 1.06× | 14.0° | PASS target, FAIL preferred |
| mj5208 + r4.11 | 1.70 Nm | 7.28× | 3.70× | 57.4° | PASS both |

Values from the 2026-11-09 re-run (measured K_t, CAD mass model). The r4.11 is
limited by the 1.7 Nm motor rating, not by K_t × 100 A, so K_t does not change
its margins. If the 1.7 Nm rating itself scales with K_t, the hot motor
(−7 %) gives about 6.8× and 3.4×.

## Continuous (thermal) gate

Requirement: ≥ 0.20 Nm sustained at stall without airflow (balance hold with
friction, cogging and wheel-speed regulation).

| Actuator | Driver proxy $K_t I_{cont}$ | Motor copper-loss model | Result |
|---|---:|---:|:---:|
| mj5208 + c1 (bare) | 0.122 Nm | limited by driver | FAIL |
| mj5208 + r4.11 (bare) | 0.293 Nm | 0.24–0.34 Nm | PASS (assumption) |

Values from the 2026-11-09 re-run: measured K_t; the motor model uses hot R
and hot K_t (update below). The text under this table is the 2026-09-04
version.

The motor model is $P = 1.5\,I^2 R$ with $R = 0.047\ \Omega$, a 60 K allowed
winding rise and an assumed winding-to-ambient resistance of 3–6 K/W; it gives
a continuous current of 12–17 A. That is a model, not a measurement, and the
bench test (stall current vs. winding temperature with the real wheel mounted
as a heat sink) is a Phase 1 task. The r4.11's 12 A bare rating happens to sit
at the low end of the same range, so the driver and motor limits are matched.

**Update 2026-11-09 (PNDLM-13, assumption instead of a measurement).** The mj5208
has no thermistor, so the thermal bench test was replaced by this assumption:

- R_th = 3–6 K/W winding to ambient, winding rise ≤ 60 K as a hard limit (magnet
  grade unknown; N-grade NdFeB loses field irreversibly above ~80 °C).
- Hot corner at +60 K: R +23.6 % (copper 0.393 %/K), K_t −6…−7 % and Kv +6…+8 %
  (NdFeB −0.10…−0.12 %/K, magnets assumed as hot as the winding, worst case).
- With the copper temperature coefficient the continuous current is 10.7–15.2 A
  (not 12–17 A); with hot K_t 0.0227 Nm/A that is 0.24–0.35 Nm against the
  0.20 Nm requirement.
- Break-even: 0.20 Nm needs 8.8 A hot, 6.8 W of copper loss, so the gate fails
  only above R_th ≈ 8.9 K/W, 1.5× the worst assumed value.
- Thermal time constant ≈ 5–10 min (estimate, C ≈ 100 J/K); swing-up and
  poke recovery are thermally negligible.

Numbers: `software/sim/app/plot_thermal.py`; story and figure:
[thermal-assumption article](../articles/2026-11-09-thermal-assumption-draft.md).

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

**Update 2026-11-09 (PNDLM-14).** Re-run with the CAD design range (I_w 0.0024 /
0.0027 / 0.0029 kg m², rotor included) and the measured motor inputs:

- torque-capped pumping ≤ 0.5 Nm: 24/24 PASS, wheel at 0.20–0.36 of no-load;
- direct lift at 1.7 Nm (r4.11, no cap): 23/24 PASS; the one FAIL is the
  pessimistic design with the lightest wheel (0.0024 kg m²) at 12 V, wheel at
  0.72 of no-load. The nominal wheel 0.0027 kg m² passes at 0.61–0.65.

The earlier advice stands: fit 10 bolts (I_w ≈ 0.003) or cap swing-up torque
at 0.5 Nm in software.

## Preliminary sensor timing budget

Re-run 2026-11-08 with the CAD mass model (`poetry run phase0`). Linearized
growth rate $\lambda = \sqrt{G/I_p}$ gives a time constant of 161 ms nominal
(G 1.111 Nm, I_p 0.0289 kg m²) and 139 ms at the fastest corner
(G 1.345 Nm). Allocating ≤ 10 % of the fastest time constant to total loop
delay and ≤ 5 % to one sample period gives **delay ≤ 13.9 ms and sample rate
≥ 144 Hz**. The earlier 116 ms / 12 ms / 173 Hz came from the planning mass
ranges.

The sampled nonlinear check (`phase0-control --pessimistic --sweep`, r4.11,
5 trials per point, re-run 2026-11-08) passes at every sample rate from 50 to
1000 Hz; passes at 15 ms delay and fails at 20 ms (0/5); passes at every angle
noise up to 2° with 20 °/s rate noise. Earlier runs on the planning ranges
failed at 50 Hz and at 2° / 20 °/s; the delay limit is unchanged. The LQR gain
is designed on the continuous model without sampling or delay, so the 20 ms
limit belongs to this controller, not to the plant. These must be repeated
with the selected sensor and estimator.

Measured actuator delay on the laptop + fdcanusb host: ≤ 13 ms detection
bound ([PNDLM-8](../hardware/desk-stand-torque-pulse-timing.md)), refined to
≤ 4.4 ms by the torque rise test
([PNDLM-9](../hardware/specs/motor-torque-fixture.md)). The 5 ms default
command delay of `phase0-control` covers it; the 15 ms pass point leaves
about 10 ms for the sensor and estimator. The RPi + pi3hat path is not
measured.

## Driver decision: moteus r4.11

| Gate | c1 | r4.11 |
|---|:---:|:---:|
| Peak, finger poke 10° | PASS 2.1× | PASS 7.3× |
| Peak, preferred 20° | FAIL 1.06× | PASS 3.7× |
| Continuous ≥ 0.2 Nm, bare board | FAIL 0.122 Nm | PASS 0.29 Nm |
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
- Bench: cogging before/after anti-cogging. Torque constant measured
  (PNDLM-9); the thermal limit is closed by assumption (PNDLM-13).
- Select the angle sensor/estimator and repeat the sampled-LQR sweep with its
  noise, quantization and latency.
- Confirm at least one coherent configuration passes every gate with margin.

Peak, continuous and momentum budgets pass for the r4.11 across the whole
planning range, so ordering the motor and driver is no longer blocked by the
torque budget.
