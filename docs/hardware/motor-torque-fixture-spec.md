# MJ5208 static torque fixture

## 1. Purpose

This fixture independently measures the mj5208 torque constant for T-005.C. A
rigid lever attached to the rotor presses vertically on a compression scale
while moteus commands short Q-axis current plateaus. The scale is the force
reference; moteus-reported torque is recorded only for comparison.

The first test is deliberately limited to ±8 A and approximately ±0.20 Nm. It
does not characterize peak torque, continuous thermal torque or torque at speed.

## 2. Measurement principle

```text
                         motor axis
                             O
                             │
    left contact ●───────────┼───────────● right contact
                 <--- 100.0 mm --->
                                                ↓ force
                                          ┌───────────┐
                                          │   scale   │
                                          └───────────┘
```

Use a balanced crossbar with a contact point on each side. Put the scale under
the contact loaded by the commanded torque direction; move it to the other side
for the opposite direction. The force must be perpendicular to the lever.

For effective radius $r$ and zero-corrected scale reading $m$:

$$
\tau = m g r
$$

With $r = 0.1000$ m and the reading expressed in grams:

$$
\tau[\mathrm{N\,m}] = m[\mathrm{g}] \times 0.000980665
$$

The fitted model is:

$$
\tau = K_t I_q + b
$$

where $I_q$ is measured Q-axis current, not merely its requested value. The
slope is the independently measured torque constant. The intercept contains
scale preload, zero error, residual cogging and fixture bias.

## 3. Mechanical design

### 3.1 Lever

- balanced metal crossbar, nominally 200 mm between contact centerlines;
- effective radius from the motor axis to each force line: **100.0 mm**;
- measure each radius after assembly to ±0.2 mm or better;
- aluminium flat bar is preferred; a printed lever is not the measurement
  reference because creep and bending add avoidable error;
- rigid hub using the mj5208 rotor M3 pattern, with no angular play;
- an adjustable rounded metal contact at each end, such as an M4 screw with a
  ball or acorn end;
- bar horizontal at the measurement point, so the vertical scale force is
  perpendicular to the radius.

The effective radius is the perpendicular distance from the motor axis to the
vertical force line. It is not automatically the physical bar length if the bar
or contact is angled.

### 3.2 Load path

- bolt the motor stator bracket and scale base to the same stiff baseplate;
- do not hold the motor, lever or scale by hand;
- constrain the scale so it cannot slide sideways;
- include a secondary hard stop just beyond normal scale deflection;
- add a loose guard or restraint that catches the lever if contact is lost;
- leave the unloaded end of the crossbar clear.

The scale is a measurement instrument, not the only structural restraint.

### 3.3 Scale

The available compression scale is rated to 3 kg with 0.1 g display resolution.
At 100 mm its nominal torque increment is 0.0000981 Nm. This is more than
sufficient; calibration accuracy, repeatability, force alignment and lever
radius dominate the uncertainty.

Check the scale before use with at least two known masses spanning roughly
0.2–1.0 kg. Record displayed resolution, repeatability, auto-zero behavior,
auto-off behavior and whether the reading is stable under a constant load.

## 4. Range

| Scale reading | Torque at 100.0 mm |
|---:|---:|
| 10 g | 0.00981 Nm |
| 100 g | 0.0981 Nm |
| 200 g | 0.196 Nm |
| 500 g | 0.490 Nm |
| 1000 g | 0.981 Nm |
| 1733 g | 1.70 Nm |

The scale can nominally contain the published 1.7 Nm motor peak at this radius,
but scale capacity alone does not qualify the fixture for a peak-torque test.
The hub, bracket, baseplate, contact, restraint, supply and thermal limits would
all require a separate Tier-C review.

## 5. Initial test protocol

### 5.1 Preconditions

- crossbar and hub inspected and fasteners marked against loosening;
- motor stator, scale and hard stop fixed to the common baseplate;
- contact radius measured and recorded for each side;
- scale checked with known masses, zeroed and prevented from auto-off;
- no hands or loose objects in the lever envelope;
- physical power disconnect immediately accessible;
- controller stopped, fault 0, bus voltage and temperature in limits;
- software watchdog and unconditional `set_stop()` path verified with a fake
  controller before the physical run.

### 5.2 Current plateaus

Use commanded Q-current plateaus of:

```text
0, 1, 2, 4, 6, 8 A
```

Run positive and negative directions separately, placing the scale beneath the
appropriate compression contact. Use measured `q_current_a` in analysis. Start
with 2–3 second plateaus and stop between plateaus. Repeat every nonzero point
three times. Abort on motion, loss of contact, unstable load, unexpected noise,
fault, voltage violation, current violation or temperature violation.

At the planning value $K_t \approx 0.025$ Nm/A, 8 A should produce about 0.20
Nm or 204 g at 100 mm. Do not raise the current range merely because the scale
has unused capacity.

### 5.3 Manual scale readings

The first version uses manual readings. The host must continue refreshing the
watchdog while current is active; a blocking terminal prompt must never pause
motor commands. Prefer this sequence:

1. run a fixed-duration plateau while recording moteus telemetry;
2. give the operator an audible/terminal timing cue;
3. stop the controller;
4. enter the observed stable or held scale reading in grams;
5. record side, current sign, radius, repetition and operator note in
   `events.csv`.

A camera recording of the scale display is useful evidence when the scale lacks
a data interface.

## 6. Planned software

`poetry run bench torque-constant` will reuse the recorder and safety layer. It
must:

- require a fixture-specific confirmation phrase;
- command current directly rather than request torque from the configured motor
  model;
- keep D-axis current at zero;
- refresh a 100 ms watchdog throughout every plateau;
- enforce the tracked 8 A initial phase-current limit;
- record requested and measured Q/D current, position, velocity, reported
  torque, bus voltage, power, controller temperature, fault and timing;
- stop between plateaus and on normal exit, fault, timeout or `Ctrl-C`;
- record manually entered scale readings and the measured contact radius;
- refuse higher-current profiles until the fixture receives a separate review.

Offline analysis will calculate torque for every reading, fit $K_t$ and an
intercept for each direction, compare slopes, report residuals and uncertainty,
and compare physical torque with moteus-reported torque.

## 7. Uncertainty and acceptance

At minimum, propagate:

- scale calibration and repeatability;
- scale resolution;
- left/right effective-radius measurement;
- lever non-perpendicularity, using the measured sine correction if needed;
- contact-position repeatability;
- measured-current variation during the reading window;
- regression residuals and positive/negative slope disagreement.

Accept the initial result when:

- force is monotonic with measured current in both directions;
- repeated readings at each plateau agree within a declared tolerance;
- positive and negative fitted $|K_t|$ values agree within 5%, or the asymmetry
  is investigated;
- the fit includes at least four nonzero current magnitudes per direction;
- no fixture motion, controller fault or safety-limit violation occurred;
- raw telemetry, events, controller configuration, dimensions and scale checks
  are preserved with the run ID.

## 8. Safety boundary

The initial fixture is approved only for short plateaus up to 8 A. It is not yet
approved for the published 1.7 Nm peak, continuous stall/thermal testing or
rotating torque-speed work. Those tests require a reviewed restraint, motor
temperature instrumentation and, for rotating tests, a retained balanced rotor
and guard.
