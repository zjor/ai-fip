# Measuring flywheel moment of inertia

A wheel's mass alone does not determine how difficult it is to accelerate.
Mass near the rim contributes more rotational inertia than mass near the axis:

\[
I = \int r^2\,dm.
\]

For the rotating-loss and acceleration experiments in
[PNDLM-11](https://zjor.youtrack.cloud/issue/PNDLM-11), we need inertia about the
wheel's central rotation axis, in kg·m². A bifilar suspension measures this
without requiring a uniform print or a known motor torque.

## First estimate from geometry

Weigh the finished wheel and measure its dimensions. For a uniform solid disk,

\[
I = \frac{1}{2}mR^2.
\]

For a uniform annular disk with a central hole,

\[
I = \frac{1}{2}m(R_o^2 + R_i^2).
\]

Here mass is in kilograms and radii are in metres. For example, a uniform
0.200 kg disk of radius 0.100 m has inertia 0.00100 kg·m².

A printed wheel has perimeter walls, top and bottom skins, infill and a hub.
These distribute material differently from a uniform disk, even when the outside
shape is circular. A CAD mass-property calculation is useful if its density
model represents the print; scaling a solid CAD model to the measured mass does
not account for nonuniform infill. Use geometry as a cross-check for the
experimental result.

## Bifilar suspension

Suspend the wheel horizontally from two equal-length, flexible, low-stretch
threads. Attach them at opposite points, equally far from the wheel centre.
The upper attachment points have the same separation as the lower points, so
the threads hang vertically and parallel at rest.

```text
       rigid support
       o           o
       |           |
     L |           | L
       |           |
       o-----+-----o   horizontal wheel, viewed from the side
       <---- d ---->
             |
       central vertical rotation axis
```

The wheel must be free to twist about its centre. Twisting it makes the threads
incline and lifts the wheel slightly. Gravity then supplies the restoring
torque, causing angular oscillations.

Use thin thread rather than stiff wire whose torsional stiffness could add an
unknown restoring torque. Keep knots secure and suspension hardware light.
The centre of mass should lie midway between the threads, directly below the
midpoint of the upper supports. This method assumes a level wheel, small
oscillations, weak damping and negligible thread mass and elasticity.

### Procedure

1. Weigh the suspended assembly, including any attached hub, bolts or hooks.
   Record mass \(m\) and the scale resolution.
2. Measure the free thread length \(L\), from upper to lower attachment point,
   and the full centre-to-centre thread separation \(d\). Record measurement
   uncertainty. Measure both threads and both upper and lower spacings to check
   the assumed geometry.
3. Twist the wheel about its vertical axis by roughly 3–5° and release without
   pushing it sideways. Avoid pendulum-like sideways swinging or rocking.
4. Time \(N = 20\) complete angular oscillations. Start and finish at the same
   phase, for example a centre crossing in the same direction. Compute
   \(T = t_N/N\), the period of one oscillation. Video of a rim marker can make
   crossings easier to identify.
5. Repeat at least three times. Repeat with a different thread length as a
   check: the calculated inertia should agree within the stated uncertainty.
   A meaningful change with length or starting angle indicates a setup effect
   to investigate rather than average away.

### Calculation and derivation

With \(d\) defined as the **full distance between threads**, the inertia is

\[
\boxed{I = \frac{mgd^2T^2}{16\pi^2L}},
\qquad g \approx 9.81\ \mathrm{m/s^2}.
\]

For a small twist \(\theta\), each lower attachment moves horizontally by
approximately \((d/2)\theta\). The wheel rises by

\[
\Delta h \approx \frac{d^2\theta^2}{8L}.
\]

Its gravitational potential energy is therefore equivalent to a torsional
spring with stiffness \(k = mgd^2/(4L)\). Substituting into
\(T = 2\pi\sqrt{I/k}\) gives the boxed formula. If attachment radius
\(a=d/2\) is used instead, the denominator becomes \(4\pi^2L\).
Confusing radius with full spacing introduces a factor-of-four error.

For an illustrative measurement, let
\(m=0.200\ \mathrm{kg}\), \(L=0.500\ \mathrm{m}\),
\(d=0.160\ \mathrm{m}\), and \(T=1.254\ \mathrm{s}\). Then

\[
I \approx 0.00100\ \mathrm{kg\,m^2}.
\]

This is a worked example, not a measured AI-FIP parameter.

## Uncertainty and checks

For small, independent standard uncertainties, propagate them as

\[
\frac{u_I}{I} \approx
\sqrt{
 \left(\frac{u_m}{m}\right)^2 +
 \left(2\frac{u_d}{d}\right)^2 +
 \left(2\frac{u_T}{T}\right)^2 +
 \left(\frac{u_L}{L}\right)^2
}.
\]

Uncertainty in local gravity is usually negligible for this bench measurement.
If total timing uncertainty is \(u_t\), its contribution to period uncertainty
is \(u_t/N\). Include observed run-to-run variation as well; do not treat scale
resolution or a ruler's smallest division as a complete uncertainty estimate.
If quoting a conservative bound instead of a standard uncertainty, state that
convention explicitly.

Both spacing and period enter squared: a 1% error in either produces about a
2% error in inertia. Longer timing intervals reduce the effect of stopwatch
reaction time. Measure attachment spacing rather than substituting the wheel
diameter.

The formula does not include systematic errors from stiff threads, unequal
lengths, a tilted wheel, moving supports, large angles, strong damping or
sideways oscillations. Compare repeated setups and the geometry estimate before
assigning a final uncertainty. A large disagreement is evidence to investigate.

## Which assembly belongs in the motor model?

The suspension measures everything that rotates with the suspended wheel.
Include the final hub and rotating fasteners where practical. Temporary hooks
or a carrier contribute inertia too: either make their contribution negligible
relative to uncertainty, calculate it, or measure the carrier separately and
subtract its inertia. Use the total suspended mass in each measurement before
subtracting inertias; subtraction adds uncertainty.

The motor acceleration experiment needs the total rotating inertia:

\[
I_{\text{rotating}} = I_{\text{wheel+hub+fasteners}} + I_{\text{motor rotor}}.
\]

Add the motor rotor only if it was not included in the suspended assembly.
Document the source and uncertainty of its contribution. For an individual
small part, the parallel-axis theorem gives
\(I_{\text{axis}}=I_{\text{part,CoM}}+mr^2\); the point-mass approximation
\(mr^2\) is appropriate only if the part's own inertia is negligible.

With angular speed \(\omega\) in rad/s and acceleration \(\alpha\) in rad/s²,

\[
I_{\text{rotating}}\alpha
= \tau_{\text{motor}} - \tau_{\text{loss}}(\omega).
\]

For positive-speed coast-down with negligible motor torque,
\(\tau_{\text{loss}}=-I_{\text{rotating}}\alpha\). Zero commanded torque does
not by itself establish zero physical motor torque: record the controller mode,
anticogging configuration and electrical conditions. An independent inertia
measurement avoids trying to identify motor torque, friction and inertia from
the same acceleration trace without enough independent information.

## Measurement record

Store actual results under
[`docs/hardware/measurements/`](../hardware/measurements/), with a dated filename.
Record the assembly identifier and photo, print settings, included parts, mass,
attachment geometry, individual timing runs, video or raw-data paths, calculated
inertia, uncertainty convention and any carrier subtraction. Keep the wheel-only
measurement distinct from the assembled motor-model value.

This article defines the method; experiment scope and completion criteria live
in [PNDLM-11](https://zjor.youtrack.cloud/issue/PNDLM-11).

## References

- [University of Minnesota: Moment of Inertia Estimation Using a Bifilar Pendulum](https://conservancy.umn.edu/server/api/core/bitstreams/57d066b3-8162-4b4a-bc68-cb0186541f48/content)
  — suspension method, equation and centre-of-mass alignment.
- [Government Degree College, Palamaner: Physics practicals, experiment 03](https://gdcplnr.edu.in/admin/uploads/1683i%20sem%20%20PRACTICALS.pdf)
  — experimental procedure and the general equation using half-spacings at the
  upper and lower attachments.
