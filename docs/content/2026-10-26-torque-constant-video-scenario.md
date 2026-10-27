# Torque constant: can I trust my simulated motor?

Status: unpublished scenario draft. This defines the story and edit, not the
final spoken script. Target length: about 2 minutes 40 seconds.

Audience: technically curious viewers and engineers without motor-control or
robotics experience. Language of the planned video: English.

Sources: [existing Russian article draft](2026-10-26-torque-constant-draft.md),
[measurement report, sections 9–10](../hardware/motor-torque-fixture-spec.md#9-results-2026-10-26),
[project rationale](../project/roadmap.md),
[completed experiment PNDLM-9](https://zjor.youtrack.cloud/issue/PNDLM-9).

## Story

The previous project exposed a gap between simulation and hardware. This time,
measure what the motor really delivers before relying on it in simulation.
A lever and a scale turn an invisible twisting effort into something visible.
The experiment produces a useful correction: in the tested linear range, the
setup delivers about 12.5% less torque per commanded amp than the controller
estimate predicts.

The central question is **how much torque does one commanded amp actually
produce?** The payoff is a more honest motor model for checking whether the
pendulum can swing up and balance. This experiment supplies one input to that
check; it does not prove the complete robot will work.

## Sequence

Times are editing targets, not constraints on the final narration.

| Time | Story beat | Picture / sound | What the viewer should understand |
|---|---|---|---|
| 0:00–0:05 | Hook: make the question physical | Split screen: lever pressing on the scale on the right; large caption **How much torque does 1 amp really give you?** on the left. Start with visible movement or changing digits. | There is a simple question and a real experiment to answer it. |
| 0:05–0:22 | Stakes: why this measurement matters | Previous pendulum footage, then simulation and current motor. If previous-build footage is unavailable, use a still or simple wheel-and-pendulum sketch. | Accelerating the wheel twists the body in the opposite direction. The motor needs enough twisting effort to swing the pendulum up and catch it. The previous simulation-to-hardware gap is the reason to measure first this time. |
| 0:22–0:40 | The idea: torque per amp | Return to motor close-up. Label torque as “twisting effort” and current as the controlled input. Reveal the relationship in words before symbols: **torque ≈ constant × current + offset**. | More torque-producing current should give more torque. The number being measured is the slope: extra torque for each extra amp. |
| 0:40–1:00 | The measurement: make torque visible | Trace the load path from motor axis along the lever to the scale. Overlay **105 mm**. Show **force × distance = torque**, followed briefly by **τ = mgr**. | The scale senses force and displays an equivalent mass. With a known lever arm, that reading gives an independent torque measurement. |
| 1:00–1:22 | Positive sweep: build a pattern | Montage of 0.5, 1, 2, 4 and 6 A labels with corresponding real scale footage. Let one plateau establish the method, then accelerate the edit. Add graph points as readings appear. | Increasing commanded current increases the measured force. Repeated readings establish a pattern rather than one impressive number. |
| 1:22–1:30 | The boundary: the 8 A attempt | Brief actual stop footage or terminal event. Caption **Fixture limit — test stopped**. | Motion crossed the software speed threshold as the fixture yielded. That point is excluded; it is not a measurement of the motor's maximum torque. |
| 1:30–1:45 | Reverse direction: check the result | Move the scale to the opposite lever end; reverse the current arrow. Use only two or three representative cuts, then overlay both data sets. | Reversing current reverses torque. In the 2–6 A region, torque per additional amp agrees within about 0.3% between directions. Their offsets differ, so the curves do not perfectly overlap. |
| 1:45–1:53 | Human beat: what 6 A feels like | Use the existing personal reaction footage briefly, with **6 A** and the lever visible. Keep the natural sound or reaction. | The plotted torque is a physical twisting effort. This is a subjective reaction, separate from the scale measurement. |
| 1:53–2:23 | Reveal: compare expectation with reality | Zoom into the linear region of the existing graph. Reveal the controller-estimate line, then the measured fit. Main caption **About 12.5% less torque than expected**. Smaller labels: **0.0279 expected / 0.0244 measured N·m per commanded A; fit: 2–6 A**. | The relationship is approximately linear over the fitted range, but its slope differs from the controller estimate. A torque readout based on that same estimate cannot independently verify it. |
| 2:23–2:40 | Consequence: close the loop | Return to pendulum simulation with a motor-gain annotation **0.875 × requested torque**. End on the real setup or presenter. | Use the measured gain when checking swing-up and balance. A correction discovered on a bench is useful before committing to the full build. |

## Explanation level

For q-current, explain the physical idea in one sentence: **the component of
the motor current that produces twisting effort**. If one more sentence fits,
explain that the controller coordinates the three winding currents with the
rotor's magnetic field. The q-axis is how it describes the torque-producing
component; it is not a separate wire or winding. Keep the mathematical
coordinate transformation out of this video.

Introduce the symbols only after the physical explanation:

- **τ:** torque, or twisting effort.
- **K:** the slope being measured, in N·m per amp.
- **I:** commanded torque-producing current for the main graph.
- **b:** the fitted offset; small biases shift the line without being its slope.

For the main on-screen fit, use **τ = K I_command + b**. The report's headline
0.0244 N·m/A is an effective gain per **commanded** amp in the 2–6 A region.
Do not label that horizontal axis as measured q-current: the controller also
adds angle-dependent current for cogging compensation. The measured-current
fits are reported separately in the source data. The distinction between
command and measurement also appears in the
[official moteus register reference](https://mjbots.github.io/moteus/protocol/registers/).

The offset is not a universal motor constant. The two directions have different
offsets, with fixture gravity and residual cogging as unresolved contributors.
For this video, a short visual explanation is enough; identifying the precise
source is not needed to understand the measured slope.

## Editorial choices

Keep one main reveal: the measured gain is lower than expected. The existing
Russian draft also contains two substantial stories—apparently missing current
explained by cogging compensation, and startup delay versus torque rise time.
Those explanations would compete with this video's central question. Their
details remain available in the linked report.

Use the split screen for the hook, then give the scale and lever the full
frame. Brand/model identification (**mjbots mj5208 + moteus**) can be a small
setup label. Introduce CAN only if visible footage benefits from a caption:
it carries requests to the controller; it does not carry the winding current.

Use measured scale readings from the footage. At 6 A, **about 0.15 N·m** is a
useful sense of scale, but the fit slope multiplied by six is not an exact
prediction of each reading because of the offsets. The personal reaction clip
does not establish torque independently and should not become an instruction
to repeat the test by hand.

The headline comparison is with the **controller's configured estimate**, not
a verified manufacturer torque guarantee. Keep its scope visible: static
measurement, tested linear range 2–6 A, this rotor position. Neither continuous
thermal capability nor torque at speed was established here.

The 8 A attempt is part of the surrounding bench story: it was recorded in the
October 25 session and omitted from the final October 26 sweep. Avoid framing
all footage as one uninterrupted run performed today. The final pooled analysis
contains 55 readings from seven runs, not just the five levels shown in the edit.
