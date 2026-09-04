# BLDC motor candidates

Researched 2026-08-10 against the [selection criteria](hardware.md#motor-selection-criteria) (15 motors evaluated from 22 scouted; sources: manufacturer datasheets, shop listings, SimpleFOC community builds).

**Method note:** reachable torque computed as τ = K_t · min(V_bus / R_terminal, 5 A) using **terminal (line-to-line) resistance** — two phases conduct, so per-phase R understates the limit ~2×. Several manufacturer "peak torque" claims cross-check against this formula (GL40 KV70: 0.15 × 3.29 A = 0.49 ≈ published 0.5 Nm peak).

## Verdict table (ranked)

| Motor | K_t, Nm/A | R_ll, Ω | τ @ 4S, Nm | ω_nl @ 4S, rpm | Mass, g | Ø, mm | Price EU | Encoder | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **CubeMars GL40 KV70** | 0.15 | 4.5 | **0.49** | 940 | 107 | 46.5 | €85 / €110 w-enc | factory variant | **Shortlist #1** — only full-caps pass, but τ ceiling = the floor |
| **iPower GM5208-24** | ~0.40 (derived) | 13.7 (11.1 meas.) | 0.43–0.60 | ~330 | 204 | 59.5 | €52 | add AS5048A in bore | **Shortlist #2** — cheap; swing-up momentum risk (Kv≈21) |
| **CubeMars G60 KV55** | 0.205 | 1.2 | **1.03** (5 A cap) | 814 | 226 | 69 | €132 | encoder bundle avail | **Stretch** — best performer; breaks mass/Ø/price caps ~10–15% |
| CubeMars GL60 KV25 | 0.45 | 5.5 | 1.21 | 318 | 230 | 69 | €132 | bundle avail | Stretch — big torque, low speed ceiling (momentum risk) |
| CubeMars GL40 II KV82.5 | 0.11 | 3.0 | 0.54 | 1221 | 125 | 46.1 | €154 | integrated 14-bit | Over budget; integrated CAN driver, not for external FOC |
| CubeMars GL60 II KV28 | 0.34 | 5.8 | 0.87 | 414 | 276 | 70.5 | €185 | none | Fail: mass+price (best-in-class cogging 1.5 cN·m) |
| iPower GM4108H-120T | ~0.1 (load spec) | 11.1 | ~0.2 | ~400 | 124 | 47 | €40 | AS5048A variant | Fail: torque — canonical SimpleFOC pendulum motor, too weak for our m·g·l |
| iPower GM6208-150T | ~0.87 (derived) | 32 | 0.40 | 163 | 249 | 69.5 | €85 w-enc | AS5048A variant | Fail: torque at 4S + size |
| Maxon EC 45 flat 60W (591477) | 0.036 | 0.94 | 0.18 (5 A cap) | 3922 | 113 | 43.5 | ~€150+? | Halls | Fail: needs ~26 A for its 0.9 Nm stall — wrong class for 5 A/4S |
| Eaglepower LA8308 90KV | 0.106 | 0.186 (unclear) | 0.53 | 1332 | 336 | 92 | €60 | none | Fail: 336 g / Ø92 blow the envelope |
| GBM2804H-100T / MKS YT2804 | ~0.064 | 10 | ~0.08 | 2220 | 50 | 35 | €16 | AS5600 integrated | Fail: torque ~6× short — light stick-pendulums only |
| MotorGo 3506 | unpublished | 9.5 (phase) | ~0.1–0.15 est | ? | 75 | 40 | €28 | magnet built-in | Fail: torque; camera-gimbal class |
| Nidec 24H677 | n/a (integrated driver) | n/a | 0.025 rated | n/a | ? | ? | €6 | FG speed only | Fail: no FOC access, 20× under torque floor — remrc-cube baseline only |

GL40 KV70 listed twice by scouts (two variants of same motor) — merged above. 4S numbers assume 14.8 V and a ≥5 A FOC driver.

## Shortlist notes

### 1. CubeMars GL40 KV70 — the in-budget pick
- Rated 0.25 Nm / peak 0.5 Nm @ 3.3 A; K_v–K_t self-consistent (63.5 rpm/V ↔ 0.15 Nm/A). 14 pole pairs, IP45, low-cogging gimbal winding, 8 mm hollow shaft.
- **On 3S only ~0.37 Nm → must run 4S.** Even then τ_max ≈ 0.5 Nm sits exactly at the criteria floor (k=1.5, θ_max=20°) — Phase 0 must confirm with real m_t·l_c whether the margin holds, or θ_max shrinks to ~15°.
- Its 107 g helps its own cause: lightest of the passing motors, lowers m_t.
- Encoder variant sold (model/IC undisclosed — verify interface before buying); base variant takes a diametric magnet in the 8 mm bore.
- Wheel mount: 4×M3 rotor-side per reseller; bolt circle unpublished — **check CubeMars STEP/2D drawing before designing the flywheel**.
- EU: OpenELAB Munich €84.65 / €109.95 w-encoder. Sources: [cubemars.com](https://www.cubemars.com/product/gl40-kv70-gimbal-motor.html), [store.cubemars.com](https://store.cubemars.com/products/gl40-kv70), [OpenELAB](https://openelab.io/products/cubemars-gl40-kv70-bldc-gimbal)

### 2. iPower GM5208-24 — the cheap proven-ish option
- K_t ~0.40–0.46 Nm/A **derived from Kv, not published**; SimpleFOC community measured lower R than spec (5.57 Ω used as phase value in their build) → τ @ 4S realistically 0.5–0.6 Nm.
- 204 g — exactly at the mass cap; Kv ≈ 21 → ω_nl @ 4S only ~330 rpm → smallest momentum ceiling of the shortlist; swing-up feasibility must be checked in Phase 0 sim first.
- A SimpleFOC user ran FOC current mode on it but called torque "unsatisfying" at a 1.2 A current limit — our 5 A budget is 4× that, so not disqualifying.
- No encoder; standard fix is AS5048A + diametric magnet on a plug in the 12.6 mm bore.
- EU: iflight-rc.eu €51.90 in stock. Sources: [iflight-rc.eu](https://iflight-rc.eu/en/products/ipower-gm5208-24-gimbal-motor), [SimpleFOC thread](https://community.simplefoc.com/t/optimizing-the-capabilities-of-ipower-gm5208-with-foc-current/2068)

### 3. CubeMars G60 KV55 — the stretch pick
- R_ll only 1.2 Ω → driver-limited: full 5 A × 0.205 = **1.03 Nm even on 3S** — 2× margin over the floor, and ω_nl @ 4S ≈ 814 rpm gives the best torque×momentum combination of the list.
- Costs the envelope: 226 g (+13% over cap), Ø69 mm, €132 (+10% over budget). The extra 120 g of motor raises required torque by ~0.1–0.15 Nm — still ~2× margin. If Phase 0 shows GL40's 0.5 Nm is not enough, this is the answer.
- Naming trap: store.cubemars.com "G60 KV55" page shows 472 g / Ø77 — that's an older/larger variant or shipping weight; series table + OpenELAB say 226 g. Verify at order time.
- EU: OpenELAB €131.95. Sources: [cubemars.com G60](https://www.cubemars.com/goods-955-G60.html), [OpenELAB](https://openelab.io/products/cubemars-g60-kv25-kv55-bldc)

## Buy list — top 7 (researched 2026-08-10)

Torque floor relaxed to **≥ 0.45 Nm @ 4S** after the battery-at-pivot decision (see log). τ@4S = K_t·min(14.8/R_ll, 5 A). Prices approximate, EUR, before shipping unless noted. Second sweep added 8 new candidates from SteadyWin / MyActuator (LK-TECH); all verified against manufacturer datasheets.

**Motors #6–7 are integrated-driver servos**: factory FOC driver + encoder onboard, commanded over CAN/RS485 — no SimpleFOC stage to build, but our own FOC code is bypassed. Architecture decision for Phase 1.

| # | Motor | τ@4S, Nm | ω_nl@4S, rpm | Mass, g | Encoder | Best buy | Price |
|---|---|---|---|---|---|---|---|
| 1 | CubeMars GL40 KV70 (w/ encoder) | 0.49 | 940 | 107 | factory (type undisclosed) | [OpenELAB EU — in stock](https://openelab.io/products/cubemars-gl40-kv70-bldc-gimbal) | €109.95 (+€19.95 ship); base €84.65; [CubeMars CN $97.99](https://store.cubemars.com/products/gl40-kv70) |
| 2 | iPower GM5208-24 | 0.5–0.6 | ~330 | 204 | none → AS5048A in bore | [iFlight EU (Austria) — in stock](https://iflight-rc.eu/en-us/products/ipower-gm5208-24-gimbal-motor) | €51.90; [iFlight CN $42.99](https://shop.iflight.com/ipower-motor-gm5208-24-brushless-gimbal-motor-pro1347) |
| 3 | CubeMars G60 KV55 | 1.03 (5 A cap) | 814 | 226 | none | [OpenELAB EU — in stock](https://openelab.io/products/cubemars-g60-kv25-kv55-bldc) | €131.95 (+€19.95); [CubeMars CN $108.99](https://store.cubemars.com/products/g60) — confirm KV55/226 g variant, store header wrongly says 472 g |
| 4 | SteadyWin GB6010-11 (w/ AS5048A) | 0.68–0.88 | ~237 | 241 (270 w-enc) | factory AS5048A or AS5600 | [SteadyWin CN — in stock](https://steadywin-motor.com/products/small-brushless-servo-motor-hollow-flat-gimbal-servo-drone-gimbal-camera-motor-8) | $39.50 w-enc (~€36); [AIFITLAB $36](https://aifitlab.com/products/steadywin-gb6010-motor) |
| 5 | CubeMars GL60 KV25 | 1.21 | 318 | 230 | optional variant | [DigiKey — 18 pcs, ~3 days](https://www.digikey.com/en/products/detail/cubemars/GL-60/16705299) | $108.99; [Oz Robotics w-enc $138.99](https://ozrobotics.com/shop/cubemars-gl60-kv25-bldc-gimbal-motor-with-encoder/); OpenELAB sold out |
| 6 | MyActuator RMD-L-5015 35T CAN (= LK-TECH MF5015 V2) | 0.82 | 636 (datasheet Kv 43; SMC title says 16KV — verify winding) | 174 | 14-bit integrated + CAN driver | [SMC Powers CN — in stock](https://shop.smc-powers.com/MF5015-CAN-16KV.html) | $84 (~€72); [RobotShop EU — restocking](https://eu.robotshop.com/products/myactuator-rmd-l-5015-35t-brushless-dc-motor-can); [datasheet](https://www.myactuator.com/l-5015-details) |
| 7 | MyActuator RMD-L-4015 20T CAN | 0.52 | 932 | 120 | 18-bit integrated + CAN driver | [SMC Powers CN — in stock](https://shop.smc-powers.com/MF4015-CAN-V2.html) | $77.50 (~€67) — confirm 20T winding with seller; [RCDrone $109 — sold out](https://rcdrone.top/products/myactuator-l-4015-20t-direct-drive-servo-motor-24v-0-49n-m-peak-torque-18-bit-encoder-can-rs485-for-robotics-and-drones) |

### Late addition — the winner: mjbots mj5208 + moteus (2026-08-10)

Dropping the SimpleFOC/5 A driver assumption reopened the low-K_t/high-current class ([details in log](../../LOG.md)):

| Motor | K_t, Nm/A | τ peak | ω_nl@4S | Mass | Price | Buy |
|---|---|---|---|---|---|---|
| **mjbots mj5208** (Kv 330) | ~0.029 | 0.58 Nm w/ moteus-c1 (20 A); **1.7 Nm** w/ r4.11 | ~4900 rpm | 193 g | $74 (657 in stock) | [mjbots.com](https://mjbots.com/products/mj5208) |

Pair with [moteus-c1](https://mjbots.com/products/moteus-c1) ($69, 20 A peak / 5 A cont bare, onboard encoder, CAN-FD, 8.9 g) or [moteus r4.11](https://mjbots.com/products/moteus-r4-11) ($94, 100 A peak, 12/32 A cont). Devkit-matched pair: board mounts over the rotor's rear diametric magnet. Bundle ≈ $143 — same as GL40+driver, with 1.2–3.5× the torque and no momentum ceiling. Caveats: US shipping + ~25–30% CZ VAT/duty; Ø63 mm; cogging character unverified (not a gimbal winding — check mjbots anti-cogging before ordering). GL40 KV70 stays as EU-stock fallback.

Honorable mentions (pass torque, fail mass/price): SteadyWin [DD7010-2](https://shop.smc-powers.com/DD7010-2-RS485-CAN.html) (1.10 Nm, 570 rpm, 261 g, $81.80 — don't confuse with -11 variant), LKMTECH [MHF6015-V3](http://shop.smc-powers.com/MHF6015-RS485.html) (1.30 Nm, 252 g, $135), CubeMars GL60 II KV28 (€144+, 276 g).

**Swing-up note:** motors with ω_nl ≲ 350 rpm (#2, #4, #5) have a small momentum ceiling — treat as balance-first unless Phase 0 shows a heavy rim closes the gap. #1, #3, #6, #7 have real speed headroom.

**Buying caveats:** SMC Powers (LK-TECH/MyActuator factory outlet) has an expired TLS certificate — content verified, order with that caveat or prefer RobotShop EU when restocked. AliExpress listings exist for most of these but pages didn't render for verification.

## Implications for Phase 0

1. **4S is effectively mandatory** — every shortlist motor loses 25% torque on 3S and GL40 drops below the floor.
2. **The market has a gap exactly at our spec**: in-budget gimbal motors top out at ~0.5 Nm reachable. Phase 0 feasibility math (real m_t, l_c) decides between "GL40 is enough", "pay/weigh up for G60 KV55", or "shrink the pendulum".
3. **Momentum capacity check** (criterion 3) is now concrete: compute H_swingup and compare against I_wheel × ω_nl for GL40 (940 rpm) vs GM5208 (330 rpm) — likely disqualifies GM5208 for swing-up.
4. Driver must deliver ≥5 A FOC (rules out SimpleFOC Shield v1-class 2 A boards; B-G431B-ESC1 or SimpleFOC PowerShield class needed) — feeds the Phase 1 driver decision.
