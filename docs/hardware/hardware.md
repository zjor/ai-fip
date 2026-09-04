# Hardware

##### Parameters
| Name               | Value | Units |
| ------------------ | ----- | ----- |
| Plastic density    | 1.25  | g/cm³ |
| Wheel mass         |       |       |
| Wheel inertia      |       |       |
| Pendulum mass      |       |       |
| Pendulum inertia   |       |       |
| Mass of equipment  |       |       |
| Mass of a motor    |       |       |
| Motor torque       |       |       |
| Pendulum length, L | 0.25  | m     |
| Wheel radius, R    | 0.125 | m     |

## Motor selection criteria

*Approved 2026-08-10. Scope: desktop DIY-kit scale, onboard battery (3S–4S), direct-drive BLDC + FOC, motor budget ≤ €120. Parametric — numbers are current estimates (m_t ≈ 0.55 kg incl. battery, l_c ≈ 0.18 m); Phase 0 of the [roadmap](../../ROADMAP.md) iterates them.*

### Operating regimes

Each regime produces a distinct constraint:

- **Balance hold** — near-zero speed, rapid small torque reversals. Stall regime: pure I²R heating with no airflow; cogging torque directly pollutes control.
- **Disturbance recovery** — peak torque at low speed; sets the max recoverable angle.
- **Swing-up** — sustained torque while the wheel spins up; back-EMF eats torque as speed grows. Sets speed headroom and momentum capacity — likely the binding constraint for low-Kv gimbal motors.

### Hard criteria (pass/fail)

| # | Criterion | Formula | First estimate |
|---|---|---|---|
| 1 | Peak torque | τ_peak ≥ k·m_t·g·l_c·sin θ_max, k = 1.5–2, θ_max = 20° | ≥ 0.5–0.65 Nm |
| 2 | Torque reachable at bus voltage | K_t·min(V_bus/R_ph, I_driver) ≥ τ_peak | kills most high-R gimbal motors — check per candidate |
| 3 | Momentum capacity | I_wheel·ω_noload ≥ H_swingup (from honest sim), ω_noload = K_v·V_bus | TBD in Phase 0 — likely binding |
| 4 | Thermal continuous torque | ≥ ~0.3·τ_peak sustained at stall, no airflow | ≥ ~0.2 Nm |
| 5 | Motor mass | ≤ 0.2 kg (feeds back into m_t) | 5208-class ok |
| 6 | Smoothness | low cogging (≤ a few % of τ_peak), FOC-friendly PMSM | gimbal-class winding |
| 7 | Encoder | integrated magnetic encoder or exposed rear shaft for diametric magnet | AS5048A / MT6701 |
| 8 | Mechanical | rotor flange/holes to mount the wheel; Ø40–60 mm | |
| 9 | Price / sourcing | ≤ €120, EU/Ali availability (kit replicability) | T-Motor GB, iPower GM |

### Optimization objective, ranked

1. **Specific torque τ/m_motor (Nm/kg)** — the feasibility lever: motor mass raises its own torque requirement.
2. **K_v sweet spot** — high enough for momentum capacity at 11–15 V, low enough that τ_peak fits the driver current limit. Roughly K_v 30–100, K_t 0.1–0.3 Nm/A, R_ph ≲ 6 Ω.
3. **Cogging + friction minimal** — control quality around zero speed.
4. **Price, then documented torque/speed curve** — the real advantage of the €120 class over AliExpress.

No composite score — rank candidates by specific torque among those passing all nine gates.

**Amendment 2026-08-10:** the I_driver = 5 A assumption (SimpleFOC-class boards) was dropped in favor of moteus-class drivers (20–100 A peak). With the current limit gone, gate 2 stops favoring high-K_t gimbal windings: for the same stator, K_t ∝ turns and R ∝ turns², so the winding-independent figure of merit is the **motor constant K_m = K_t/√R** (torque per √watt of copper loss), and the practical limits become driver current and thermal (I²R at stall), not bus voltage. This reopened low-K_t robot motors → decision: mjbots mj5208 + moteus (see [motor-candidates.md](motor-candidates.md)).

## Driver: moteus — интеграция (researched 2026-08-11)

**Компания:** mjbots Robotic Systems LLC — США, Cambridge, Massachusetts; основана в 2019 Josh Pieper. Прошивка и софт открыты (Apache 2.0, [github.com/mjbots/moteus](https://github.com/mjbots/moteus)). ⚠️ На сайте: "shipping holiday — new orders ship August 17, 2026".

**Интерфейс:** CAN-FD 5 Mbps (разъём JST PH-3). Два протокола: register protocol (команды управления: position/velocity/torque + телеметрия) и diagnostic protocol (конфигурация, консоль). Классический CAN не подходит — нужны CAN-FD кадры; при длинной/шумной шине BRS отключается → 1 Mbps.

**Языки/библиотеки (в репо `lib/`):** Python (`pip install moteus`, asyncio API: `Controller.set_position()/set_stop()/query()`, `make_*` + `transport.cycle()` для мультимоторных шин), C++, Rust. GUI: `pip install moteus-gui` → `tview` (телеметрия, плоты, конфиг, REPL).

**Подключение к хосту (варианты):**
1. **ПК (dev): [fdcanusb](https://mjbots.com/products/fdcanusb) / mjcanfd-usb-1x** — USB→CAN-FD адаптер, Linux/Win/mac. Python-библиотека находит его автоматически (`transport=None`).
2. **Raspberry Pi (боевой контур): [pi3hat](https://mjbots.com/products/mjbots-pi3hat-r4-5)** — шляпа с CAN-FD + IMU на борту (IMU пригодится для угла маятника!); контур управления ≥1 kHz, `pip install moteus-pi3hat`.
3. **MCU (ESP32/STM32):** ESP32 имеет только классический CAN (TWAI) — нужен внешний SPI-контроллер CAN-FD (MCP2518FD) либо STM32G4 с FDCAN. Сложнее; для старта не рекомендуется.

**Электрика (из [electrical-setup](https://github.com/mjbots/moteus/blob/main/docs/guides/electrical-setup.md)):** питание XT30, полярность не защищена; hot-plug запрещён (inrush в конденсаторы — подключить всё, потом подавать питание / использовать mjpower-ss); фазы A/B/C припаиваются в любом порядке; терминатор CAN на конце шины.

**Рекомендуемый стек для маятника:** mj5208 + **moteus r4.11** (решение 2026-09-04: c1 даёт лишь 0.50 Nm peak и 0.125 Nm continuous без обдува — не проходит 20° и continuous-gate; см. [phase-0-feasibility.md](../physics/phase-0-feasibility.md)) + Raspberry Pi с pi3hat (CAN-FD и IMU одним устройством) — Python-контур для LQR/NN; fdcanusb для настольной отладки и tview.

**Константы moteus (проверено по прошивке, 2026-09-04):** K_t = 8.27/Kv (не 60/(2π·Kv)); фазное напряжение ограничено ≈0.4·V_bus, поэтому холостая скорость ≈ 0.7·Kv·V_bus (mj5208 при 12 V: 2740 rpm). Питание: 4S LiPo на оси вращения маятника.

## Legacy — 2025 stepper plan

##### Stepper motor torque
Considering [Nema 17](https://www.gme.cz/v/1500760/nema-17-dc-motor-krokovy-5v-2a-pro-tiskarny)
Current: <2A
Static torque: ~0.2...0.4 Nm (0,48Nm)
Dimensions: 42,3 x 42,3 x 39,8 mm
Mass: 0.22 ... 0.5 (depending on the model)
`Model: 17HS4401`
`1.7A, 0.42Nm`

###### BOM
| Name            | Qty | Price | URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Description              |
| --------------- | --- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Bearing housing | 3   |       | [ali](https://www.aliexpress.com/item/4000909587517.html?src=google&pdp_npi=4%40dis!USD!4.42!4.42!!!!!%40!10000010495781078!ppc!!!&src=google&albch=shopping&acnt=298-731-3000&isdl=y&slnk=&plac=&mtctp=&albbt=Google_7_shopping&aff_platform=google&aff_short_key=UneMJZVf&gclsrc=aw.ds&&albagn=888888&&ds_e_adid=&ds_e_matchtype=&ds_e_device=c&ds_e_network=x&ds_e_product_group_id=&ds_e_product_id=en4000909587517&ds_e_product_merchant_id=106987257&ds_e_product_country=CZ&ds_e_product_language=en&ds_e_product_channel=online&ds_e_product_store_id=&ds_url_v=2&albcp=21554289993&albag=&isSmbAutoCall=false&needSmbHouyi=false&gad_source=1&gad_campaignid=21564543250&gbraid=0AAAAAqc5ie0WuFHEaSblUCPCG4KTQBp6v&gclid=Cj0KCQjww-HABhCGARIsALLO6Xy_Jjb4slaIrayIS9Dwk_6NPYk8YTbeTtoi2NCjvTpJFOWOctNzCPAaAnIMEALw_wcB) | Used for holding a shaft |
| ESP32 DevkitC   | 1   |       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                          |
| A4988           | 1   |       | [hookup instruction](https://lastminuteengineers.com/a4988-stepper-motor-driver-arduino-tutorial/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Stepper motor driver     |
| L7805CV         | 1   |       | [mouser.com](https://cz.mouser.com/ProductDetail/STMicroelectronics/L7805CV?qs=9NrABl3fj%2FqplZAHiYUxWg%3D%3D&mgh=1&utm_id=20199859582&utm_source=google&utm_medium=cpc&utm_marketing_tactic=emeacorp&gad_source=1&gad_campaignid=20199863194&gbraid=0AAAAADn_wf2rCa-8CGSUYbrYQenH4ZJB5&gclid=Cj0KCQjwrPHABhCIARIsAFW2XBMUfr6bIE1hzJSUL7XIqviX-eSuV99Fblm9-hHNfOPhV8GZBxk4uEMaAuwYEALw_wcB)                                                                                                                                                                                                                                                                                                                                                                                                                                     | 5V voltage regulator     |
| MPU9250         | 1   |       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Accel unit               |
