# Project roadmap — resurrection 2026

Принцип: **hardware feasibility first**. Проект умер в 2025 из-за разрыва
симуляция↔железо. Теперь порядок определяется проверяемыми gates:

```text
MuJoCo ↔ RK4
      ↓
bench skills + measurements
      ↓
honest simulation + classical-control gate
      ↓
final design + physical build + real LQR
      ↓
PPO baseline → residual PPO → sim-to-real
```

Policy training не начинается, пока honest simulation с измеренными
ограничениями не пройдёт classical-control gate. Основной RL этап начинается
после проверки этой модели на реальном устройстве с LQR и swing-up: иначе policy
будет учиться использовать ошибки симулятора, а не управлять маятником.

Единая очередь конкретных действий и dependencies: [tasks.md](tasks.md).

## Phase 0 — Simulation foundations and feasibility math

- [ ] Завершить минимальную MuJoCo-модель: free fall, diagnostic torque pulse и
  LQR trajectories воспроизводимо совпадают с analytical RK4 в явных пределах
- [x] Предварительный torque/momentum budget: mj5208 + r4.11 держит 20° с
  запасом 3.7× на пессимистичном углу; swing-up требует H ≈ 0.2–0.45 Nms →
  I_w ≥ 0.0015–0.0022 kg·m², рекомендация 0.003 →
  [phase-0-feasibility.md](../physics/phase-0-feasibility.md)
- [ ] Закрыть предварительное feasible window и sensor timing budget настолько,
  насколько это возможно до bench measurements
- [ ] **Gate:** структура модели, координаты, знаки actuation и numerical
  integration проверены; неизвестные hardware parameters явно помечены

## Phase 1 — Hardware bring-up and measured inputs

- [x] Выбраны mjbots mj5208 и moteus r4.11; order MJ5921 доставлен
- [ ] Освоить безопасную bench-работу с hardware: питание и emergency stop,
  calibration, torque/velocity/position commands, telemetry и fault handling
- [ ] Измерить motor/controller envelope: torque-speed, current/voltage behavior,
  cogging, friction, latency и continuous thermal limit
- [ ] Выбрать host, pendulum-angle sensor, battery и charger; измерить noise,
  sample rate, latency, размеры и массы
- [ ] Закрыть влияющие на физику design inputs: stator mounting, battery offset,
  mass distribution, flywheel inertia и clearances; допускаются bench fixtures и
  fit prototypes, но не final manufacture
- [ ] **Gate:** simulator inputs имеют измеренный или datasheet-backed nominal
  value, диапазон неопределённости и provenance

## Phase 2 — Honest simulation and classical-control gate

Симуляция включает measured torque-speed envelope, current and voltage limits,
flywheel contribution to pendulum inertia, friction/cogging, sensor sampling,
noise/quantization, estimator behavior, command delay и battery voltage range.

- [ ] Sampled LQR стабилизирует верхнее положение и останавливает колесо во всём
  объявленном parameter range
- [ ] Energy swing-up + LQR catch проходят с реальными torque и wheel-speed
  limits
- [ ] Worst-case runs воспроизводимо проходят balance, disturbance recovery,
  swing-up и sensor-timing criteria
- [ ] **Gate:** если classical control не проходит honest simulation, вернуться
  в Phase 1 и изменить components/design; final mechanics не изготавливать

## Phase 3 — Final build and real classical-control validation

- [ ] После Phase 2 gate завершить printable mechanics и изготовить детали
- [ ] Собрать механику и электронику; реализовать sensor/estimator pipeline и
  real-time torque command path
- [ ] Стабилизировать и despin wheel с LQR на реальном устройстве
- [ ] Выполнить swing-up + catch на реальном устройстве
- [ ] Сопоставить recorded hardware telemetry с honest simulation и обновить
  модель по измеренным discrepancies
- [ ] **Gate:** реальный classical controller работает, а residual model error
  измерен и достаточно ограничен для domain randomization

## Phase 4 — RL, sim-to-real and content

- [ ] Освоить необходимые RL foundations и зафиксировать deployable observation
  и action interfaces
- [ ] Построить Gymnasium environment из validated honest model с deterministic
  evaluation, measured domain randomization и held-out parameter corners
- [ ] Обучить небольшой PPO baseline и сравнить с LQR по physical success,
  angle, wheel speed, saturation, current и energy metrics
- [ ] Проверить residual PPO: `motor command = LQR command + learned correction`;
  сравнить pure PPO, residual PPO и classical controller
- [ ] Экспортировать actor в ONNX, измерить target-host latency и выполнить
  staged sim-to-real deployment
- [ ] Статьи, промо-видео и выставки — см. [цели в README](../README.md)

## Log

- **2026-08-09** — roadmap создан, проект воскрешён из архива
- **2026-08-10** — критерии выбора мотора сформулированы ([hardware.md](../hardware/hardware.md#motor-selection-criteria))
- **2026-09-04** — torque budget посчитан, драйвер moteus r4.11 выбран, батарея на оси ([phase-0-feasibility.md](../physics/phase-0-feasibility.md))
