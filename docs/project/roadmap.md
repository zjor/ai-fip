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

Единая очередь конкретных действий и dependencies: [YouTrack PNDLM](https://zjor.youtrack.cloud/projects/PNDLM).

## Фазы и gates

Содержание фаз и их состояние ведутся в epics YouTrack. Здесь только порядок
фаз и критерии перехода (gates), которых нет в YouTrack.

- **Phase 0 — Simulation foundations and feasibility math.** Epic нет.
  MuJoCo ↔ RK4 и предварительный torque/momentum budget завершены
  ([phase-0-feasibility.md](../physics/phase-0-feasibility.md)); feasible
  window и sensor timing budget закрываются в T-006 и T-012.
  **Gate:** структура модели, координаты, знаки actuation и numerical
  integration проверены; неизвестные hardware parameters явно помечены.
- **Phase 1 — Hardware bring-up and measured inputs.**
  [PNDLM-1](https://zjor.youtrack.cloud/issue/PNDLM-1) motor and controller (Done),
  [PNDLM-2](https://zjor.youtrack.cloud/issue/PNDLM-2) mechanical, power and sensing design.
  **Gate:** каждый simulator input имеет измеренное или datasheet-backed
  nominal value, диапазон неопределённости и provenance.
- **Phase 2 — Honest simulation and classical-control gate.**
  [PNDLM-3](https://zjor.youtrack.cloud/issue/PNDLM-3).
  **Gate:** sampled LQR, despin и swing-up + catch воспроизводимо проходят
  worst-case runs во всём объявленном parameter range. Если classical control
  не проходит honest simulation — вернуться в Phase 1 и изменить
  components/design; final mechanics не изготавливать.
- **Phase 3 — Final build and real classical-control validation.**
  [PNDLM-4](https://zjor.youtrack.cloud/issue/PNDLM-4).
  **Gate:** реальный classical controller работает, а residual model error
  измерен и достаточно ограничен для domain randomization.
- **Phase 4 — RL, sim-to-real and content.**
  [PNDLM-5](https://zjor.youtrack.cloud/issue/PNDLM-5) PPO controller,
  [PNDLM-6](https://zjor.youtrack.cloud/issue/PNDLM-6) robustness and sim-to-real,
  [PNDLM-7](https://zjor.youtrack.cloud/issue/PNDLM-7) publication and demo.
  Residual PPO: `motor command = LQR command + learned correction`.
  **Финиш:** Definition of done в [README](../../README.md).

## Открытые задачи и зависимости (по состоянию на 10.11.2026)

Связи «depends on / is required for» из YouTrack PNDLM по состоянию на
10.11.2026. Источник истины — YouTrack; при расхождении верен YouTrack. Завершённые задачи (T-002 =
[PNDLM-10](https://zjor.youtrack.cloud/issue/PNDLM-10), T-005.A–F =
[PNDLM-8](https://zjor.youtrack.cloud/issue/PNDLM-8),
[PNDLM-11](https://zjor.youtrack.cloud/issue/PNDLM-11),
[PNDLM-9](https://zjor.youtrack.cloud/issue/PNDLM-9),
[PNDLM-12](https://zjor.youtrack.cloud/issue/PNDLM-12),
[PNDLM-13](https://zjor.youtrack.cloud/issue/PNDLM-13),
[PNDLM-14](https://zjor.youtrack.cloud/issue/PNDLM-14)) в граф не входят.

Можно начинать сейчас (все зависимости закрыты):
[PNDLM-15](https://zjor.youtrack.cloud/issue/PNDLM-15) T-003,
[PNDLM-16](https://zjor.youtrack.cloud/issue/PNDLM-16) T-006,
[PNDLM-17](https://zjor.youtrack.cloud/issue/PNDLM-17) T-007.

`A -> B`: A должна быть завершена до B.

- [PNDLM-15](https://zjor.youtrack.cloud/issue/PNDLM-15) T-003 battery offset -> [PNDLM-18](https://zjor.youtrack.cloud/issue/PNDLM-18) T-008 printable mechanics
- [PNDLM-16](https://zjor.youtrack.cloud/issue/PNDLM-16) T-006 host + sensor -> [PNDLM-18](https://zjor.youtrack.cloud/issue/PNDLM-18) T-008 printable mechanics
- [PNDLM-16](https://zjor.youtrack.cloud/issue/PNDLM-16) T-006 host + sensor -> [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model
- [PNDLM-17](https://zjor.youtrack.cloud/issue/PNDLM-17) T-007 battery + charger -> [PNDLM-18](https://zjor.youtrack.cloud/issue/PNDLM-18) T-008 printable mechanics
- [PNDLM-17](https://zjor.youtrack.cloud/issue/PNDLM-17) T-007 battery + charger -> [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model
- [PNDLM-18](https://zjor.youtrack.cloud/issue/PNDLM-18) T-008 printable mechanics -> [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model
- [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model -> [PNDLM-20](https://zjor.youtrack.cloud/issue/PNDLM-20) T-010 LQR stabilize + despin
- [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model -> [PNDLM-21](https://zjor.youtrack.cloud/issue/PNDLM-21) T-011 swing-up + catch
- [PNDLM-19](https://zjor.youtrack.cloud/issue/PNDLM-19) T-009 honest MuJoCo model -> [PNDLM-22](https://zjor.youtrack.cloud/issue/PNDLM-22) T-012 close feasibility gates
- [PNDLM-20](https://zjor.youtrack.cloud/issue/PNDLM-20) T-010 LQR stabilize + despin -> [PNDLM-21](https://zjor.youtrack.cloud/issue/PNDLM-21) T-011 swing-up + catch
- [PNDLM-20](https://zjor.youtrack.cloud/issue/PNDLM-20) T-010 LQR stabilize + despin -> [PNDLM-22](https://zjor.youtrack.cloud/issue/PNDLM-22) T-012 close feasibility gates
- [PNDLM-21](https://zjor.youtrack.cloud/issue/PNDLM-21) T-011 swing-up + catch -> [PNDLM-22](https://zjor.youtrack.cloud/issue/PNDLM-22) T-012 close feasibility gates
- [PNDLM-22](https://zjor.youtrack.cloud/issue/PNDLM-22) T-012 close feasibility gates -> [PNDLM-23](https://zjor.youtrack.cloud/issue/PNDLM-23) T-013 build final device
- [PNDLM-23](https://zjor.youtrack.cloud/issue/PNDLM-23) T-013 build final device -> [PNDLM-24](https://zjor.youtrack.cloud/issue/PNDLM-24) T-014 LQR + swing-up on hardware
- [PNDLM-24](https://zjor.youtrack.cloud/issue/PNDLM-24) T-014 LQR + swing-up on hardware -> [PNDLM-26](https://zjor.youtrack.cloud/issue/PNDLM-26) T-016 Gymnasium environment
- [PNDLM-26](https://zjor.youtrack.cloud/issue/PNDLM-26) T-016 Gymnasium environment -> [PNDLM-27](https://zjor.youtrack.cloud/issue/PNDLM-27) T-017 PPO baseline
- [PNDLM-27](https://zjor.youtrack.cloud/issue/PNDLM-27) T-017 PPO baseline -> [PNDLM-28](https://zjor.youtrack.cloud/issue/PNDLM-28) T-018 robustness + sim-to-real

Без связей в YouTrack:

- [PNDLM-25](https://zjor.youtrack.cloud/issue/PNDLM-25) T-015 RL foundations — в любое время
- [PNDLM-29](https://zjor.youtrack.cloud/issue/PNDLM-29) T-019 articles, demonstrations, media — в любое время
- [PNDLM-30](https://zjor.youtrack.cloud/issue/PNDLM-30) T-020 browser demonstration — по описанию после
  policy из validated RL pipeline, т.е. фактически после T-017 (связь не заведена)

```text
 READY NOW
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │ T-003 (15)   │  │ T-006 (16)   │  │ T-007 (17)   │
 │ battery      │  │ host+sensor  │  │ battery +    │
 │ offset       │  │              │  │ charger      │
 └──────┬───────┘  └───┬──────┬───┘  └───┬──────┬───┘
        │              │      │          │      │
        └──────┬───────┘      │   ┌──────┘      │
               │   ┌──────────┼───┘             │
               ▼   ▼          │                 │
        ┌──────────────┐      │                 │
        │ T-008 (18)   │      │                 │
        │ printable    │      │                 │
        │ mechanics    │      │                 │
        └──────┬───────┘      │                 │
               ▼              ▼                 ▼
        ┌─────────────────────────────────────────┐
        │ T-009 (19)  honest MuJoCo model         │
        └──────┬──────────────┬──────────────┬────┘
               ▼              │              │
        ┌──────────────┐      │              │
        │ T-010 (20)   │      │              │
        │ LQR + despin │      │              │
        └───┬──────┬───┘      │              │
            │      ▼          ▼              │
            │   ┌──────────────────┐         │
            │   │ T-011 (21)       │         │
            │   │ swing-up + catch │         │
            │   └────────┬─────────┘         │
            ▼            ▼                   ▼
        ┌─────────────────────────────────────────┐
        │ T-012 (22)  close feasibility gates     │
        └──────────────────┬──────────────────────┘
                           ▼
                 T-013 (23) build device
                           ▼
                 T-014 (24) LQR on hardware
                           ▼
                 T-016 (26) Gymnasium env
                           ▼
                 T-017 (27) PPO baseline
                           ▼
                 T-018 (28) robustness, sim-to-real
                           ┆ (только по описанию, связи нет)
                           ▼
                 T-020 (30) browser demo

 БЕЗ СВЯЗЕЙ:  T-015 (25) RL foundations   T-019 (29) articles/media
```

Числа в скобках — номер PNDLM issue.

## Log

- **2026-08-09** — roadmap создан, проект воскрешён из архива
- **2026-08-10** — критерии выбора мотора сформулированы ([hardware.md](../hardware/hardware.md#motor-selection-criteria))
- **2026-09-04** — torque budget посчитан, драйвер moteus r4.11 выбран, батарея на оси ([phase-0-feasibility.md](../physics/phase-0-feasibility.md))
