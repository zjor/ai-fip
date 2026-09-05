# AI-FIP — AI-controlled Flywheel Inverted Pendulum

A flywheel (reaction wheel) inverted pendulum stabilized by a neural network trained with deep reinforcement learning. The network knows nothing about the physics or its own "body"; it learns by trial and error in simulation how to swing up and hold the upright position, and is then transferred to a real device.

## Overview

Digital soul aka AI-FIP (flywheel inverted pendulum) — обратный маятник, реакционное колесо. Само-стабилизирующаяся система, управляемая нейронной сетью. НС обучается в виртуальном пространстве, ничего не зная про устройство механизма и аналитические законы. Методом проб и ошибок (DRL) она должна научиться сама управлять двигателем и считывать угол, чтобы удерживать вертикальное положение.

### Motivation

Это модельная задача обучения НС, ничего не знающей о природе своего тела; потом она сможет обучиться управлять телом любой сложности. Для меня — проектирование механической системы, устройства, изучение текущего состояния нейронных сетей и механизмов обучения с подкреплением.

### Goal

Сделать устройство, из которого можно будет сделать DIY-kit, записать курс, классное промо-видео, сделать стенд на фестиваль света подобный Signal, походить по школам и институтам, показывать для вдохновления будущих поколений.

### Definition of done

- DIY-kit
- YouTube promo-video
- Участие в выставках (Signal, Maker Faire)
- Попробовать разные алгоритмы стабилизации
- Статья на Хабр и на аналогичном популярном зарубежном портале
- Прочитать >3 лекций для общественности
- Записать видео-интервью

## Process

- [Current tasks](project/tasks.md) — single prioritized list of unfinished work
- [Roadmap](project/roadmap.md) — phases, milestones and gates
- [Progress log](project/log.md) — completed work and decisions
- [Project-management process](project/README.md) — ownership and update rules

## Repository map

Organized by concept. `software/` runs on a laptop, `hardware/` runs on the device.

```
project/               tasks, roadmap, log and project-management process
docs/                  knowledge: physics, hardware, DRL notes, reference papers
software/
  sim/                 Python dynamics (RK4), LQR and PID regulators
  rl/                  Gymnasium environment, SB3/PPO training, ONNX export, trained models
  web/                 browser demo running the exported ONNX policy
hardware/
  firmware/
    esp32-stepper/     legacy 2025 build: ESP32 + MPU6050 + stepper
    moteus-host/       2026 build: Raspberry Pi + pi3hat + moteus driver (planned)
  cad/                 mechanics (planned)
  tools/mpu-renderer/  serial → WebSocket bridge for IMU angles
```

Trained policies are produced by `software/rl` into `software/rl/models/`. Consumers copy them at build or deploy time: the web build copies the ONNX file, the device firmware will do the same.

## Documentation

- [Documentation index](docs/README.md)
- [Physics: equations of motion and controllability](docs/physics/equations-of-motion.md)
- [Physics primer](docs/physics/primer.md)
- [Hardware and motor selection criteria](docs/hardware/hardware.md)
- [BLDC motor candidates](docs/hardware/motor-candidates.md)
- [Reference papers](docs/references/README.md)

## Notebooks

- [Regular pendulum damping simulation](https://colab.research.google.com/drive/1u6tl5SG2cvKg8DLndMQ9u2ieP07KyDuc#scrollTo=PbFrm5GsAxk4)
- [Stabilizing FIP with LQR](https://colab.research.google.com/drive/1kIb0vfg7HsaBy3xPXdWx9-hCGLleFcXF#scrollTo=zqa5uH2bmCwl)
