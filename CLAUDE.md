# CLAUDE.md

AI-FIP: a flywheel (reaction wheel) inverted pendulum, to be stabilized by a neural network trained with deep reinforcement learning in simulation and transferred to a real device. Personal research/maker project; docs are a mix of English and Russian, keep whichever the file already uses.

## Layout (by concept)

| Path | Concept | Stack |
|---|---|---|
| `docs/physics/` | equations of motion, controllability, torque budget | Markdown + LaTeX |
| `docs/hardware/` | motor selection criteria, candidates, moteus driver notes, BOM | Markdown |
| `docs/drl/`, `docs/references/` | DRL notes, reference papers (numbered PDFs, index in README) | |
| `software/sim/` | RK4 dynamics, LQR/PID regulators | Python, Poetry, package `app` |
| `software/rl/` | Gymnasium env `fip_env`, SB3 training `train/`, ONNX export `export/`, artifacts `models/` | Python, Poetry |
| `software/web/` | browser demo running the ONNX policy | TypeScript, webpack, pnpm |
| `hardware/firmware/esp32-stepper/` | legacy 2025 build (ESP32 + MPU6050 + stepper) | PlatformIO, Arduino |
| `hardware/firmware/moteus-host/` | planned 2026 build (RPi + pi3hat + moteus) | stub |
| `hardware/tools/mpu-renderer/` | serial → WebSocket bridge for IMU angles | TypeScript |

Rule: `software/` runs on a laptop, `hardware/` runs on the device. Each subproject has its own dependency file and README; there is no root build.

## Process files (read these first for context)

- `README.md` — vision and Definition of done. Rarely changes.
- `ROADMAP.md` — phases 0–4 with gates and milestone checkboxes. The 2026 principle is *hardware feasibility first*: no training code until an honest simulation with real motor limits shows LQR stabilizes the system.
- `LOG.md` — dated, append-only journal of progress and decisions with rationale. Newest entry on top. Add an entry when you make or change a decision.
- Component `README.md` → `Status / Next` section — fine-grained todo items. Do not put todos in the roadmap or code comments.

## Conventions

- Trained policies are produced only by `software/rl` into `software/rl/models/`. Consumers (web, firmware) copy at build/deploy time; never hand-copy a model into another folder. `*.onnx` there is tracked, `*.pth` is ignored.
- Physical parameters are intentionally not shared between sim, rl and firmware. Each keeps its own block; the Phase 2 gate compares them in `docs/hardware/`.
- Python: Poetry per subproject, Python ≥3.12. Run modules from the subproject root, e.g. `poetry run python -m train.fip_solver --onnx` in `software/rl`.
- Web: `pnpm install`, `pnpm build` (webpack, copies the ONNX from `../rl/models`).
- Plain Markdown only, no Obsidian syntax (wikilinks, callouts). Relative links between docs must resolve.
- Do not commit `dist/`, `logs/`, `.idea/`, `.pio/`, `node_modules/`; root `.gitignore` covers them.
- Commits: only when asked. Branch is `master`.
- Agent process artifacts: implementation plans go to `.claude/plans/YYYY-MM-DD-<topic>.md`; design specs go to `docs/<concept>/<topic>-spec.md` (e.g. `docs/hardware/cad-spec.md`). Never create `docs/superpowers/`; `docs/` is organized by concept only.
