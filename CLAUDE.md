# CLAUDE.md

AI-FIP: a flywheel (reaction wheel) inverted pendulum, to be stabilized by a neural network trained with deep reinforcement learning in simulation and transferred to a real device. Personal research/maker project; docs are a mix of English and Russian, keep whichever the file already uses.

## Layout (by concept)

| Path | Concept | Stack |
|---|---|---|
| `docs/project/` | tasks, roadmap, progress log, process rules | Markdown |
| `docs/physics/` | equations of motion, controllability, torque budget | Markdown + LaTeX |
| `docs/hardware/` | motor selection criteria, candidates, moteus driver notes, BOM | Markdown |
| `docs/drl/`, `docs/references/` | DRL notes, reference papers (numbered PDFs, index in README) | |
| `docs/articles/` | article and video drafts (YAML frontmatter), figures rendered by `software/sim/app/plot_*.py` | Markdown |
| `software/sim/` | RK4 dynamics, LQR/PID regulators | Python, Poetry, package `app` |
| `software/rl/` | Gymnasium env `fip_env`, SB3 training `train/`, ONNX export `export/`, artifacts `models/` | Python, Poetry |
| `software/web/` | browser demo running the ONNX policy | TypeScript, webpack, pnpm |
| `hardware/firmware/esp32-stepper/` | legacy 2025 build (ESP32 + MPU6050 + stepper) | PlatformIO, Arduino |
| `hardware/firmware/moteus-host/` | planned 2026 build (RPi + pi3hat + moteus) | stub |
| `hardware/tools/mpu-renderer/` | serial → WebSocket bridge for IMU angles | TypeScript |

Rule: `software/` runs on a laptop, `hardware/` runs on the device. Each subproject has its own dependency file and README; there is no root build.

## Project management (read these first for context)

- YouTrack project: [PNDLM](https://zjor.youtrack.cloud/projects/PNDLM). Use this project for AI-FIP issue operations; MCP access is configured in `.codex/config.toml` and `.mcp.json`. Setup details: `docs/project/README.md`.

- `README.md` — vision and Definition of done. Rarely changes.
- YouTrack PNDLM is the single source of truth for every unfinished action or open decision. Use Epic → Task and keep one globally ordered queue across hardware, software, learning and content; do not duplicate task state in the repository.
- `docs/project/roadmap.md` — phases 0–4 with milestones and gates, plus a dated snapshot of open tasks and their dependency graph (legacy T-IDs → PNDLM links). The 2026 principle is *hardware feasibility first*: no training code until an honest simulation with real motor limits shows LQR stabilizes the system and stops the wheel.
- `docs/project/log/` — dated, append-only journal split into files by period (index in `log/README.md`; write to the newest file) of completed progress and decisions with rationale. Newest entry on top. Do not add future task lists; put them in YouTrack PNDLM.
- `docs/project/README.md` — process ownership and update rules.
- Component READMEs and technical notes provide context only. Link to YouTrack issues (legacy IDs resolve through the roadmap snapshot) instead of maintaining `Status / Next` lists or progress checkboxes.

## Conventions

- Trained policies are produced only by `software/rl` into `software/rl/models/`. Consumers (web, firmware) copy at build/deploy time; never hand-copy a model into another folder. `*.onnx` there is tracked, `*.pth` is ignored.
- Physical parameters are intentionally not shared between sim, rl and firmware. Each keeps its own block; the Phase 2 gate compares them in `docs/hardware/`.
- Python: Poetry per subproject, Python ≥3.12. Run modules from the subproject root, e.g. `poetry run python -m train.fip_solver --onnx` in `software/rl`.
- Web: `pnpm install`, `pnpm build` (webpack, copies the ONNX from `../rl/models`).
- Plain Markdown only, no Obsidian syntax (wikilinks, callouts). Relative links between docs must resolve.
- Do not commit `dist/`, `logs/`, `.idea/`, `.pio/`, `node_modules/`; root `.gitignore` covers them.
- Commits: only when asked. Branch is `master`.
- Agent process artifacts: implementation plans go to `.claude/plans/YYYY-MM-DD-<topic>.md`; design specs go to `docs/<concept>/specs/<topic>.md` (e.g. `docs/hardware/specs/cad.md`). Never create `docs/superpowers/`; `docs/` is organized by concept only.
- Bench tasks: once the planned protocol gives the headline number, stop adding measurement steps. Write up the result with explicit assumptions (e.g. "±5 mNm residual disturbance, worst case") and record open questions in the spec as documented deviations, not as new runs. Offer at most one extra experiment, clearly optional, and do not block the write-up on it. Reason: bench time is expensive, and a bounded assumption is enough for the Phase 0 feasibility gate.

## Agent knowledge

- Do not use the local auto-memory (`~/.claude/projects/.../memory/`) for this project. Everything an agent should know must travel with git.
- Working rules and preferences go into this file. Repeatable workflows go into project skills in `.claude/skills/<name>/SKILL.md`. Project facts go into `docs/` or YouTrack PNDLM, as described above.
- Project skills: `explain-ticket` — explain a PNDLM ticket in a fixed six-section shape.
