# Project management

This directory is the operational section of the AI-FIP documentation. It
keeps process rules, milestones and engineering history. Unfinished work lives
in YouTrack PNDLM.

- [YouTrack PNDLM](https://zjor.youtrack.cloud/projects/PNDLM) — all unfinished work
- [Task reference index](tasks.md) — static legacy ID → YouTrack issue mapping
- [Roadmap](roadmap.md) — phases, milestones and go/no-go gates
- [Log](log.md) — append-only history of progress and decisions, newest first

## YouTrack connection

Project: [PNDLM — Pendulum](https://zjor.youtrack.cloud/projects/PNDLM).
Use `PNDLM` for AI-FIP issue operations; the shared server also exposes other
projects, so scope issue searches explicitly.

- Codex: `.codex/config.toml` connects to `https://zjor.youtrack.cloud/mcp`;
  `.codex/youtrack-headers.sh` reads `YOUTRACK_API_TOKEN` from root `.env.local`.
- Claude Code: `.mcp.json` uses the same endpoint and environment variable.
  Export the token before launching: `set -a; . ./.env.local; set +a`.
- `.env.local` is Git-ignored and must stay private. On another checkout, supply
  the token locally and update the absolute helper path in `.codex/config.toml`.
- Start a new agent session after changing MCP configuration.

The backlog migrated on 2026-10-25: seven Epics and 23 child Tasks, preserving
full descriptions and 27 explicit task dependencies. YouTrack is now the single
source of truth; `tasks.md` contains only a static ID mapping.

## Rules

1. Every unfinished action or open decision lives in YouTrack PNDLM, regardless of
   whether it concerns hardware, simulation, controls, learning, RL or content.
2. `roadmap.md` tracks phase outcomes and gates, not fine-grained actions.
3. Component READMEs and technical documents provide context and link to task
   issues; they do not maintain separate task lists or progress checkboxes.
   Use Epic → Task; create dependencies with issue links. Legacy IDs resolve
   through `tasks.md`; new work uses PNDLM issue IDs.
4. `log.md` records completed work and decisions with rationale. It does not own
   future work. Historical entries are not rewritten merely because an old task
   has since moved or been completed.
5. `Now` is globally prioritized and limited to three active tasks. `Waiting`
   contains externally blocked work; `Next` is ordered; `Later` is the backlog.
   Maintain this global queue in YouTrack. Currently queue placement and order
   are recorded in task descriptions, not dedicated fields or board ordering.
   These planning categories are separate from the issue State.
6. A task states an observable completion condition. When completed, mark it
   Done in YouTrack and add a log entry if it changed project state or a decision.
   Do not recreate a repository backlog or maintain a synchronized copy.

The current governing principle is **hardware feasibility first**: no new policy
training until an honest simulation with real motor limits demonstrates that
classical control can stabilize the pendulum and stop the wheel.
