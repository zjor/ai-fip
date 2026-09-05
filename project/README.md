# Project management

This directory is the operational home of AI-FIP. It separates project state
from technical documentation and keeps unfinished work in one place.

- [Tasks](tasks.md) — the single prioritized list of all unfinished work
- [Roadmap](roadmap.md) — phases, milestones and go/no-go gates
- [Log](log.md) — append-only history of progress and decisions, newest first

## Rules

1. Every unfinished action or open decision lives in `tasks.md`, regardless of
   whether it concerns hardware, simulation, controls, learning, RL or content.
2. `roadmap.md` tracks phase outcomes and gates, not fine-grained actions.
3. Component READMEs and technical documents provide context and link to task
   IDs; they do not maintain separate task lists or progress checkboxes.
4. `log.md` records completed work and decisions with rationale. It does not own
   future work. Historical entries are not rewritten merely because an old task
   has since moved or been completed.
5. `Now` is globally prioritized and limited to three active tasks. `Waiting`
   contains externally blocked work; `Next` is ordered; `Later` is the backlog.
6. A task states an observable completion condition. When completed, remove it
   from `tasks.md` and add a log entry if it changed project state or a decision.

The current governing principle is **hardware feasibility first**: no new policy
training until an honest simulation with real motor limits demonstrates that
classical control can stabilize the pendulum and stop the wheel.

