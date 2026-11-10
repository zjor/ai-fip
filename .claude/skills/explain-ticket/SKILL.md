---
name: explain-ticket
description: Explain a YouTrack PNDLM ticket (experiment, measurement, decision) in a fixed six-section shape - hypothesis, what was measured, why it matters for the model, verdict, open questions, publishable material. Use when the user asks to read, explain or summarize a PNDLM issue.
---

# Explain a PNDLM ticket

Goal of the answer. The user wants to:
- understand and remember what the experiment taught;
- confirm that the project can go on with these results;
- get material for follow-up questions;
- get material for an article, a graph or a video.

## Steps

1. Read the issue with `mcp__youtrack__get_issue` (ask for 10 recent comments). If `totalComments` is larger, read the rest with `mcp__youtrack__get_issue_comments`.
2. Read the repo files the ticket names (reports in `docs/hardware/`, derived JSON in `docs/hardware/measurements/`, the entry in the newest file of `docs/project/log/`). Read only the parts you need.
3. Check every number against its source. If the ticket contradicts itself (for example, a margin that does not follow from the numbers), mark it **inferred** and name the one check that settles it.
4. Write the answer in the shape below.

## Answer shape

Language: the one the user asks for (often Russian). Plain language, short sentences, exact numbers, units, run IDs and file paths.

Header line: ticket ID, legacy ID, state, close date, commit. Name the source (ticket text, comments, `file:line`).

1. **Гипотеза / Hypothesis.** What we expected. The threshold that decides pass or fail.
2. **Что измеряли / What we measured.** The setup and the method. Key numbers with units and uncertainty. What limits the precision. Which old numbers the result replaces.
3. **Зачем / Why it matters for the model.** Which simulator or hardware parameter it sets, with nominal and pessimistic values. Which gate or check it feeds. Side findings that change the model.
4. **Вердикт / Verdict.** Can we proceed? One line: yes / yes with conditions / no. Name the margin and the conditions.
5. **Открытые вопросы / Open questions.** What the data does not answer and which issue the question moved to. Then 2–4 good follow-up questions to dig deeper.
6. **Материал для публикации / Publishable material.**
   - One graph: axes, data source file.
   - One article angle: the story in one sentence.
   - One video shot: what to film.
   Point to existing data files. Do not invent data.

Keep sections 1–3 short. If the ticket has no content for a section, keep the heading and say so in one line.
