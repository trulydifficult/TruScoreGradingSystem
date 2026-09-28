# FILE: project_brain/REFACTOR_STATUS.md

# TruScoreGradingSystem — Refactor Status

Last updated: 2026-09-28

## Status Legend

- NOT STARTED — no complete audit performed
- PARTIAL AUDIT — some files inspected, insufficient for refactoring
- AUDITED — current + historical implementations reviewed
- PLANNED — consolidation/refactor design completed
- IN PROGRESS — code changes underway
- VERIFIED — refactor completed and tested

## Project Status

| System | Current Code Audit | Archaeology | Refactor Plan | Implementation | Verification |
|---|---|---|---|---|---|
| Main Window | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Card Manager | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| TruScore Grading Engine | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Photometric Runtime | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Dataset Studio | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Annotation Studio | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Phoenix Trainer | PARTIAL AUDIT | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED |
| Guru / Continuous Learning | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Mobile API | PARTIAL AUDIT | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Flutter Mobile App | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| CardSight Integration | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |
| Market / Decision Pipeline | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED |

## Enterprise Architecture

Initial target architecture has been defined in:

- `TruScore_Project_Brain_v1.md`
- `TruScore_Modules_v1.md`
- `TruScore_Pipeline_v1.md`
- `TruScore_Refactor_Plan_v1.md`

The target architecture is intentionally cleaner than the current source tree.

Refactoring must move the working implementation toward those boundaries incrementally.

## Current Priority

Repository archaeology and architectural understanding must precede aggressive rewriting.

The immediate objective is to determine:

- what currently works
- what is duplicated
- what is misplaced
- what is incomplete
- what historical code contains superior functionality
- what documentation reflects reality
- what documentation represents research or future design

## Refactoring Rule

No subsystem should be marked AUDITED until both the current implementation and relevant historical implementations have been examined.

No subsystem should be marked VERIFIED until appropriate tests or runtime checks confirm the refactor preserved or deliberately changed behavior.
