# TruScoreGradingSystem — Project Bootstrap

## Purpose

This document is the starting point for every substantial AI-assisted development session involving TruScoreGradingSystem.

Its purpose is to allow a new session to reconstruct the project correctly before modifying code.

## Repository

Authoritative GitHub repository:

`trulydifficult/TruScoreGradingSystem`

Default branch:

`main`

## Source Hierarchy

### 1. Current Working Implementation

`src/`

The `src/` directory represents the current working application as it exists today.

Current source code and verified runtime behavior take precedence over documentation claims about implementation status.

### 2. Target Enterprise Architecture

`project_brain/`

This directory defines where TruScore is intentionally going during the enterprise refactor.

It contains:

- target module boundaries
- pipeline design
- refactoring rules
- project state
- architectural decisions
- handoff information

The project brain describes the desired architecture. It does not imply that the current `src/` tree already conforms to that architecture.

### 3. Code Archaeology Sources

Potentially valuable previous implementations exist throughout the repository.

Known locations include:

- `Checkplz/`
- backup files
- older TruGrade-era scripts
- replaced root-level scripts
- experimental implementations
- oddly named historical files
- duplicate classes/functions
- Git history
- any additional archived code discovered later

`Checkplz/` is not merely an archive.

Files were placed there when newer scripts replaced older work but the older files appeared to contain functionality or ideas worth retaining.

Historical code must therefore be inspected before equivalent functionality is recreated.

Absence of functionality from the current `src/` implementation does NOT prove that the functionality was never implemented elsewhere in the repository.

### 4. Research and Future Design

`src/Docs/`

This area contains research, architecture ideas, future capabilities, experiments, and historical documentation.

Documentation is not proof of implementation.

Any claim that a documented feature currently works must be verified against current source code and/or runtime behavior.

## Required Session Startup

Before substantial development work:

1. Read this file.
2. Read `CURRENT_STATE.md`.
3. Read `REFACTOR_STATUS.md`.
4. Read `HANDOFF.md`.
5. Read the relevant existing architecture files:
   - `TruScore_Project_Brain_v1.md`
   - `TruScore_Modules_v1.md`
   - `TruScore_Pipeline_v1.md`
   - `TruScore_Refactor_Plan_v1.md`
6. Read relevant decisions under `project_brain/decisions/`.
7. Inspect the current implementation in `src/`.
8. Search the entire repository for related older or duplicate implementations before replacing or creating functionality.
9. Inspect Git history when it may explain why an implementation changed.

## Development Rules

- Preserve verified working functionality unless a deliberate architectural decision changes it.
- Refactor toward the architecture defined in `project_brain/`.
- Do not casually rewrite working algorithms merely because cleaner code can be written.
- Eliminate duplicate ownership.
- Do not introduce additional duplicate implementations.
- Never silently substitute an unrelated model, algorithm, or fallback.
- Never fabricate training metrics, grading results, success states, or test results.
- Missing functionality must fail explicitly.
- Verify substantial changes with tests or runtime checks.
- When one defect is found, search for similar occurrences elsewhere.
- Record important architectural decisions.
- Update project-memory documents after substantial development work.

## Core Refactoring Principle

The objective is not simply to make TruScore run.

The objective is to transform the existing working application into a coherent, testable, maintainable professional architecture without losing valuable functionality created during earlier development.
