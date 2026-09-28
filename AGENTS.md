# TruScoreGradingSystem

Before substantial work, read:

- project_brain/PROJECT_BOOTSTRAP.md
- project_brain/CURRENT_STATE.md
- project_brain/REFACTOR_STATUS.md
- project_brain/CODE_ARCHAEOLOGY.md
- project_brain/HANDOFF.md
- relevant files under project_brain/decisions/

Follow the architecture and refactoring rules in project_brain/.

src/ is the current working implementation.

Before replacing or implementing substantial functionality, search the entire
repository for previous implementations, duplicates, backups, Checkplz material,
older TruGrade code, experimental code, and relevant Git history.

Do not assume missing functionality was never previously implemented.

Preserve verified working behavior unless an explicit architectural decision
changes it.

Never fabricate successful metrics, results, training behavior, or fallbacks.
Unavailable functionality must fail explicitly.

After substantial work, update the appropriate project_brain continuity files.
