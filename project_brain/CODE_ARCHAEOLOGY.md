# TruScoreGradingSystem — Code Archaeology

## Purpose

TruScore contains valuable code that may not exist in the current implementation.

Earlier development sessions frequently created new scripts without fully reviewing previous implementations.

As a result, older or discarded-looking files may contain functionality, algorithms, architectural ideas, or integrations that should be recovered.

This document records those discoveries so they do not need to be rediscovered in future sessions.

## Archaeology Rule

Before implementing or replacing substantial functionality:

1. Inspect the current `src/` implementation.
2. Search the entire repository for:
   - matching function names
   - matching class names
   - older module names
   - TruGrade-era names
   - backups
   - alternate implementations
   - experimental implementations
   - related documentation
3. Inspect `Checkplz/`.
4. Inspect other archive/backup areas.
5. Search Git history when useful.
6. Compare implementations.
7. Record findings here.
8. Only then decide whether to retain, merge, rewrite, or remove code.

## Known Archaeology Sources

### Checkplz/

Purpose:

Contains files intentionally retained when newer scripts were created because the older versions appeared to contain valuable functionality or ideas.

It must not automatically be treated as obsolete code.

### Checkplz/phoenix_tensorzero_training/

Status:

PARTIALLY INSPECTED

Current counterpart:

`src/modules/phoenix_trainer/`

Potentially valuable files identified:

- `photometric_dataset.py`
- `photometric_training.py`
- `model_architectures.py`
- `revolutionary_training_engine.py`
- `revolutionary_llm_meta_learning.py`
- `enterprise_trainer.py`
- `enterprise_training_interface.py`
- TensorZero-related integration code
- older Phoenix queue/training-studio components

Preliminary finding:

The historical Phoenix tree contains substantially more ambitious training concepts than some current Phoenix implementations.

In particular, the historical photometric trainer contains a genuine multi-light training design using images, lighting directions, surface normals, depth, albedo, and photometric losses.

Do not recreate photometric-training functionality without fully evaluating this implementation.

### Checkplz/claude_fix/

Status:

NOT AUDITED

Purpose/quality:

Unknown until inspected.

Must be searched when its files overlap a subsystem being refactored.

### Old / Backup Files Already Identified

- `src/shared/essentials/truscore_logging_BACKUP.py`
- `src/shared/truscore_system/truscore_border_detection.pybackup`
- `src/trugrade_border_detection.py`

Status:

NOT AUDITED

These files demonstrate that historical implementations are not confined to `Checkplz/`.

## Additional Historical Code

The user has indicated that additional archived scripts exist and were intentionally never deleted.

Their locations and relevance must be cataloged as they are discovered.

## Classification

Historical implementations should eventually receive one of these labels:

### SALVAGE

Contains useful implementation that should be incorporated into the refactored system.

### REFERENCE

Contains useful design or algorithmic ideas but should not be copied directly.

### SUPERSEDED

Current implementation is verified to contain equivalent or better functionality.

### BROKEN

Implementation is defective and should not be restored, but may still contain useful lessons.

### DUPLICATE

Equivalent functionality exists elsewhere and ownership should be consolidated.

### UNKNOWN

Not yet sufficiently inspected.

## Important Principle

Never delete historical code solely because a newer file exists.

First determine whether the newer implementation retained all valuable behavior.
