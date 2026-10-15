# Decision 0005 — Phoenix Observability Channels and Atomic IPC

Date: 2026-10-15
Status: ACCEPTED

## Context

The persistent Queue Operations Monitor and immutable run records substantially
improved Phoenix observability, but production acceptance exposed two additional
problems:

1. Opening all Phoenix INFO logging to the terminal flooded the CLI with useful
   but low-priority detail. This made actual failures easy to miss.
2. The Queue read `training_status.json` while the Trainer was rewriting it and
   observed a transient empty/partial document, producing a JSONDecodeError
   traceback even though model training itself completed successfully.

The sole operator intentionally keeps the TruScore CLI sparse so unexpected
errors remain visually obvious. Detailed training telemetry is still required,
but it belongs in the Queue Operations Monitor and immutable per-run logs.

## Decision

1. Phoenix has three distinct observability channels with separate purposes:

   **Operator CLI**
   - concise lifecycle/readiness summary only
   - accelerator/backend and VRAM
   - available system RAM
   - free disk space
   - dataset training start
   - Run ID
   - direct run-log path
   - dataset training completion
   - warnings/errors
   - every CLI error is followed by an exact diagnostic log path when available

   **Queue Operations Monitor**
   - persistent cross-job operational narrative
   - live state, epoch, progress, metrics
   - detailed worker lifecycle and run-log tail
   - unattended queue history

   **Immutable Run Record**
   - full forensic detail for one training attempt
   - resolved config, environment/source revision, dataset fingerprint
   - model config, metrics JSONL, checkpoints, run.log, terminal manifest

2. INFO-level implementation/training detail is not automatically terminal-visible.
   Phoenix console handlers use an explicit allowlist marker
   (`phoenix_cli=True`) managed by `operator_logging.py`.

3. Full Python tracebacks remain in file/run logs. The CLI receives a concise
   error summary and a direct path to the detailed log. A traceback should reach
   the terminal only if Phoenix cannot create or locate its normal diagnostic
   log infrastructure.

4. Trainer/Queue code must not globally loosen TruScore logging filters. The
   sparse Phoenix operator policy is local to Phoenix loggers.

5. Cross-process JSON files used as Phoenix IPC are published atomically:
   - serialize complete JSON to a temporary file in the same directory
   - flush and fsync the temporary file
   - publish with `os.replace()`

6. `training_status.json`, active/completed `training_run.json`, run manifests,
   resolved run JSON artifacts, and Queue state use the shared atomic persistence
   helper where appropriate.

7. Monitor readers treat transient absence or a legacy partial JSON document as
   "no update this frame", not as a model-training failure.

8. Monitoring/IPC errors and model-training errors remain semantically distinct.
   A transient monitor read problem must not mark a successfully trained dataset
   failed or visually resemble a trainer traceback in the CLI.

9. Detailed messages such as class discovery, split counts, per-batch loss,
   checkpoint paths, repeated dataset/model initialization milestones, and full
   resolved configuration remain available in the Operations Monitor and run.log,
   not the sparse CLI.

10. Duplicate informational milestones should be reduced over time at their
    ownership boundaries, but CLI cleanliness must not depend on manually pruning
    every detailed message.

## Consequences

- The shell remains useful as an error-focused operational surface.
- Detailed model training evidence is not lost; it is moved to the channels built
  for it.
- Future Phoenix features can add verbose diagnostics without accidentally
  polluting the CLI.
- Cross-process Queue/Trainer JSON polling no longer relies on timing luck.
- A long unattended queue can be reviewed at three levels:
  terminal summary -> Queue timeline -> exact immutable run record.
