# Decision 0004 — Phoenix Queue as Persistent Operations Console

Date: 2026-10-15
Status: ACCEPTED

## Context

Phoenix preserves process isolation: the Queue is the control plane and each
training job runs in a separate Trainer worker process. This architecture is
correct for long-running and unattended work, but the first production smoke
runs exposed an operator-observability defect.

A fast Trainer worker could appear for less than a second, exit after successful
training, and leave the operator looking at the unchanged Queue details panel.
The only obvious state change was Pending -> Completed. For multi-hour unattended
queues this is unacceptable: the operator must be able to return later and
reconstruct exactly when datasets started, progressed, completed, failed, and
handed off to the next job.

Historical code under Checkplz included useful UI concepts:
- persistent current-job status
- epoch/overall progress
- live metric panels
- queue lifecycle status

Those implementations also contained simulated training and duplicate training
ownership. Only the monitoring concepts are recovered.

## Decision

1. The Phoenix Training Queue is the persistent operator console for the entire
   queue session.
2. Trainer worker windows are disposable implementation details and must never be
   the sole source of training status, progress, metrics, logs, or outcome.
3. Queue operations are recorded to an append-only durable event stream:
   `exports/training_queue/monitor/queue_events.jsonl`.
4. Queue monitor history survives Queue process restarts. The UI loads recent
   history from the durable event stream when opened.
5. The Queue UI contains a persistent Operations Monitor showing:
   - current state
   - current dataset
   - progress percentage
   - epoch/total epochs
   - current metrics
   - chronological lifecycle/worker timeline
6. The timeline records at least:
   - Queue session opened/closed
   - unattended queue started/stopped
   - dataset dispatched
   - Trainer worker launched
   - active training status/epoch/metrics
   - worker run.log entries
   - dataset completed
   - dataset failed
   - unresolved worker exit
   - queue becoming idle
7. The Trainer publishes `training_run.json` at run start, not only after
   completion, so the Queue can discover and tail the active immutable run record.
8. Queue monitoring reads cross-process state from persisted run/status artifacts;
   it does not execute or own model training.
9. Phoenix Trainer and Queue processes expose their own INFO lifecycle messages to
   the terminal without weakening global TruScore console filtering.
10. A user returning after unattended work must be able to cross-reference the
    Queue event timeline with each dataset's immutable run record.
11. Clearing completed datasets from the visible queue must not erase durable
    operations history.
12. Queue monitor records are operational history, not model-quality metrics and
    not a substitute for the immutable per-run provenance record.

## Recovered historical concepts

Reviewed:
- `Checkplz/phoenix_tensorzero_training/training/phoenix_training_queue.py`
- `Checkplz/phoenix_tensorzero_training/phoenix_training_queue.py`
- `Checkplz/phoenix_tensorzero_training/training/components/training_panel.py`
- `Checkplz/phoenix_tensorzero_training/training/components/metrics_panel.py`
- `Checkplz/phoenix_tensorzero_training/training/phoenix_training_studio.py`

Recovered:
- persistent current-job visibility
- progress presentation
- live metrics presentation
- lifecycle status history as an operator concept

Rejected:
- simulated training
- duplicate queue-side training execution
- fabricated/synthetic runtime metrics presented as real
- moving training ownership back into the Queue process

## Consequences

- A three-job unattended queue has one chronological operator timeline across all
  three jobs.
- A worker may exit immediately after a fast job without destroying operator
  visibility.
- The Queue remains useful when minimized and later restored.
- Run-level forensic artifacts remain authoritative for a specific trained model;
  queue history provides the cross-job operational narrative.
- Future GPU utilization, ETA, throughput, resource telemetry, notification, and
  remote monitoring features should extend this monitor rather than create a
  second competing operations console.
