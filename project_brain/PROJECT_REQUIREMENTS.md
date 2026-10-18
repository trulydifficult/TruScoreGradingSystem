# TruScore — Product Requirements and Acceptance Contract

Established: 2026-10-18
Status: INITIAL REQUIREMENTS BASELINE — product target, not proof of implementation
Original operator intent: [PROJECT_MISSION.md](PROJECT_MISSION.md), preserved unchanged.
Adopted geometry correction: [ADR 0012](decisions/0012-dual-border-precision-contour-contract.md).

## Authority and interpretation

PROJECT_MISSION records **why** TruScore exists. This file identifies
**what** each subsystem must deliver and how to evaluate it. Architecture
decisions describe **how**; `src/`, tests and verified runtime evidence
determine **what works today**. A successful training run, exported
dataset, launch or CI check does not alone establish product acceptance.

Origins below: **MISSION** directly derived from the user's statement;
**ADR0012** previously adopted precision/ground-truth correction;
**ENGINEERING** acceptance/quality guardrail inferred from those goals;
**FUTURE** aspirational business outcome. Where accuracy targets are not
approved, write **TBD**, not fabricated measurements. Proposed engineering
contracts and thresholds remain subject to deliberate decisions.

## Product flow

`card images → identification → specialized condition evidence → grade
distribution → comparable sales + population → grading economics →
evidence-backed advice`

Annotation Studio, Dataset Studio and Phoenix are the model factory.
Card Manager is the operator's integration and inspection workspace.
Guru synthesizes evidence and gives the recommendation. The mobile app
is a client of the backend. Missing data yields an explicit partial or
unavailable result; no silent model substitution or synthetic success.

## Traceable requirements

| ID | Origin | Capability / required output | How to verify acceptance |
|---|---|---|---|
| FACT-01 | MISSION | Annotation Studio captures editable, correctly labeled ground truth for each specific model task. | Independently review annotations against source images; preserve class meaning, rotation, provenance and transforms. |
| FACT-02 | MISSION | Dataset Studio validates, converts, exports and publishes task-compatible labeled datasets to Phoenix. | Source-image/label fidelity, negative compatibility checks, real application Export-to-Trainer and Queue acceptance. |
| FACT-03 | MISSION + ENGINEERING | Phoenix trains the specifically selected model, with real hardware execution, metrics, checkpoints and run provenance; architecture remains backend-neutral. | Model-specific trainer and Queue tests, actual GPU job and held-out task metric; smoke training and quality are distinct gates. |
| FACT-04 | MISSION | Specialized grading models can be independently replaced, ordered and compared. | Versioned I/O, downstream-contract tests and controlled comparisons on identical held-out images. |
| BORDER-01 | MISSION + ADR0012 | Recover two **independent native-pixel contours**, the physical card edge and printed graphic border, with ordered geometry. | Human-reviewed traces and per-boundary localization error on held-out images; approved error limit TBD. |
| BORDER-02 | MISSION + ADR0012 | Measure 24 complete-side contour-to-contour stations: top 5, bottom 5, left 7, right 7. | Known-geometry/skew/rotation tests, full-side sample placement and per-station distance/ratio error; limit TBD. |
| BORDER-03 | MISSION | Report real card centering and skew from measured contours, not approximate boxes. | Independent hand-traced centering/skew reference comparisons and repeatability; limit TBD. |
| BORDER-04 | ADR0012 | Bboxes/OBBs or fabricated rectangular masks cannot masquerade as contour/mask ground truth. | Explicit validation rejection and task labeling; current YOLO11s detection is auxiliary coarse localization only. |
| COND-01 | MISSION | Evaluate corner condition, identify *which* corner is damaged and where. | Human-reviewed corner-localization/condition benchmark; precision/recall and severity agreement, targets TBD. |
| COND-02 | MISSION | Evaluate edge wear, localized damage and severity. | Labeled location/severity benchmark with failure cases; targets TBD. |
| COND-03 | MISSION | Evaluate surface damage and photometric stereo evidence. | Independently reviewed damage masks and calibrated multi-light acquisition/photometric comparison; targets TBD. |
| COND-04 | MISSION | Integrate condition evidence into grade probability distribution and explanation. | Held-out independent grades, calibrated probabilities, repeatability and explicit uncertainty; targets TBD. |
| IDENT-01 | MISSION | Identify year, brand, set, player and relevant variant with OCR/VLM or verified alternatives. | Labeled cross-set/year/variant confusion cases, confidence and abstention behavior; target TBD. |
| MARKET-01 | MISSION | Retrieve relevant PSA population counts and recent comparable auction results. | Source links, timestamps, correct identity/grade pairing, missing/stale data handling. |
| GURU-01 | MISSION | Explain identification, centering/skew, corner/edge/surface findings, photometric evidence, grade odds, market and population. | Each conclusion traceable to versioned measurement/model/market evidence; failures not hidden. |
| GURU-02 | MISSION + ENGINEERING | Advise whether professional grading is economically worthwhile and why. | Reproducible grade-weighted expected value, costs, uncertainty and sensitivity cases; held-out outcome checks TBD. |
| CARD-01 | MISSION | Card Manager runs and exposes the active independent workflows and Guru for an operator-selected card. | Real `run_truscore.py` GUI journey, overlays, metrics, partial/error paths and model provenance. |
| MOBILE-01 | MISSION | Mobile is a front end to the same backend grading/recommendation pipeline. | Versioned API and equivalent answers for the same evidence, including errors; runtime acceptance pending. |
| MOBILE-02 | MISSION | Record every scan and answer as a reviewable longitudinal case. | Immutable/versioned input and prediction provenance, answer and timestamp; privacy/retention policy TBD. |
| LEARN-01 | MISSION + ENGINEERING | Consumer scans can contribute to model improvement without self-labeling loops. | Separate unverified predictions, user corrections, reviewed labels and independent graded outcomes; evaluate before promotion. |
| FUTURE-01 | FUTURE | Potential grading-service credibility or grading-company partnership. | Separate independent validation, operational and business/compliance plans; not a current implemented feature. |

## Feature acceptance checklist

For each requirement ID, record: purpose; input and coordinate frame;
label/source provenance; exact output and consumers; canonical owning
module; archaeological comparisons; realistic failure/abstention
behavior; deterministic tests; actual application integration; held-out
reference metrics and approved thresholds; test/commit/run IDs; and
accurately scoped VERIFIED / PARTIAL / IN PROGRESS state.

Do not claim VERIFIED merely because a model trains or a dataset exports.
No stub masks, generated predictions or box-to-mask approximations may be
labeled ground truth. Never fabricate quality numbers. Preserve working
implementation unless a deliberate requirement correction authorizes a
tested change. See [ADR 0010](decisions/0010-authoritative-memory-and-session-recovery.md)
for the verification hierarchy. Current implementation states are maintained
in SESSION_START, REFACTOR_STATUS, owner audits and the source itself.

## Immediate accepted direction (not completed implementation)

ADR 0012 corrects the YOLO11s-as-primary mistake: YOLO boxes can be a
coarse detection baseline, not precision 24-point centering proof.
Design human-editable independent contours; test corrected full-side
24-station geometry (including skew, rotation and missing intersections);
then benchmark Detectron2/PointRend and other candidate masks/contours
against **held-out human-reviewed physical boundary and station error**.
Choose architecture and accuracy thresholds based on evidence, not the
existence of a training pipeline.
