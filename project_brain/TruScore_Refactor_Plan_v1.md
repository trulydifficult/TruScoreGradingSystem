# TruScoreGradingSystem --- Refactor Plan (v1)

## 1. PURPOSE

Guide controlled refactoring of the existing system without breaking
working functionality.

This plan must be followed exactly. No creative deviations.

------------------------------------------------------------------------

## 2. CORE RULE

DO NOT CHANGE WORKING LOGIC

Refactoring means: - moving code - organizing code - reducing
duplication

NOT: - rewriting algorithms - changing outputs - improving behavior

------------------------------------------------------------------------

## 3. REFACTOR STRATEGY

### Step 1: Select Anchor File

Choose ONE file: - main grading script OR - main pipeline script

Only this file is modified initially.

------------------------------------------------------------------------

### Step 2: Identify Responsibilities

Label each section as: - ingestion - identification - grading - market -
decision - UI / misc

------------------------------------------------------------------------

### Step 3: Extract Functions

For each responsibility: - Find a complete working function - Copy it to
correct module - DO NOT MODIFY FUNCTION LOGIC

------------------------------------------------------------------------

### Step 4: Replace with Imports

Replace original code with imports and function calls.

------------------------------------------------------------------------

### Step 5: Verify Behavior

After each extraction: - run the system - confirm outputs are identical

If behavior changes: - revert immediately

------------------------------------------------------------------------

### Step 6: Remove Duplicates

-   keep the most complete version
-   delete others

------------------------------------------------------------------------

### Step 7: Repeat

Continue until file is reduced to orchestration logic only.

------------------------------------------------------------------------

## 4. STOP CONDITIONS

Stop if: - outputs change unexpectedly - extracted function fails -
unclear ownership of logic

------------------------------------------------------------------------

## 5. SUCCESS CRITERIA

-   clean orchestration
-   no embedded business logic
-   all logic in modules

------------------------------------------------------------------------

## 6. AI INSTRUCTIONS

-   operate on one file at a time
-   do not rewrite logic
-   follow module definitions strictly

------------------------------------------------------------------------

## 7. WARNING

This process is slow by design. Rushing will break the system.
