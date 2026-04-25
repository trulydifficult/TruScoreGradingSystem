# TruScoreGradingSystem --- Pipeline Specification (v1)

## 1. PURPOSE

Define the exact execution flow and data contracts for the system.

## 2. PIPELINE FLOW

process_card(image): 1. identify_card(image) 2. grade_card(image) 3.
analyze_market(card_data, grade_distribution) 4.
make_decision(grade_distribution, market_data)

## 3. FUNCTION CONTRACTS

### identify_card(image)

Returns: - card_id - name - set - player - confidence

### grade_card(image)

Returns: - probability distribution of grades

### analyze_market(card_data, grade_distribution)

Returns: - prices by grade - raw price - population data

### make_decision(grade_distribution, market_data)

Returns: - expected value - grading cost - recommendation - reasoning

## 4. MASTER FUNCTION

process_card returns: - card - grade - market - decision

## 5. RULES

-   No skipping pipeline steps
-   No duplicate responsibilities
-   Strict schema adherence

## 6. FAILURE HANDLING

-   identification fail → stop
-   grading fail → partial
-   market fail → still return grade
