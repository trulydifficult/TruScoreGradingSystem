PROJECT MEMORY ENGINE - BOOTSTRAP STATE
Mission

We are building ProjectMemoryEngine before refactoring TruScoreGradingSystem.

The goal is to create a completely local, free, model-agnostic memory engine that reconstructs project state for any AI coding assistant (OpenCode, Cursor, Aider, Codex, etc.).

The Memory Engine is developed with the same architectural discipline that will later be applied to TruScore.

Rules
One module at a time.
No duplicate logic.
No business logic in main.py.
No architecture changes without documenting them.
Keep responses concise.
Prioritize implementation over explanation.
Current Folder Structure
ProjectMemoryEngine/

docs/
    VISION.md
    ARCHITECTURE.md
    ROADMAP.md

config/
decisions/
memory/
sessions/
scripts/
tests/

bootstrap.py
cli.py
context_builder.py
decisions.py
indexer.py
knowledge_graph.py
main.py
scanner.py
sessions.py

STATUS.md
requirements.txt
Current Status

Completed

Folder structure
Module layout
STATUS.md
bootstrap.py created
sessions.py created
cli.py created
main.py created
Current Problem

Running

python main.py bootstrap

did not expose the command correctly.

Changing main.py to:

from cli import app

app()

results in:

python main.py --help

showing only the root command.

The next task is to fix the Typer application so that the bootstrap command is correctly registered.

Do not skip ahead to semantic indexing, embeddings, AI integration, or knowledge graphs until the CLI is functioning.

Development Philosophy

ProjectMemoryEngine is not an AI memory.

It is a project state reconstruction engine.

Every future AI session should begin by reconstructing state from persistent project artifacts rather than relying on conversation history.
