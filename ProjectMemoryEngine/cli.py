import typer
from pathlib import Path
from rich import print

from bootstrap import ProjectBootstrap

app = typer.Typer()


@app.command()
def bootstrap():
    engine = ProjectBootstrap(Path("."))
    print(engine.reconstruct())


@app.command()
def doctor():
    required = [
        "docs/VISION.md",
        "docs/ARCHITECTURE.md",
        "docs/ROADMAP.md",
        "STATUS.md",
        "sessions",
        "decisions",
        "memory",
    ]

    from pathlib import Path

    print("\nProject Doctor\n")

    missing = False

    for item in required:
        path = Path(item)

        if path.exists():
            print(f"✓ {item}")
        else:
            print(f"✗ {item}")
            missing = True

    if missing:
        print("\nProject is NOT ready.")
    else:
        print("\nProject is healthy.")
