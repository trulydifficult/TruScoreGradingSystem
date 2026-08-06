import typer
from pathlib import Path
from rich import print

from scanner import RepositoryScanner
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
    
@app.command()
def scan():
    import json

    scanner = RepositoryScanner(Path("."))
    result = scanner.scan()

    output = Path("memory/repository_map.json")

    output.parent.mkdir(exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\nFiles scanned: {result['file_count']}")
    print(f"Saved: {output}\n")
