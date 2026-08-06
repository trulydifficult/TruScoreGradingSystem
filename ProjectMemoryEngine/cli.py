import typer
from pathlib import Path
from rich import print

from scanner import RepositoryScanner
from context_builder import ContextBuilder
from bootstrap import ProjectBootstrap

app = typer.Typer()


@app.command()
def bootstrap():
    engine = ProjectBootstrap(Path("."))
    result = engine.reconstruct()

    print(result)
    print("\nBootstrap state saved: memory/bootstrap_state.md\n")


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
    
@app.command()
def context():
    builder = ContextBuilder(Path("."))
    result = builder.build()

    output = Path("memory/project_context.md")

    output.parent.mkdir(exist_ok=True)

    output.write_text(result, encoding="utf-8")

    print(f"\nContext saved: {output}\n")
