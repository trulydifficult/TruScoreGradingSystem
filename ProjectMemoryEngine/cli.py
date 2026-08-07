import typer
from pathlib import Path
from rich import print

from scanner import RepositoryScanner
from context_builder import ContextBuilder
from bootstrap import ProjectBootstrap

app = typer.Typer()


@app.command()
def bootstrap():
    scanner = RepositoryScanner(Path("."))
    scan_result = scanner.scan()
    scanner.save(scan_result)

    engine = ProjectBootstrap(Path("."))
    result = engine.reconstruct()

    builder = ContextBuilder(Path("."))
    context = builder.build()
    output = builder.save(context)

    print(f"\nFiles scanned: {scan_result['file_count']}")
    print("Bootstrap state saved: memory/bootstrap_state.md")
    print(f"Session context saved: {output}\n")


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
    scanner = RepositoryScanner(Path("."))

    result = scanner.scan()
    output = scanner.save(result)

    print(f"\nFiles scanned: {result['file_count']}")
    print(f"Repository map saved: {output}\n")
    
@app.command()
def context():
    builder = ContextBuilder(Path("."))

    result = builder.build()
    output = builder.save(result)

    print(f"\nSession context saved: {output}\n")
