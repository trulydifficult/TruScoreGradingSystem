import typer
from pathlib import Path
from rich import print

from sessions import SessionManager
from scanner import RepositoryScanner
from context_builder import ContextBuilder
from bootstrap import ProjectBootstrap
from decisions import DecisionManager
from indexer import RepositoryIndexer

app = typer.Typer()


@app.command()
def bootstrap():
    engine = ProjectBootstrap(Path("."))

    result = engine.reconstruct()
    
    builder = ContextBuilder(Path("."))
    context = builder.build()
    builder.save(context)

    print("\nProject Bootstrap Complete\n")

    print(f"Documents: {len(result['documents'])}")
    print(
        f"Latest session: "
        f"{'loaded' if result['latest_session'] else 'none'}"
    )
    print(f"Decisions: {len(result['decisions'])}")

    repository_map = result.get("repository_map", {})
    
    print(
        f"Project: "
        f"{repository_map.get('project', 'Unknown')}"
    )

    print(
        f"Repository files: "
        f"{repository_map.get('file_count', 0)}"
    )
    
    print(
        f"Indexed files: "
        f"{len(result.get('repository_index', []))}"
    )
    
    changes = result.get("changes", {})

    print(
        f"Changes: "
        f"{len(changes.get('added', []))} added, "
        f"{len(changes.get('modified', []))} modified, "
        f"{len(changes.get('deleted', []))} deleted"
    )

    print("\nBootstrap state saved.\n")

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
    
@app.command()
def session(title: str):
    manager = SessionManager(Path("sessions"))

    print(f"\nCreating session: {title}\n")
    print("Enter session content.")
    print("Press Enter on a blank line when finished.\n")

    lines = []

    while True:
        line = input()

        if not line:
            break

        lines.append(line)

    content = "\n".join(lines)

    session_content = f"# Session: {title}\n\n{content}"

    path = manager.create_session(title, session_content)

    builder = ContextBuilder(Path("."))
    context = builder.build()
    builder.save(context)

    print(f"\nSession created: {path}")
    print("Session context updated.\n")
    
@app.command()
def sessions():
    manager = SessionManager(Path("sessions"))

    session_files = manager.list_sessions()

    if not session_files:
        print("\nNo sessions found.\n")
        return

    print("\nSessions:\n")

    for path in session_files:
        print(path.name)

    print()
    
@app.command()
def latest():
    manager = SessionManager(Path("sessions"))

    content = manager.read_latest_session()

    if content is None:
        print("\nNo sessions found.\n")
        return

    print(f"\n{content}\n")
    
@app.command()
def show(filename: str):
    manager = SessionManager(Path("sessions"))

    content = manager.read_session(filename)

    if content is None:
        print(f"\nSession not found: {filename}\n")
        return

    print(f"\n{content}\n")
    
@app.command()
def find(query: str):
    manager = SessionManager(Path("sessions"))

    results = manager.find_sessions(query)

    if not results:
        print(f"\nNo sessions found matching: {query}\n")
        return

    print("\nMatching sessions:\n")

    for path in results:
        print(path.name)

    print()
    
@app.command()
def delete(filename: str):
    manager = SessionManager(Path("sessions"))

    if manager.delete_session(filename):
        print(f"\nSession deleted: {filename}\n")
    else:
        print(f"\nSession not found: {filename}\n")
        
@app.command()
def count():
    manager = SessionManager(Path("sessions"))

    print(f"\nSessions: {manager.count_sessions()}\n")
    
@app.command()
def decision(title: str):
    manager = DecisionManager(Path("decisions"))

    print(f"\nCreating decision: {title}")
    print("Enter decision content.")
    print("Press Enter on a blank line when finished.\n")

    lines = []

    while True:
        line = input()

        if not line:
            break

        lines.append(line)

    content = "\n".join(lines)

    path = manager.create_decision(title, content)

    print(f"\nDecision created: {path}\n")
    
@app.command()
def decisions():
    manager = DecisionManager(Path("decisions"))

    decision_files = manager.list_decisions()

    if not decision_files:
        print("\nNo decisions found.\n")
        return

    print("\nDecisions:\n")

    for path in decision_files:
        print(path.name)

    print()
    
@app.command()
def show_decision(filename: str):
    manager = DecisionManager(Path("decisions"))

    content = manager.read_decision(filename)

    if content is None:
        print(f"\nDecision not found: {filename}\n")
        return

    print(f"\n{content}\n")
    
@app.command()
def find_decisions(query: str):
    manager = DecisionManager(Path("decisions"))

    results = manager.find_decisions(query)

    if not results:
        print(f"\nNo decisions found matching: {query}\n")
        return

    print("\nMatching decisions:\n")

    for path in results:
        print(path.name)

@app.command()
def delete_decision(filename: str):
    manager = DecisionManager(Path("decisions"))

    if manager.delete_decision(filename):
        print(f"\nDecision deleted: {filename}\n")
    else:
        print(f"\nDecision not found: {filename}\n")
        
@app.command()
def count_decisions():
    manager = DecisionManager(Path("decisions"))

    print(f"\nDecisions: {manager.count_decisions()}\n")
    
@app.command()
def find_file(query: str):
    indexer = RepositoryIndexer(Path("."))

    results = indexer.find_file(query)

    if not results:
        print(f"\nNo files found matching: {query}\n")
        return

    print("\nMatching files:\n")

    for file in results:
        print(
            f"{file['path']} "
            f"({file['lines']} lines)"
        )

    print()
    
@app.command()
def index_stats():
    indexer = RepositoryIndexer(Path("."))

    stats = indexer.stats()

    print("\nRepository Index\n")
    print(f"Files: {stats['file_count']}")
    print(f"Lines: {stats['total_lines']}")
    print(f"Bytes: {stats['total_size']}")
    print()
    
@app.command()
def changes():
    indexer = RepositoryIndexer(Path("."))

    result = indexer.detect_changes()

    print("\nRepository Changes\n")

    print(f"Added: {len(result['added'])}")
    for path in result["added"]:
        print(f"  + {path}")

    print(f"Modified: {len(result['modified'])}")
    for path in result["modified"]:
        print(f"  ~ {path}")

    print(f"Deleted: {len(result['deleted'])}")
    for path in result["deleted"]:
        print(f"  - {path}")

    print()
    
@app.command()
def refresh_index():
    indexer = RepositoryIndexer(Path("."))

    result = indexer.refresh()

    changes = result["changes"]

    print("\nRepository Index Refreshed\n")

    print(f"Files: {len(result['files'])}")
    print(f"Added: {len(changes['added'])}")
    print(f"Modified: {len(changes['modified'])}")
    print(f"Deleted: {len(changes['deleted'])}")

    print()