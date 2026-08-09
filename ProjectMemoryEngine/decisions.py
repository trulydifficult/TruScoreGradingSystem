from pathlib import Path
from datetime import datetime


class DecisionManager:
    """Manages persistent architectural and project decisions."""

    def __init__(self, decisions_dir: Path):
        self.decisions_dir = Path(decisions_dir)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

    def create_decision(self, title: str, content: str) -> Path:
        """Create a persistent decision record."""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_title = "".join(
            character if character.isalnum() or character in "-_"
            else "_"
            for character in title
        ).strip("_")

        filename = f"{timestamp}_{safe_title}.md"
        path = self.decisions_dir / filename

        decision = (
            f"# Decision: {title}\n\n"
            f"Date: {datetime.now().isoformat()}\n\n"
            f"{content}\n"
        )

        path.write_text(decision, encoding="utf-8")

        return path

    def list_decisions(self) -> list[Path]:
        """Return all decisions, newest first."""

        return sorted(
            self.decisions_dir.glob("*.md"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        
    def read_decision(self, filename: str) -> str | None:
        """Return the contents of a specific decision."""

        path = self.decisions_dir / filename

        if not path.exists():
            return None

        return path.read_text(encoding="utf-8")
        
    def find_decisions(self, query: str) -> list[Path]:
        """Return decisions whose filename contains the query."""

        query = query.lower()

        return [
            path
            for path in self.list_decisions()
            if query in path.name.lower()
        ]
        
    def delete_decision(self, filename: str) -> bool:
        """Delete a decision by filename."""

        path = self.decisions_dir / filename

        if not path.exists():
            return False

        path.unlink()

        return True
        
    def count_decisions(self) -> int:
        """Return the number of stored decisions."""

        return len(self.list_decisions())