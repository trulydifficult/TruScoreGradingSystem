from pathlib import Path
from datetime import datetime


class ProjectBootstrap:
    """Reconstructs the known state of a project from persistent memory."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.docs_dir = self.project_root / "docs"
        self.sessions_dir = self.project_root / "sessions"
        self.decisions_dir = self.project_root / "decisions"

    def _read_file(self, path: Path) -> str:
        if not path.exists():
            return ""

        return path.read_text(encoding="utf-8").strip()

    def load_documents(self) -> dict:
        """Load permanent project knowledge."""

        documents = {}

        for filename in [
            "VISION.md",
            "ARCHITECTURE.md",
            "ROADMAP.md",
        ]:
            path = self.docs_dir / filename
            documents[filename] = self._read_file(path)

        return documents

    def load_latest_session(self) -> str:
        """Return the newest session record."""

        if not self.sessions_dir.exists():
            return ""

        session_files = sorted(
            self.sessions_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not session_files:
            return ""

        return self._read_file(session_files[0])

    def load_decisions(self) -> list[str]:
        """Load persistent architectural/project decisions."""

        if not self.decisions_dir.exists():
            return []

        decisions = []

        for file in sorted(self.decisions_dir.glob("*.md")):
            content = self._read_file(file)

            if content:
                decisions.append(content)

        return decisions

    def reconstruct(self) -> dict:
        """Reconstruct known project state."""

        return {
            "timestamp": datetime.now().isoformat(),
            "documents": self.load_documents(),
            "latest_session": self.load_latest_session(),
            "decisions": self.load_decisions(),
        }
