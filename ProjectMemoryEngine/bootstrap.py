from pathlib import Path
from datetime import datetime
import json
from sessions import SessionManager
from scanner import RepositoryScanner
from decisions import DecisionManager


class ProjectBootstrap:
    """Reconstructs the known state of a project from persistent memory."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.scanner = RepositoryScanner(self.project_root)
        
        self.docs_dir = self.project_root / "docs"
        self.sessions_dir = self.project_root / "sessions"
        self.decisions_dir = self.project_root / "decisions"
        self.memory_dir = self.project_root / "memory"
        
        self.session_manager = SessionManager(self.sessions_dir)
        self.decision_manager = DecisionManager(self.decisions_dir)

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

        content = self.session_manager.read_latest_session()

        if content is None:
            return ""

        return content

    def load_decisions(self) -> list[str]:
        """Load persistent architectural/project decisions."""

        decisions = []

        for path in self.decision_manager.list_decisions():
            content = self.decision_manager.read_decision(path.name)

            if content:
                decisions.append(content)

        return decisions
        
    def save_context(self, data: dict):
        output = self.project_root / "memory" / "bootstrap_state.md"

        output.parent.mkdir(exist_ok=True)

        content = "# Project Bootstrap State\n\n"

        content += f"Generated: {data['timestamp']}\n\n"

        content += "## Documents\n\n"

        for name, text in data["documents"].items():
            content += f"### {name}\n\n{text}\n\n"

        content += "## Latest Session\n\n"
        content += data["latest_session"]

        content += "\n\n## Decisions\n\n"

        for decision in data["decisions"]:
            content += decision
            content += "\n\n"
            
        content += "\n## Repository Map\n\n"

        repository_map = data["repository_map"]

        content += f"Project: {repository_map.get('project', '')}\n"
        content += f"Files: {repository_map.get('file_count', 0)}\n"

        output.write_text(content, encoding="utf-8")

    def reconstruct(self) -> dict:
        """Reconstruct known project state."""

        self.scanner.scan()

        data = {
            "timestamp": datetime.now().isoformat(),
            "documents": self.load_documents(),
            "latest_session": self.load_latest_session(),
            "decisions": self.load_decisions(),
            "repository_map": self.load_repository_map(),
        }

        self.save_context(data)

        return data
        
    def load_repository_map(self) -> dict:
        """Load the latest repository map."""

        path = self.memory_dir / "repository_map.json"

        if not path.exists():
            return {}

        return json.loads(path.read_text(encoding="utf-8"))
