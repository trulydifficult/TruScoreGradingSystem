from datetime import datetime
from pathlib import Path
from typing import Optional
import re

class SessionManager:
    """Manages persistent Project Memory Engine session records."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_session(self, title: str, content: str) -> Path:
        """Create a persistent session record."""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_")

        filename = f"{timestamp}_{safe_title}.md"
        path = self.sessions_dir / filename

        path.write_text(content, encoding="utf-8")

        return path

    def get_latest_session(self) -> Optional[Path]:
        """Return the newest session file, if one exists."""

        sessions = self.list_sessions()

        if not sessions:
            return None

        return sessions[0]

    def read_latest_session(self) -> Optional[str]:
        """Return the contents of the newest session."""

        latest = self.get_latest_session()

        if latest is None:
            return None

        return latest.read_text(encoding="utf-8")

    def list_sessions(self) -> list[Path]:
        """Return all session files, newest first."""

        return sorted(
            self.sessions_dir.glob("*.md"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        
    def read_session(self, filename: str) -> Optional[str]:
        """Return the contents of a specific session."""

        path = self.sessions_dir / filename

        if not path.exists():
            return None

        return path.read_text(encoding="utf-8")
        
    def find_sessions(self, query: str) -> list[Path]:
        """Return sessions whose filename contains the query."""

        query = query.lower()

        return [
            path
            for path in self.list_sessions()
            if query in path.name.lower()
        ]
        
    def delete_session(self, filename: str) -> bool:
        """Delete a session by filename."""

        path = self.sessions_dir / filename

        if not path.exists():
            return False

        path.unlink()

        return True
        
    def count_sessions(self) -> int:
        """Return the number of stored sessions."""

        return len(self.list_sessions())