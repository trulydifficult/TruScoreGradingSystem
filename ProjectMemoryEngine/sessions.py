from pathlib import Path
from typing import Optional


class SessionManager:
    """Manages persistent Project Memory Engine session records."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_session(self) -> Optional[Path]:
        """Return the newest session file, if one exists."""

        session_files = list(self.sessions_dir.glob("*.md"))

        if not session_files:
            return None

        return max(session_files, key=lambda path: path.stat().st_mtime)

    def read_latest_session(self) -> Optional[str]:
        """Return the contents of the newest session."""

        latest = self.get_latest_session()

        if latest is None:
            return None

        return latest.read_text(encoding="utf-8")
