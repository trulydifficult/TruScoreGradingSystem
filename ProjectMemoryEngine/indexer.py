import hashlib
import json
from pathlib import Path


class RepositoryIndexer:
    """Indexes source files in a repository."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def index_file(self, path: Path) -> dict:
        """Return basic indexed information for a file."""

        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        file_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        return {
            "path": str(path.relative_to(self.project_root)),
            "size": path.stat().st_size,
            "lines": len(content.splitlines()),
            "hash": file_hash,
        }

    def index_repository(self) -> list[dict]:
        """Index supported source files in the repository."""

        files = []

        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue

            if path.parts[0] in {
                ".git",
                ".venv",
                "__pycache__",
            }:
                continue

            if path.suffix not in {
                ".py",
                ".md",
                ".json",
                ".txt",
            }:
                continue

            files.append(self.index_file(path))

        return files

    def save_index(self, files: list[dict]) -> Path:
        """Save the repository index."""

        output = (
            self.project_root
            / "memory"
            / "repository_index.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(files, indent=2),
            encoding="utf-8",
        )

        return output

    def read_index(self) -> list[dict]:
        """Load the persistent repository index."""

        index_path = (
            self.project_root
            / "memory"
            / "repository_index.json"
        )

        if not index_path.exists():
            return []

        return json.loads(
            index_path.read_text(
                encoding="utf-8"
            )
        )

    def find_file(self, query: str) -> list[dict]:
        """Find indexed files matching a path fragment."""

        query = query.lower()

        return [
            file
            for file in self.read_index()
            if query in file["path"].lower()
        ]

    def get_file(self, path: str) -> dict | None:
        """Return an indexed file by exact path."""

        for file in self.read_index():
            if file["path"] == path:
                return file

        return None

    def stats(self) -> dict:
        """Return basic repository index statistics."""

        index = self.read_index()

        return {
            "file_count": len(index),
            "total_lines": sum(
                file["lines"] for file in index
            ),
            "total_size": sum(
                file["size"] for file in index
            ),
        }

    def detect_changes(self) -> dict:
        """Compare the current repository against the saved index."""

        old_index = self.read_index()

        if not old_index:
            return {
                "added": [],
                "modified": [],
                "deleted": [],
            }

        old_files = {
            file["path"]: file
            for file in old_index
        }

        current_files = {
            file["path"]: file
            for file in self.index_repository()
        }

        added = [
            path
            for path in current_files
            if path not in old_files
        ]

        deleted = [
            path
            for path in old_files
            if path not in current_files
        ]

        modified = [
            path
            for path in current_files
            if path in old_files
            and current_files[path]["hash"]
            != old_files[path]["hash"]
        ]

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
        }

    def refresh(self) -> dict:
        """Detect changes and refresh the repository index."""

        changes = self.detect_changes()

        files = self.index_repository()
        self.save_index(files)

        return {
            "files": files,
            "changes": changes,
        }