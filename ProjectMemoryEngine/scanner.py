from pathlib import Path


class RepositoryScanner:
    IGNORE_DIRS = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
    }
    IGNORE_FILES = {
        "repository_map.json",
    }
    
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def scan(self):
        files = []

        for path in self.root_path.rglob("*"):
            if path.is_file():
                if path.name in self.IGNORE_FILES:
                    continue

                if any(part in self.IGNORE_DIRS for part in path.parts):
                    continue

                files.append(
                    {
                        "path": str(path.relative_to(self.root_path)),
                        "type": path.suffix.replace(".", ""),
                        "size": path.stat().st_size,
                    }
                )

        return {
            "project": self.root_path.resolve().name,
            "root": str(self.root_path),
            "file_count": len(files),
            "files": files,
        }