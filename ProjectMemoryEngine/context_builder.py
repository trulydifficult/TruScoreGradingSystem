from pathlib import Path


class ContextBuilder:
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def _read_file(self, path: Path) -> str:
        if not path.exists():
            return ""

        return path.read_text(encoding="utf-8").strip()

    def build(self):
        context = []

        sources = [
            "STATUS.md",
            "docs/VISION.md",
            "docs/ARCHITECTURE.md",
            "docs/ROADMAP.md",
            "memory/repository_map.json",
        ]

        for file in sources:
            path = self.root_path / file

            content = self._read_file(path)

            if content:
                context.append(
                    f"\n\n===== {file} =====\n\n{content}"
                )

        decisions_dir = self.root_path / "decisions"

        if decisions_dir.exists():
            for path in sorted(decisions_dir.glob("*.md")):
                content = self._read_file(path)

                if content:
                    context.append(
                        f"\n\n===== {path} =====\n\n{content}"
                    )

        sessions_dir = self.root_path / "sessions"

        if sessions_dir.exists():
            session_files = sorted(
                sessions_dir.glob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if session_files:
                path = session_files[0]
                content = self._read_file(path)

                if content:
                    context.append(
                        f"\n\n===== {path} =====\n\n{content}"
                    )

        return "".join(context)

    def save(self, content: str):
        output = self.root_path / "memory" / "session_context.md"

        output.parent.mkdir(exist_ok=True)

        output.write_text(content, encoding="utf-8")

        return output