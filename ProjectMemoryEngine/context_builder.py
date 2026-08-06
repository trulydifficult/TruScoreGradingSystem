from pathlib import Path


class ContextBuilder:
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def build(self):
        sources = [
            "STATUS.md",
            "docs/VISION.md",
            "docs/ARCHITECTURE.md",
            "docs/ROADMAP.md",
        ]

        context = []

        for file in sources:
            path = self.root_path / file

            if path.exists():
                context.append(
                    f"\n\n===== {file} =====\n\n"
                    + path.read_text(encoding="utf-8")
                )

        return "".join(context)