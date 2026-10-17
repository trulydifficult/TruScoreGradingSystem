"""Guard the TruScore project-brain recovery contract against silent drift.

These checks validate document navigation and scope, not application behavior.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
BRAIN = ROOT / "project_brain"


class ProjectMemoryContractTests(unittest.TestCase):
    def test_all_authoritative_startup_documents_are_present(self):
        for name in (
            "AGENTS.md", "README.md", "project_brain/PROJECT_BOOTSTRAP.md",
            "project_brain/PROJECT_MEMORY_INDEX.md",
            "project_brain/SESSION_START.md",
            "project_brain/CURRENT_STATE.md",
            "project_brain/REFACTOR_STATUS.md",
            "project_brain/HANDOFF.md",
            "project_brain/CODE_ARCHAEOLOGY.md",
        ):
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_startup_documents_point_to_one_live_snapshot(self):
        for name in (
            "AGENTS.md", "project_brain/PROJECT_BOOTSTRAP.md",
            "project_brain/PROJECT_MEMORY_INDEX.md",
            "project_brain/HANDOFF.md",
            "project_brain/CURRENT_STATE.md",
        ):
            with self.subTest(path=name):
                content = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("SESSION_START.md", content, name)

        current = (BRAIN / "SESSION_START.md").read_text(encoding="utf-8")
        self.assertIn("refactor/phoenix-foundation", current)
        self.assertIn("DATASET_TO_PHOENIX_HANDOFF_AUDIT.md", current)
        self.assertIn("segmentation", current.lower())
        self.assertIn("NEXT", current.upper())

    def test_every_decision_is_indexed(self):
        index = (BRAIN / "PROJECT_MEMORY_INDEX.md").read_text(encoding="utf-8")
        decisions = sorted((BRAIN / "decisions").glob("*.md"))
        self.assertGreaterEqual(len(decisions), 10)
        for decision in decisions:
            with self.subTest(decision=decision.name):
                self.assertIn("decisions/" + decision.name, index)

    def test_canonical_index_and_snapshot_have_no_broken_relative_links(self):
        for path in (
            BRAIN / "PROJECT_MEMORY_INDEX.md",
            BRAIN / "SESSION_START.md",
            BRAIN / "PROJECT_BOOTSTRAP.md",
            BRAIN / "HANDOFF.md",
            ROOT / "README.md",
        ):
            text = path.read_text(encoding="utf-8")
            for url in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                url = url.split("#", 1)[0]
                if not url or "://" in url or url.startswith("mailto:"):
                    continue
                with self.subTest(source=path.name, link=url):
                    self.assertTrue((path.parent / url).resolve().is_file(),
                                    f"{path}: missing {url}")

    def test_historical_handoff_points_to_active_snapshot(self):
        content = (BRAIN / "HANDOFF.md").read_text(encoding="utf-8")
        markers = [match.start() for match in re.finditer(
            r"^# NEW SESSION START HERE\b", content, flags=re.MULTILINE
        )]
        self.assertTrue(markers)
        self.assertIn("SESSION_START.md", content[markers[-1]:])
        self.assertIn("older October 15 marker", content.lower().replace(
            "the older October 15", "older October 15"
        ).replace("OLDER", "older"))

    def test_truthfulness_notice_is_visible(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("unverified targets", readme)
        self.assertIn("run_truscore.py", readme)
        status = (BRAIN / "REFACTOR_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("PARTIAL VERIFICATION", status)


if __name__ == "__main__":
    unittest.main()
