# /src/Logs/log_cleanup.py
from pathlib import Path
import shutil

def archive_old_logs():
    log_dir = Path(__file__).parent
    legacy_dir = log_dir / "legacy"
    legacy_dir.mkdir(exist_ok=True)

    for file in log_dir.glob("*.log"):
        shutil.move(file, legacy_dir / file.name)

if __name__ == "__main__":
    archive_old_logs()
    print("Legacy logs archived. New session logs will be created on-demand.")
