from pathlib import Path
from sessions import SessionManager

manager = SessionManager(Path("sessions"))

path = manager.create_session(
    "test",
    "# Test Session\n\nSession memory is working."
)

print(f"Created: {path}")
print(manager.read_latest_session())