from scanner import RepositoryScanner
from pathlib import Path

scanner = RepositoryScanner(Path("."))

result = scanner.scan()

print(result["file_count"])
print(result["files"][:3])