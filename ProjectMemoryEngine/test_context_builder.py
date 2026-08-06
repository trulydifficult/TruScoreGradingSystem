from context_builder import ContextBuilder
from pathlib import Path

builder = ContextBuilder(Path("."))

result = builder.build()

print(result)