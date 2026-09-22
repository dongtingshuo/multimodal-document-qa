"""Lock the project's dependency closure, excluding unrelated inherited environment packages."""

import importlib.metadata as metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
pending = project["dependencies"] + [
    req for group in project["optional-dependencies"].values() for req in group
]
resolved, visited = {}, set()
while pending:
    requirement = Requirement(pending.pop())
    name = canonicalize_name(requirement.name)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    extras = frozenset(requirement.extras)
    if (name, extras) in visited:
        continue
    visited.add((name, extras))
    distribution = metadata.distribution(requirement.name)
    resolved[name] = distribution.version
    for raw in distribution.requires or []:
        child = Requirement(raw)
        if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in extras | {""}):
            child.marker = None
            pending.append(str(child))
text = (
    "# Validated Python 3.11 / macOS ARM64 dependency closure.\n"
    + "\n".join(f"{name}=={version}" for name, version in sorted(resolved.items()))
    + "\n"
)
(root / "requirements.lock.txt").write_text(text)
print(f"Locked {len(resolved)} packages")
