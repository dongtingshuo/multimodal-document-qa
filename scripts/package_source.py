"""Create an allowlisted local source snapshot, excluding secrets and runtime data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("docqa", "ui", "scripts", "tests", "datasets", "assets", ".streamlit", ".github", "文档问答.app")
FILES = ('docs/CONFIGURATION.md', 'docs/EVALUATION_GUIDE.md', 'docs/EXPERIMENT_RESULTS.md', 'docs/PROJECT_STRUCTURE.md', 'docs/USER_GUIDE.md', 'docs/evaluation.md', "README.md", "README.zh-CN.md", "CONTRIBUTING.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.lock.txt", "requirements-report.lock.txt",
         ".env.example", ".gitignore", "启动文档问答.command")


def source_files(root):
    paths = [root / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend((root / name).rglob("*"))
    return sorted(path for path in paths if path.is_file() and not path.is_symlink()
                  and "__pycache__" not in path.parts and path.suffix != ".pyc"
                  and path.name not in {".DS_Store", ".env"})


def package(root, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            # Preserve executable bits so macOS launcher scripts remain usable.
            archive.write(path, arcname=name)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(hashes, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("源码归档校验失败")
    return len(hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print("Archived source files:", package(ROOT, args.output))
