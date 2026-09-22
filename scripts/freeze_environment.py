"""Record actual local model bytes, cache revisions and dependency/code fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "datasets/runtime_manifest.json",
                    help="新一轮验收请指定新路径，避免覆盖旧基线摘要")
args = parser.parse_args()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


models = []
for repo in ["models--BAAI--bge-small-zh-v1.5", "models--BAAI--bge-reranker-base"]:
    base = ROOT / "models/huggingface/hub" / repo / "snapshots"
    for snapshot in base.glob("*"):
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                models.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "revision": snapshot.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha(path),
                    }
                )
clip = ROOT / "models/chinese-clip/clip_cn_vit-b-16.pt"
if clip.exists():
    models.append({"path": str(clip.relative_to(ROOT)), "bytes": clip.stat().st_size, "sha256": sha(clip)})
code = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "docqa").glob("*.py"))}
manifest = {"models": models, "code": code, "requirements_lock_sha256": sha(ROOT / "requirements.lock.txt")}
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Model files verified:", len(models), "; manifest:", out)
