"""Apply deterministic margin filtering to cached parsing without re-running models.
Requires the supervisor to be stopped; originals, raw parsing and before snapshots survive.
"""

import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docqa.config import Settings
from docqa.parsers import DoclingParser, remove_repeated_margin_artwork
from docqa.retrieval import HybridRetriever
from docqa.schemas import Element
from docqa.store import Store

settings = Settings(local_models_only=True, api_key="")
store = Store(settings.data_dir)
retriever = HybridRetriever(store, settings)
report_path = ROOT / "data/corpus/import-report.json"
report = json.loads(report_path.read_text())
with (settings.data_dir / "server.lock").open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("请先停止管理当前数据目录的服务，避免与查询并发修改") from None
    for item in report["documents"]:
        doc = store.document(item["document_id"])
        path = ROOT / "data/corpus" / f"{doc['id']}-elements.json"
        before = json.loads(path.read_text())
        backup = path.with_name(path.stem + "-before-refinement.json")
        if not backup.exists():
            backup.write_text(json.dumps(before, ensure_ascii=False))
        filtered = remove_repeated_margin_artwork([Element.model_validate(e) for e in before])
        ignored = {e["id"] for e in before} - {e.id for e in filtered}
        store.replace_elements(doc["id"], filtered)
        chunks = retriever.chunks(filtered)
        doc.update(element_count=len(filtered), chunk_count=len(chunks), parser_version=DoclingParser.version)
        doc["ignored_margin_artwork"] = sorted(set(doc.get("ignored_margin_artwork", [])) | ignored)
        store.put_document(doc)
        (settings.data_dir / "documents" / doc["id"] / "parsed/chunks.json").write_text(
            json.dumps([c.model_dump() for c in chunks], ensure_ascii=False)
        )
        path.write_text(json.dumps([e.model_dump() for e in filtered], ensure_ascii=False, indent=2))
        item.update(
            element_count=len(filtered),
            kinds={k: sum(e.kind == k for e in filtered) for k in ["text", "table", "image", "page"]},
            parser_version=DoclingParser.version,
        )
        print(doc["name"], "removed recurrent margin graphics:", len(ignored), flush=True)
    retriever.invalidate()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
