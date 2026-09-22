import csv
import json

import pytest

from scripts.review_experiment import save_review


def test_offline_review_preserves_source_and_rejects_blank_or_overwrite(tmp_path):
    source = tmp_path / "experiment.json"
    result = {
        "id": "sample", "status": "ready", "rows": [{
            "question_id": "q", "group": "E", "strategy": "hybrid_reranker", "category": "text",
            "question": "value?", "reference_answer": "18", "answerable": True,
            "answer_accuracy": None, "seconds": 1,
            "recall@1": 1, "recall@3": 1, "recall@5": 1, "recall@10": 1, "mrr@10": 1,
            "answer": {"status": "answered", "text": "18 [E1]"},
        }],
    }
    source.write_text(json.dumps(result))
    original = source.read_bytes()
    sheet = save_review(source, tmp_path / "sheet.csv")
    target = tmp_path / "reviewed.json"
    with pytest.raises(ValueError, match="没有填写"):
        save_review(source, target, sheet)
    assert not target.exists()
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    rows[0].update(reviewer="test-only", answer_accuracy="1", is_refusal="0", citation_supported="1")
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_review(source, target, sheet)
    assert json.loads(target.read_text())["rows"][0]["answer_accuracy"] == 1
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="已存在"):
        save_review(source, target, sheet)
