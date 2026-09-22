import copy

import pytest

from scripts.summarize_visual_ablation import audit


def result():
    rows = []
    for group in "DEF":
        rows.append({
            "group": group, "question_id": "one", "answerable": True,
            "retrieved": ["e1"], "recall@5": 1.0, "mrr@10": 1.0, "seconds": 1.0,
            "answer": {"model": "fixed", "status": "answered", "image_count": int(group != "D"),
                       "context": [{"label": "E1", "element_id": "e1", "text": "value 7",
                                    "truncated": False}]},
        })
    return {"status": "ready", "rows": rows}


def test_audit_accepts_paired_inputs_without_inventing_scores():
    data = result()
    original = copy.deepcopy(data)
    report = audit(data)
    assert len(report["summary"]) == 3
    assert all(row["de_same_text"] for row in report["checks"])
    assert data == original
    assert all("answer_accuracy" not in row for row in report["summary"])


@pytest.mark.parametrize("case", ["missing", "duplicate", "changed_text", "changed_model", "d_image"])
def test_audit_rejects_invalid_comparisons(case):
    data = result()
    if case == "missing":
        data["rows"].pop()
    elif case == "duplicate":
        data["rows"].append(copy.deepcopy(data["rows"][0]))
    elif case == "changed_text":
        data["rows"][1]["answer"]["context"][0]["text"] = "value 9"
    elif case == "changed_model":
        data["rows"][2]["answer"]["model"] = "different"
    else:
        data["rows"][0]["answer"]["image_count"] = 1
    with pytest.raises(ValueError):
        audit(data)
