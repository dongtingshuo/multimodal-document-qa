"""Export generated answers for blind human review without exposing the answer key."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _safe_provenance(result):
    source = result.get("provenance", {})
    return {
        key: source[key]
        for key in [
            "started_at",
            "finished_at",
            "retrieval_version",
            "prompt_sha256",
            "models",
            "device",
            "generation_top_k",
            "max_images",
            "context_char_budget",
        ]
        if key in source
    }


def blind_rows(result):
    rows = []
    for row in result.get("rows", []):
        answer = row.get("answer") or {}
        rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "group": row["group"],
                "strategy": row["strategy"],
                "category": row["category"],
                "answer_status": answer.get("status", "not_generated"),
                "answer_text": answer.get("text", ""),
                "citations": answer.get("citations", {}),
                "evidence": answer.get("evidence", []),
                "image_count": answer.get("image_count", 0),
            }
        )
    return rows


def answer_key(result):
    return [
        {
            "question_id": row["question_id"],
            "group": row["group"],
            "category": row["category"],
            "answerable": row["answerable"],
            "reference_answer": row.get("reference_answer", ""),
            "scoring_points": row.get("scoring_points", []),
            "gold": row.get("gold", []),
            "gold_evidence": row.get("gold_evidence", []),
        }
        for row in result.get("rows", [])
    ]


def export(input_path, output_dir=None):
    input_path = Path(input_path)
    result = json.loads(input_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    common = {
        "source_result": input_path.name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "provenance": _safe_provenance(result),
    }
    blind = {**common, "review_rows": blind_rows(result)}
    key = {**common, "answer_key": answer_key(result)}
    blind_path = output_dir / f"{stem}.blind.json"
    key_path = output_dir / f"{stem}.answer-key.json"
    blind_path.write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    return blind_path, key_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="已完成的实验 JSON")
    parser.add_argument("--output-dir", type=Path, help="输出目录；默认与实验 JSON 相同")
    args = parser.parse_args()
    blind_path, key_path = export(args.result, args.output_dir)
    print("盲评文件：", blind_path)
    print("答案键文件：", key_path)


if __name__ == "__main__":
    main()
