"""Export or import human review files offline without changing the source experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docqa.review_io import export_reviews, import_reviews


def save_review(result_path, output, sheet=None):
    result_path, output = Path(result_path), Path(output)
    if output.exists() or output.resolve() == result_path.resolve():
        raise ValueError("输出文件已存在，请使用新文件名，保留原始结果")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ready":
        raise ValueError("请先完成实验再导出或导入人工评分")
    if sheet is None:
        content = export_reviews(result)
    else:
        updated, _ = import_reviews(result, Path(sheet).read_bytes())
        content = json.dumps(updated, ensure_ascii=False, indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(content)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--import-sheet", type=Path, help="导入人工填写的 CSV，保存为新的结果 JSON")
    args = parser.parse_args()
    print(save_review(args.result, args.output, args.import_sheet))
