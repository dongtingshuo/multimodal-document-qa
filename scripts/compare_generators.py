"""Compare two saved generation runs after checking their shared inputs."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docqa.comparison import compare
from docqa.review_io import export_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/model-comparison'))
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in (args.left, args.right)]
    result = compare(*runs)
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.left, args.right)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (args.output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    for run in runs:
        (args.output / (run['id'] + '.review.csv')).write_bytes(export_reviews(run))
    print(json.dumps({'pairs': len(result['rows']), 'checks_passed': len(result['checks']),
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
