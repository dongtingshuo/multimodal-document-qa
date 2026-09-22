"""Submit or resume an experiment and continuously export durable progress."""
import argparse
import csv
import json
import time
from pathlib import Path

import httpx
from api_address import api_address


def export(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (result['id'] + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    rows = [{k: v for k, v in row.items() if k != 'trace'} for row in result.get('rows', [])]
    if rows:
        with path.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?')
    parser.add_argument('--api', help='完整 API 前缀；默认读取实际后端端口')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--strategies', nargs='+', default=['bm25', 'dense', 'hybrid', 'hybrid_reranker'])
    parser.add_argument('--groups', nargs='+', choices=list('ABCDEF'))
    parser.add_argument('--question-ids', nargs='+')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--expected-provider', choices=['ollama', 'bailian'],
                        help='后端不匹配时拒绝提交，防止意外使用云端')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--resume-experiment', help='继续已暂停的实验 ID，自动使用原配置')
    operation.add_argument('--watch-experiment', help='仅查看已有实验 ID，不重新提交')
    operation.add_argument('--pause-experiment', help='暂停已有实验，等待当前题目保存并导出结果')
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--output', type=Path, default=Path('data/experiments'))
    args = parser.parse_args()
    api = api_address(args.api)
    with httpx.Client(timeout=300) as client:
        if args.pause_experiment:
            response = client.post(api + f'/experiments/{args.pause_experiment}/pause')
            response.raise_for_status()
            eid = args.pause_experiment
        elif args.watch_experiment:
            eid = args.watch_experiment
        elif args.resume_experiment:
            response = client.post(api + f'/experiments/{args.resume_experiment}/resume')
            response.raise_for_status()
            eid = response.json()['id']
        else:
            if args.dataset is None:
                parser.error('新建实验需要题集文件')
            data = json.loads(args.dataset.read_text(encoding='utf-8'))
            questions = data['questions'] if isinstance(data, dict) else data
            if args.question_ids:
                selected = set(args.question_ids)
                unknown = selected - {q['id'] for q in questions}
                if unknown:
                    parser.error('未知题目 ID：' + ', '.join(sorted(unknown)))
                questions = [q for q in questions if q['id'] in selected]
            response = client.post(api + '/experiments', json={
                'questions': questions, 'strategies': args.strategies, 'groups': args.groups,
                'generate': args.generate, 'expected_provider': args.expected_provider,
                'document_ids': data.get('document_ids') if isinstance(data, dict) else None,
                'split': args.split,
            })
            response.raise_for_status()
            eid = response.json()['id']
        print('实验 ID：' + eid, flush=True)
        deadline = time.monotonic() + args.timeout
        last = None
        while True:
            response = client.get(api + '/experiments/' + eid)
            response.raise_for_status()
            result = response.json()
            state = (result['status'], result.get('completed', 0))
            if state != last:
                export(result, args.output)
                print(f"{state[0]}：{state[1]} / {result.get('total', '?')}", flush=True)
                last = state
            if result['status'] in {'ready', 'failed', 'paused'}:
                break
            if time.monotonic() >= deadline:
                raise SystemExit(f'等待超时，进度已导出；实验 {eid} 仍在运行，不会重复提交')
            time.sleep(2)
        export(result, args.output)
        if result['status'] == 'failed':
            raise SystemExit(result['error'])
        if result['status'] == 'paused':
            print(f'已暂停；使用 --resume-experiment {eid} 继续，已有结果不会重复提交。', flush=True)
            return
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
