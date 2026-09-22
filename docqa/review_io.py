"""Portable human review worksheets, bound to original answers and imported atomically."""
import copy
import csv
import hashlib
import io
import json

from .evaluation import Review, apply_review

COLUMNS = ['experiment_id', 'group', 'question_id', 'answer_sha256', 'question', 'answer',
           'reference_answer', 'reviewer', 'answer_accuracy', 'is_refusal', 'citation_supported', 'note']


def cell(value):
    text = str(value or '')
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def answer_hash(row):
    content = json.dumps(row.get('answer'), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def export_reviews(result):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    for row in result.get('rows', []):
        if row.get('answer', {}).get('status') not in {'answered', 'no_evidence'}:
            continue
        previous = row.get('review', {})
        score = previous.get('answer_accuracy')
        writer.writerow({
            'experiment_id': cell(result['id']), 'group': cell(row['group']), 'question_id': cell(row['question_id']),
            'answer_sha256': answer_hash(row), 'question': cell(row['question']),
            'answer': cell(row['answer']['text']), 'reference_answer': cell(row.get('reference_answer')),
            'reviewer': cell(previous.get('reviewer')), 'answer_accuracy': '' if score is None else score,
            'is_refusal': '' if not previous else int(previous['is_refusal']),
            'citation_supported': '' if previous.get('citation_supported') is None else int(previous['citation_supported']),
            'note': cell(previous.get('note')),
        })
    return output.getvalue().encode('utf-8-sig')


def boolean(value, label, optional=False):
    text = value.strip().lower()
    if optional and not text:
        return None
    if text in {'1', 'true', '是'}:
        return True
    if text in {'0', 'false', '否'}:
        return False
    raise ValueError(f'{label}需填写 1（是）或 0（否）')


def import_reviews(result, content):
    if result.get('status') != 'ready':
        raise ValueError('实验尚未完成，不能导入评分')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('请保存为 UTF-8 CSV 后再导入') from exc
    reader = csv.DictReader(io.StringIO(text, newline=''))
    if not reader.fieldnames or not set(COLUMNS) <= set(reader.fieldnames):
        raise ValueError('评分表列不完整，请使用本系统导出的模板')
    by_key = {(cell(row['group']), cell(row['question_id'])): row for row in result['rows']}
    seen, reviews = set(), []
    for line, record in enumerate(reader, 2):
        if None in record or any(value is None for value in record.values()):
            raise ValueError(f'第 {line} 行 CSV 列数错误')
        key = (record['group'], record['question_id'])
        if key in seen:
            raise ValueError(f'第 {line} 行题目重复')
        seen.add(key)
        row = by_key.get(key)
        if record['experiment_id'] != cell(result['id']) or row is None:
            raise ValueError(f'第 {line} 行不属于该实验')
        if (record['answer_sha256'] != answer_hash(row) or record['question'] != cell(row['question'])
                or record['answer'] != cell(row.get('answer', {}).get('text'))):
            raise ValueError(f'第 {line} 行问题或回答与原始结果不一致，请重新导出评分表')
        fields = ['reviewer', 'answer_accuracy', 'is_refusal', 'citation_supported', 'note']
        if not any(record[field].strip() for field in fields):
            continue  # A blank worksheet is never interpreted as a zero score.
        if not record['reviewer'].strip():
            raise ValueError(f'第 {line} 行缺少评分人')
        score = record['answer_accuracy'].strip()
        if score and score not in {'0', '0.0', '0.5', '1', '1.0'}:
            raise ValueError(f'第 {line} 行得分只能为 0、0.5 或 1')
        reviews.append(Review(
            question_id=row['question_id'], group=row['group'], reviewer=record['reviewer'].strip(),
            answer_accuracy=float(score) if score else None,
            is_refusal=boolean(record['is_refusal'], '是否拒答'),
            citation_supported=boolean(record['citation_supported'], '引用是否支持答案', optional=True),
            note=record['note'],
        ))
    if not reviews:
        raise ValueError('没有填写完整的评分；空白项不会自动记为 0 分')
    updated = copy.deepcopy(result)
    for review in reviews:
        apply_review(updated, review)
    return updated, len(reviews)
