import copy
import csv
import io

import pytest

from docqa.review_io import export_reviews, import_reviews


def result_fixture():
    return {'id': 'run', 'status': 'ready', 'rows': [
        {'question_id': f'q{i}', 'group': 'E', 'question': '=危险公式' if i == 0 else '数值？',
         'reference_answer': '18', 'strategy': 'hybrid_reranker', 'category': 'text', 'answerable': True,
         'answer_accuracy': None, 'recall@1': 1, 'recall@3': 1, 'recall@5': 1, 'recall@10': 1,
         'mrr@10': 1, 'seconds': 1, 'answer': {'status': 'answered', 'text': '18 [E1]'}}
        for i in range(2)
    ]}


def sheet_bytes(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def read_sheet(result):
    return list(csv.DictReader(io.StringIO(export_reviews(result).decode('utf-8-sig'))))


def test_blank_worksheet_is_not_scored_and_formula_is_escaped():
    result = result_fixture()
    rows = read_sheet(result)
    assert rows[0]['question'].startswith("'=")
    assert rows[0]['answer_accuracy'] == rows[0]['is_refusal'] == ''
    with pytest.raises(ValueError, match='没有填写完整'):
        import_reviews(result, export_reviews(result))
    assert all(row['answer_accuracy'] is None for row in result['rows'])


def test_import_is_atomic_and_bound_to_original_answer():
    result = result_fixture()
    before = copy.deepcopy(result)
    rows = read_sheet(result)
    rows[0].update(reviewer='评审', answer_accuracy='1', is_refusal='0', citation_supported='1')
    rows[1].update(reviewer='评审', answer_accuracy='invalid', is_refusal='0')
    with pytest.raises(ValueError):
        import_reviews(result, sheet_bytes(rows))
    assert result == before
    rows[1].update(answer_accuracy='0.5')
    changed = copy.deepcopy(rows)
    changed[0]['answer'] = '改写答案'
    with pytest.raises(ValueError, match='原始结果不一致'):
        import_reviews(result, sheet_bytes(changed))
    updated, count = import_reviews(result, sheet_bytes(rows))
    assert count == 2 and updated['summary'][0]['answer_accuracy'] == 0.75
    assert updated['summary'][0]['citation_reviewed'] == 1
    assert result == before


def test_duplicate_and_wrong_experiment_rejected():
    result = result_fixture()
    rows = read_sheet(result)
    rows[0].update(reviewer='reviewer', answer_accuracy='1', is_refusal='0')
    with pytest.raises(ValueError, match='重复'):
        import_reviews(result, sheet_bytes([rows[0], rows[0]]))
    rows[0]['experiment_id'] = 'another'
    with pytest.raises(ValueError, match='不属于'):
        import_reviews(result, sheet_bytes(rows))
