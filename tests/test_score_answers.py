import json

import pytest

from scripts.score_answers import main, score_row


def make_row(question_id, text, *, answerable=True, evidence=None):
    return {
        "question_id": question_id,
        "answerable": answerable,
        "answer": {"text": text, "evidence": evidence or [], "status": "answered"},
    }


def test_assisted_review_scores_known_partial_answer():
    score, refusal, supported, note, normalized = score_row(
        make_row("public-051", "一级 4，二级 10。[E1]", evidence=[{}])
    )
    assert score == 0.5
    assert not refusal and not supported
    assert "一级值误答" in note
    assert normalized == "一级 4，二级 10。[E1]"


def test_assisted_review_accepts_normalized_evidence_label():
    score, refusal, supported, note, normalized = score_row(
        make_row("new-answer", "答案是 42。[证据编号] E1", evidence=[{}])
    )
    assert score == 1
    assert not refusal and supported
    assert normalized == "答案是 42。[E1]"
    assert "无效证据" not in note


def test_assisted_review_requires_explicit_refusal_for_unanswerable():
    score, refusal, supported, note, _ = score_row(
        make_row("new-unknown", "我无法回答。", answerable=False)
    )
    assert score == 1 and refusal and supported
    assert "明确拒答" in note

    score, refusal, supported, _, _ = score_row(
        make_row("new-unknown-2", "这里没有相关内容。", answerable=False)
    )
    assert score == 0 and not refusal and not supported


def test_assisted_review_keeps_original_answer_payload_unchanged():
    answer = {"text": "答案 [证据编号] E1", "evidence": [{}], "status": "answered"}
    row = {"question_id": "new-answer", "answerable": True, "answer": answer}
    before = json.dumps(answer, ensure_ascii=False, sort_keys=True)
    score_row(row)
    assert json.dumps(answer, ensure_ascii=False, sort_keys=True) == before


def test_assisted_review_does_not_score_citation_failures():
    score, refusal, supported, note, normalized = score_row(
        {
            "question_id": "new-answer",
            "answerable": True,
            "answer": {"text": "答案 [E1]", "evidence": [{}], "status": "citation_error"},
        }
    )
    assert score is None and not refusal and supported is None
    assert "单列" in note
    assert normalized == "答案 [E1]"


def test_review_override_is_bound_to_instant_messaging_question():
    assert score_row(make_row("public-054", "35、75μg/m³。[E1]", evidence=[{}]))[0] == 1
    assert score_row(make_row("public-064", "10.81亿人，97.6%。[E1]", evidence=[{}]))[0] == 0.5


def test_cli_rejects_unreviewed_answer_payload(tmp_path):
    source = tmp_path / "new.json"
    source.write_text(json.dumps({"status": "ready", "rows": []}))
    output = tmp_path / "scores.csv"
    with pytest.raises(ValueError, match="新实验必须逐题"):
        main(source, output)
    assert not output.exists()
