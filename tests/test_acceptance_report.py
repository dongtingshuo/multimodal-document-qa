import copy

import pytest
from test_experiment_resume import fixture_data

from docqa.evaluation import run_experiment
from scripts.summarize_local_acceptance import verify_run


@pytest.mark.parametrize("corruption", ["missing", "metric", "summary", "duplicate"])
def test_report_rejects_incomplete_or_inconsistent_archive(tmp_path, corruption):
    settings, store, retriever, request, _ = fixture_data(tmp_path)
    request.groups = ["D", "E"]
    request.generate = False
    result = {"status": "ready", **run_experiment(request, retriever, store)}
    questions = request.model_dump(mode="json")["questions"]
    verify_run(result, questions, request.split, ["D", "E"], False)
    broken = copy.deepcopy(result)
    if corruption == "missing":
        broken["rows"].pop()
    elif corruption == "metric":
        broken["rows"][0]["recall@5"] = 0.123
    elif corruption == "summary":
        broken["summary"][0]["mrr@10"] = 0.123
    else:
        broken["rows"][-1] = copy.deepcopy(broken["rows"][0])
    with pytest.raises(ValueError):
        verify_run(broken, questions, request.split, ["D", "E"], False)
