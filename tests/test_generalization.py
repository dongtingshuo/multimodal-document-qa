from pathlib import Path

from docqa.config import Settings
from docqa.parsers import LocalParser
from scripts.run_generalization import build_corpus, make_questions


def test_new_corpus_questions_are_built_from_reparsed_elements(tmp_path):
    documents = build_corpus(tmp_path)
    parser = LocalParser(Settings(data_dir=tmp_path, enable_ocr=False))
    parsed = {
        document["id"]: parser.parse(
            Path(document["path"]), document["id"], tmp_path / "parsed" / document["id"]
        )
        for document in documents
    }
    questions = make_questions(parsed)
    assert len(questions) == 12
    assert {question["category"] for question in questions} == {"text", "table", "chart", "unanswerable"}
    assert sum(not question["answerable"] for question in questions) == 1
    element_ids = {element.id for result in parsed.values() for element in result["elements"]}
    assert all(element_id in element_ids for question in questions for element_id in question["relevant_element_ids"])
    assert any(element.kind == "table" for result in parsed.values() for element in result["elements"])
    assert any(element.kind == "image" for result in parsed.values() for element in result["elements"])
