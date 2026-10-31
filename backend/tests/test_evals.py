import pytest

from evals.dataset import GOLDEN, HANDWRITTEN, normalize, read_jsonl, validate
from src.config import PROCESSED

SECTIONS = {"x:s1": {"id": "x:s1", "text": "The employer shall pay gratuity at the rate of fifteen days’ wages."}}


def item(**kw):
    base = {"id": "q1", "type": "rule", "question": "How much gratuity?", "answer": "15 days' wages per year.",
            "gold_sections": ["x:s1"], "answerable": True, "quote": "rate of fifteen days' wages"}
    return {**base, **kw}


def test_valid_item_passes():
    assert validate(item(), SECTIONS) == []


def test_quote_match_ignores_case_whitespace_and_quote_style():
    assert normalize("Fifteen  days’\nwages") == normalize("fifteen days' wages")
    assert validate(item(quote="RATE OF  fifteen days’ wages"), SECTIONS) == []


def test_invented_quote_is_rejected():
    assert "quote not found verbatim in the gold sections" in validate(item(quote="thirty days' wages"), SECTIONS)


def test_unknown_gold_section_is_rejected():
    assert any("unknown gold sections" in p for p in validate(item(gold_sections=["x:s9"]), SECTIONS))


def test_question_naming_a_section_is_rejected():
    assert any("section/rule number" in p for p in validate(item(question="What does section 53 say?"), SECTIONS))


def test_unanswerable_needs_no_quote_or_gold():
    oos = item(type="out_of_scope", answerable=False, gold_sections=[], quote=None)
    del oos["quote"]
    assert validate(oos, SECTIONS) == []


needs_corpus = pytest.mark.skipif(not (PROCESSED / "sections.jsonl").exists(), reason="run the parser first")


@needs_corpus
@pytest.mark.parametrize("path", [HANDWRITTEN, GOLDEN], ids=["handwritten", "golden"])
def test_dataset_files_pass_validation(path):
    from evals.dataset import load_sections

    rows = read_jsonl(path)
    if not rows:
        pytest.skip(f"{path.name} not created yet")
    sections = load_sections()
    assert {r["id"]: validate(r, sections) for r in rows if validate(r, sections)} == {}
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
