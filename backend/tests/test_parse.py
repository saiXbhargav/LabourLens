import json
import re

import pytest

from src.ingest.parse import (
    FORM,
    OUT,
    RULE_TITLE,
    extract_definitions,
    heading_case,
    is_next_number,
    join_lines,
)


# ---------------------------------------------------------------- unit tests


@pytest.mark.parametrize(
    "num, prev, expected",
    [
        ("1", None, True),
        ("2", None, False),
        ("54", "53", True),
        ("55", "53", True),  # one omitted section is tolerated
        ("60", "53", False),  # "60." at a line start is probably not a section
        ("7A", "7", True),
        ("7B", "7A", True),
        ("7A", "8", False),
    ],
)
def test_is_next_number(num, prev, expected):
    assert is_next_number(num, prev) is expected


@pytest.mark.parametrize(
    "text, title",
    [
        ("Short title and commencement.– (1) These rules", "Short title and commencement"),
        ("Hours of work for normal working day. - (1) The number", "Hours of work for normal working day"),
        ("Definitions.-(1) In these rules", "Definitions"),
    ],
)
def test_rule_title(text, title):
    assert RULE_TITLE.match(text).group(1) == title


def test_rule_title_ignores_hyphenated_words():
    m = RULE_TITLE.match("Notice under sub-section (2) of section 5 shall be given")
    assert m is None


def test_form_heading_only_matches_standalone_heading():
    assert FORM.match("FORM I")
    assert FORM.match("FORM-XII")
    assert not FORM.match("Form-XII;")
    assert not FORM.match("FORM I shall be submitted")


def test_join_lines_paragraphs_and_hyphens():
    text = join_lines(["(1) An appeal under sub-", "section (2) shall lie", "(2) The appeal", "Provided that it is filed."])
    assert text == "(1) An appeal under sub-section (2) shall lie\n(2) The appeal\nProvided that it is filed."


def test_heading_case():
    assert heading_case("DUTIES OF EMPLOYER AND EMPLOYEES, ETC.") == "Duties of Employer and Employees, etc."
    assert heading_case("EMPLOYEE’S COMPENSATION") == "Employee’s Compensation"


def test_extract_definitions_lettered_with_nested_items():
    section = {
        "id": "x:s2", "source_id": "x", "label": "Wages Code §2",
        "text": '\n'.join([
            "In this Code, unless the context otherwise requires,—",
            '(y) "wages" means all remuneration and includes—',
            "(i) basic pay;",
            "(a) any bonus payable under any law",
            '(z) "worker" means any person employed',
        ]),
    }
    defs = extract_definitions(section)
    assert [d["term"] for d in defs] == ["wages", "worker"]
    assert "basic pay" in defs[0]["text"] and "any bonus" in defs[0]["text"]
    assert defs[0]["label"] == "Wages Code §2(y)"


def test_extract_definitions_numbered_and_stops_at_subsection_2():
    numbered = {"id": "s:s2", "source_id": "s", "label": "SS Code §2",
                "text": '(1) "agent" means a person\n(2) "aggregator" means a digital intermediary'}
    assert [d["clause"] for d in extract_definitions(numbered)] == ["1", "2"]

    rules = {"id": "r:r2", "source_id": "r", "label": "IR Rules r.2",
             "text": "(1) In these rules,—\n(a) “Code” means the Code\n(2) Words not defined here have the Code's meaning"}
    defs = extract_definitions(rules)
    assert [d["term"] for d in defs] == ["code"]
    assert "Words not defined" not in defs[0]["text"]


# ---------------------------------------------------------------- output checks

SECTIONS = OUT / "sections.jsonl"
needs_output = pytest.mark.skipif(not SECTIONS.exists(), reason="run `python -m src.ingest.parse` first")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(l) for l in SECTIONS.open(encoding="utf-8")]


@pytest.fixture(scope="module")
def by_label(rows):
    return {r["label"]: r for r in rows}


@needs_output
@pytest.mark.parametrize(
    "source_id, count",
    [
        ("code_on_wages_2019", 69),
        ("industrial_relations_code_2020", 104),
        ("social_security_code_2020", 164),
        ("osh_code_2020", 143),
        ("ss_central_rules_2026", 69),
    ],
)
def test_section_counts_match_official_texts(rows, source_id, count):
    numbers = [r["number"] for r in rows if r["source_id"] == source_id and r["kind"] in ("section", "rule")]
    assert len(numbers) == count
    assert numbers[-1] == str(count)


@needs_output
def test_gratuity_section(by_label):
    s = by_label["SS Code §53"]
    assert s["title"] == "Payment of gratuity"
    assert s["chapter_title"] == "Gratuity"
    assert "five years" in s["text"] and "fixed term employment" in s["text"]
    assert s["subsections"][0]["number"] == "1"


@needs_output
def test_margin_notes_do_not_leak_into_text(by_label):
    s = by_label["SS Code §29"]
    assert s["title"] == "Contributions"
    assert "shall comprise contribution" in s["text"]


@needs_output
def test_no_noise_in_text(rows):
    noise = re.compile(r"[ऀ-ॿ�]|GAZETTE OF INDIA|[ﬁﬂ]")
    assert [r["label"] for r in rows if noise.search(r["text"])] == []


@needs_output
def test_ids_unique(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


@needs_output
def test_wages_definition_lists_components():
    defs = [json.loads(l) for l in (OUT / "definitions.jsonl").open(encoding="utf-8")]
    wages = next(d for d in defs if d["label"] == "Wages Code §2(y)")
    assert wages["term"] == "wages"
    assert "basic pay" in wages["text"] and "house rent allowance" in wages["text"]
