import pytest

from evals.judge import parse_verdict
from evals.metrics import (
    calc_match, citations, first_hit_rank, hit_at_k, numbers_in, percentile, recall_at_k, reciprocal_rank,
)

# each retrieved chunk lists the sections it overlaps
RETRIEVED = [["a:s1"], ["b:s2", "b:s3"], ["c:s9"], ["a:s4"]]


def test_hit_and_rank():
    gold = {"b:s3"}
    assert first_hit_rank(RETRIEVED, gold) == 2
    assert not hit_at_k(RETRIEVED, gold, 1) and hit_at_k(RETRIEVED, gold, 2)
    assert reciprocal_rank(RETRIEVED, gold) == 0.5
    assert reciprocal_rank(RETRIEVED, {"z:s1"}) == 0.0


def test_recall_counts_each_gold_section():
    gold = {"a:s1", "a:s4"}
    assert recall_at_k(RETRIEVED, gold, 2) == 0.5  # only a:s1 in top 2
    assert recall_at_k(RETRIEVED, gold, 4) == 1.0


@pytest.mark.parametrize("text, expected", [
    ("Rs 1,50,000 in total", 150000),
    ("₹1.5 lakh", 150000),
    ("about 2 crore", 20_000_000),
    ("12 days of leave", 12),
])
def test_numbers_in_indian_formats(text, expected):
    assert expected in numbers_in(text)


def test_calc_match_tolerance():
    assert calc_match("You get Rs 1,12,500.", 112500)
    assert calc_match("Rs 3,000.40", 3000)
    assert not calc_match("Rs 1,15,000", 112500)


@pytest.mark.parametrize("text, cited", [
    ("Yes [1]. Also [2][3].", [1, 2, 3]),
    ("See [1, 4].", [1, 4]),
    ("Rules [2-4] apply", [2, 3, 4]),
    ("No citations here", []),
])
def test_citations(text, cited):
    assert citations(text) == cited


def test_parse_verdict():
    assert parse_verdict('{"correctness": 2, "faithful": true, "reason": "ok"}', True) == \
        {"correctness": 2, "faithful": True, "judge_reason": "ok"}
    assert parse_verdict('{"correct_behaviour": false, "reason": "invented"}', False)["correct_behaviour"] is False
    assert "judge_error" in parse_verdict("not json", True)


def test_is_refusal():
    from evals.metrics import is_refusal

    assert is_refusal("I could not find the answer in the provided texts.")
    assert is_refusal("The exact amount cannot be calculated because the passages do not specify the formula.")
    assert not is_refusal("Yes. You are eligible for gratuity after one year [1].")


def test_kappa():
    from evals.spotcheck import kappa

    assert kappa([(2, 2), (0, 0), (1, 1)]) == 1.0
    assert kappa([(2, 0), (0, 2)]) < 0
    assert kappa([]) is None


def test_percentile():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([], 50) is None
