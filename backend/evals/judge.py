"""LLM-as-judge prompts. The judge sees the reference answer and the gold quotes (for
correctness) and the retrieved passages (for faithfulness)."""

from __future__ import annotations

import json

from evals.dataset import item_quotes

JUDGE_SYSTEM = """You grade answers produced by a question-answering system about India's labour codes.
Be strict and consistent. Judge only what is asked. Return JSON only."""

ANSWERABLE = """Question:
{question}

Reference answer (written and checked by a human):
{reference}

Supporting quotes from the law:
{quotes}

Passages the system was given:
{passages}

System's answer:
{answer}

Grade the system's answer:
- "correctness": 2 if it states all key facts of the reference (numbers, conditions, exceptions) with nothing that
  contradicts it; 1 if it is partly right (a key fact or condition missing, or a minor error); 0 if it is wrong,
  contradicts the reference, or fails to answer (e.g. says the information is not available).
  Extra correct detail is fine. Wording does not matter, substance does.
- "faithful": true if every factual claim in the answer is supported by the passages the system was given; false if
  it states anything not found in those passages (even if true in reality).
- "reason": one short sentence.

Return: {{"correctness": 0|1|2, "faithful": true|false, "reason": "..."}}"""

UNANSWERABLE = """Question:
{question}

This question cannot be answered from the texts the system has (central labour codes and central rules).
Expected behaviour: {behaviour}

System's answer:
{answer}

Grade it:
- "correct_behaviour": true if the answer does what is expected; false if it gives a specific answer, figure or
  rule that the texts do not contain (an invented or outside-knowledge answer), or otherwise fails the expectation.
- "reason": one short sentence.

Return: {{"correct_behaviour": true|false, "reason": "..."}}"""

BEHAVIOURS = {
    "refuse": "say that the information is not in the provided texts, without inventing an answer",
    "explain_not_specified": ("explain that the texts leave this to be notified / do not state the figure, "
                              "without inventing a figure"),
}


def judge_prompt(item: dict, answer: str, passages: str) -> str:
    if item["answerable"]:
        quotes = "\n".join(f"- {q}" for q in item_quotes(item)) or "(none)"
        return ANSWERABLE.format(question=item["question"], reference=item["answer"], quotes=quotes,
                                 passages=passages, answer=answer)
    return UNANSWERABLE.format(question=item["question"],
                               behaviour=BEHAVIOURS[item.get("expected_behavior", "refuse")], answer=answer)


def parse_verdict(raw: str, answerable: bool) -> dict:
    """Normalise the judge's JSON; a malformed verdict is recorded rather than crashing the run."""
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            v = v[0]
    except (json.JSONDecodeError, IndexError):
        return {"judge_error": raw[:200]}
    if answerable:
        score = v.get("correctness")
        return {"correctness": int(score) if score in (0, 1, 2, "0", "1", "2") else None,
                "faithful": bool(v["faithful"]) if "faithful" in v else None,
                "judge_reason": v.get("reason", "")}
    ok = v.get("correct_behaviour", v.get("correct_behavior"))
    return {"correct_behaviour": bool(ok) if ok is not None else None, "judge_reason": v.get("reason", "")}
