"""Draft evaluation questions with an LLM, grounded in specific sections.

Every draft must include a verbatim quote from its gold section(s); drafts whose quote is
not found, or that fail other checks, are retried once and otherwise kept with their
problems listed so the reviewer can fix or reject them. Nothing here is final: all drafts
go through human review (python -m evals.review) before entering the golden set.

Usage (from backend/):
    python -m evals.draft_questions             # writes evals/drafts.jsonl
"""

from __future__ import annotations

import json
import sys
import time

from evals.dataset import DRAFTS, load_sections, read_jsonl, validate, write_jsonl
from src.config import PROCESSED
from src.generation.llm import GeminiLLM

# One question per seed section; the style alternates between precise and everyday wording.
SINGLE = [
    "Wages Code §3", "Wages Code §5", "Wages Code §9", "Wages Code §13", "Wages Code §15", "Wages Code §17",
    "Wages Code §18", "Wages Code §20", "Wages Code §26", "Wages Code §29", "Wages Code §39", "Wages Code §45",
    "IR Code §4", "IR Code §62", "IR Code §67", "IR Code §70", "IR Code §71", "IR Code §72", "IR Code §74",
    "IR Code §83", "SS Code §53", "SS Code §55", "SS Code §56", "SS Code §57", "SS Code §59", "SS Code §60",
    "SS Code §64", "SS Code §65", "SS Code §67", "SS Code §113", "SS Code §114", "SS Code §124",
    "OSH Code §6", "OSH Code §25", "OSH Code §26", "OSH Code §28", "OSH Code §32", "OSH Code §43",
    "OSH Code §44", "Wages Rules r.5", "Wages Rules r.6", "Wages Rules r.52", "SS Rules r.33", "SS Rules r.35",
    "SS Rules r.48", "OSH Rules r.58", "OSH Rules r.64", "IR Rules r.25", "IR Rules r.27", "IR Rules r.37",
]

# Questions that need both sections to answer fully.
PAIRS = [
    ("SS Code §53", "SS Rules r.33"), ("Wages Code §14", "OSH Code §27"), ("OSH Code §25", "Wages Rules r.5"),
    ("IR Code §70", "IR Rules r.27"), ("IR Code §62", "IR Rules r.25"), ("SS Code §60", "SS Code §64"),
    ("SS Code §67", "OSH Rules r.58"), ("Wages Code §16", "Wages Code §17"), ("SS Code §113", "SS Rules r.48"),
    ("Wages Code §26", "Wages Code §29"), ("IR Code §67", "IR Code §69"), ("OSH Code §32", "OSH Code §25"),
    ("IR Code §74", "IR Code §75"), ("Wages Code §18", "Wages Code §20"), ("OSH Code §43", "OSH Code §28"),
]

# (source_id, defined term)
DEFINITIONS = [
    ("industrial_relations_code_2020", "fixed term employment"), ("social_security_code_2020", "gig worker"),
    ("social_security_code_2020", "platform worker"), ("social_security_code_2020", "aggregator"),
    ("social_security_code_2020", "platform work"), ("industrial_relations_code_2020", "retrenchment"),
    ("industrial_relations_code_2020", "lay-off"), ("industrial_relations_code_2020", "strike"),
    ("industrial_relations_code_2020", "worker"), ("code_on_wages_2019", "employee"),
    ("osh_code_2020", "inter-state migrant worker"), ("social_security_code_2020", "unorganised worker"),
    ("osh_code_2020", "contract labour"), ("code_on_wages_2019", "same work or work of a similar nature"),
    ("social_security_code_2020", "family"),
]

SYSTEM = """You write test questions for a question-answering system about India's labour codes.
Each question must be fully answerable from the text you are given, by someone who has not seen it.
Return JSON only."""

STYLES = {
    "rule": "Ask precisely, in plain English (like a careful HR person or employee would).",
    "everyday": ("Ask the way an ordinary worker or small employer would: casual, first person, maybe "
                 "with a short personal situation. Use everyday words, not the legal wording of the text."),
}

TEMPLATE = """{instruction}

Rules:
- Do NOT mention section, rule or clause numbers in the question, and don't copy long phrases from the text.
- "answer": a correct, self-contained answer in 1-3 sentences, based only on the text.
- "quote": an EXACT, verbatim excerpt of the text (one continuous span, 10-60 words) that supports the answer.
  Copy it character for character from the text, including punctuation.

Return a JSON object: {{"question": "...", "answer": "...", "quote": "..."}}

{texts}"""


def section_block(s: dict) -> str:
    title = f" ({s['title']})" if s.get("title") else ""
    return f"=== {s['label']}{title} ===\n{s['text'][:6000]}"


def draft_one(llm: GeminiLLM, prompt: str, item: dict, sections: dict, retries: int = 1) -> dict:
    feedback = ""
    for attempt in range(retries + 1):
        try:
            raw, model = llm.generate(prompt + feedback, system=SYSTEM, json_mode=True)
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
        except Exception as e:
            item["problems"] = [f"generation failed: {str(e)[:200]}"]
            return item
        item.update(question=data.get("question", "").strip(), answer=data.get("answer", "").strip(),
                    quote=data.get("quote", "").strip(), model=model)
        item["problems"] = validate(item, sections)
        if not item["problems"]:
            break
        feedback = ("\n\nYour previous attempt had these problems: " + "; ".join(item["problems"])
                    + ". Fix them; the quote must be copied exactly from the text.")
    return item


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sections = load_sections()
    by_label = {s["label"]: s for s in sections.values()}
    defs = {(d["source_id"], d["term"]): d for d in read_jsonl(PROCESSED / "definitions.jsonl")}
    pib = [s for s in sections.values() if s["source_id"] == "pib_labour_codes_explainer_2025"]

    missing = [l for l in SINGLE + [x for p in PAIRS for x in p] if l not in by_label]
    missing += [f"definition {k}" for k in DEFINITIONS if k not in defs]
    if missing:
        raise SystemExit(f"unknown seeds: {missing}")

    jobs: list[tuple[dict, str]] = []
    for i, label in enumerate(SINGLE):
        style = "rule" if i % 2 == 0 else "everyday"
        s = by_label[label]
        jobs.append(({"id": f"single{i + 1:02d}", "type": style, "gold_sections": [s["id"]], "answerable": True},
                     TEMPLATE.format(instruction=f"Write ONE question answered by this text. {STYLES[style]}",
                                     texts=section_block(s))))
    for i, (a, b) in enumerate(PAIRS):
        sa, sb = by_label[a], by_label[b]
        jobs.append(({"id": f"multi{i + 1:02d}", "type": "multi_section", "gold_sections": [sa["id"], sb["id"]],
                      "answerable": True},
                     TEMPLATE.format(
                         instruction=("Write ONE question whose complete answer needs information from BOTH texts "
                                      "below (e.g. a right from one and the procedure or time limit from the other). "
                                      "The quote may come from either text. " + STYLES["rule"]),
                         texts=section_block(sa) + "\n\n" + section_block(sb))))
    for i, key in enumerate(DEFINITIONS):
        d = defs[key]
        jobs.append(({"id": f"def{i + 1:02d}", "type": "definition", "gold_sections": [d["section_id"]],
                      "gold_definition": d["label"], "answerable": True},
                     TEMPLATE.format(
                         instruction=(f"Write ONE question asking who or what counts as '{d['term']}' (or whether "
                                      f"some situation falls within it). {STYLES['everyday']}"),
                         texts=f"=== {d['label']} (definition of '{d['term']}') ===\n{d['text']}")))
    for i, s in enumerate(pib):
        jobs.append(({"id": f"changed{i + 1:02d}", "type": "what_changed", "gold_sections": [s["id"]],
                      "answerable": True},
                     TEMPLATE.format(
                         instruction=("Write ONE question about what is new or what changed for workers or "
                                      "employers under the labour codes, answered by this government explainer. "
                                      + STYLES["everyday"]),
                         texts=section_block(s))))

    llm = GeminiLLM(retries=3)
    done = {d["id"]: d for d in read_jsonl(DRAFTS) if not d.get("problems")}  # resume: keep good drafts
    out = []
    for n, (item, prompt) in enumerate(jobs, 1):
        if item["id"] in done:
            out.append(done[item["id"]])
            continue
        item = draft_one(llm, prompt, item, sections)
        out.append(item)
        flag = "ok " if not item["problems"] else "!! "
        print(f"[{n:3}/{len(jobs)}] {flag}{item['id']:10} {item.get('question', '')[:90]}"
              + (f"\n          problems: {item['problems']}" if item["problems"] else ""))
        write_jsonl(DRAFTS, out + [done[j["id"]] for j, _ in jobs[n:] if j["id"] in done])  # save progress
        time.sleep(2)  # stay under free-tier rate limits

    ok = sum(not d["problems"] for d in out)
    print(f"\n{ok}/{len(out)} drafts pass automatic checks. Wrote {DRAFTS.name}")


if __name__ == "__main__":
    main()
