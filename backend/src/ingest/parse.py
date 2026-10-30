"""Parse the downloaded labour codes and central rules into structured records.

Output (data/processed/):
    sections.jsonl     one record per section (codes), rule (rules) or schedule
    definitions.jsonl  one record per defined term (section 2 / rule 2)

Usage (from backend/):
    python -m src.ingest.parse
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

BACKEND = Path(__file__).resolve().parents[2]
DATA = BACKEND / "data"
RAW = DATA / "raw"
OUT = DATA / "processed"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HEADER = re.compile(
    r"^(\d+|THE GAZETTE OF INDIA.*|\d+\s+THE GAZETTE OF INDIA.*|\[PART II.*|PART II\s*[—–-]\s*SEC.*"
    r"|SEC\.\s*\d+(\(\w+\))?\]?)$"
)
FOOTER = re.compile(r"^(_{3,}|[—–-]{3,}|UPLOADED BY.*|.*SECRETARY TO THE GOVT.*)$", re.I)
CHAPTER = re.compile(r"^CHAPTER\s+([IVXLC]+[A-Z]?)$")
PART = re.compile(r"^PART\s+([IVX]+)$")  # divisions inside a chapter (e.g. OSH Code ch. XI)
SCHEDULE = re.compile(r"^(THE\s+)?((FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+)?SCHEDULE(\s*[-–—]?\s*[IVX\d]+)?$")
# Only a standalone heading like "FORM I" / "FORM-XII" starts the forms; "Form-XII;" mid-sentence does not.
FORM = re.compile(r"^FORM\s*[-–—]?\s*[IVXLC\d]+[A-Z]?$")
SIGNATURE = re.compile(r"^\[(F\.\s*No|No\.|File No)", re.I)
SECTION_START = re.compile(r"^(\d+)([A-Z]{0,2})\.\s*(.*)$")
UPPER_TITLE = re.compile(r"^[A-Z0-9 ,.'’&()/\-–—]+$")
MARGIN_ACT_REF = re.compile(r"^\d+\s+of\s+\d{4}\.?$")
PARA_START = re.compile(r"^(\(\w{1,5}\)|Provided|Explanation|Illustration|\d{1,3}\.\s+[A-Z])")
LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "etc"}
SUBSECTION = re.compile(r"^\((\d+[A-Z]?)\)\s*")
# "Title.–", "Title. -", "Title.-" or "Title –"; a bare hyphen needs the full stop (so "sub-section" is safe).
RULE_TITLE = re.compile(r"^(.{3,250}?)(?:\.\s*[-–—]|\s*[–—])\s*")
# Codes letter their definitions (a), (b)...; the SS Code numbers them (1), (2)...
DEFINITION = re.compile(r"^\(([a-z]{1,3}|\d{1,3})\)\s*[“\"]([^”\"]{1,80})[”\"]")
# Some gazette fonts render quotes as ―term‖ and apostrophes as ‘
ODD_QUOTES = re.compile(r"―([^‖―]{1,80})‖")

# Body text of the codes sits at x≈117–479 on an A4 page; margin notes are outside
# (left notes reach x≈107, right notes start at x≈485).
CODE_MARGIN_LEFT, CODE_MARGIN_RIGHT = 112, 481


@dataclass
class Line:
    page: int  # 0-based
    x0: float
    y0: float
    x1: float
    text: str


@dataclass
class Record:
    source_id: str
    kind: str  # "section" | "rule" | "schedule" | "explainer"
    number: str
    chapter: str | None
    chapter_title: str | None
    start: Line
    lines: list[Line] = field(default_factory=list)
    title: str | None = None
    part: str | None = None
    part_title: str | None = None


# ---------------------------------------------------------------- text helpers


def page_lines(page: pymupdf.Page, split_margin: bool = False) -> list[Line]:
    """Lines in reading order. With split_margin, spans in the margin columns become their
    own lines: PyMuPDF sometimes merges a margin note into the body line level with it."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            groups: dict[str, list[dict]] = {}
            for span in line["spans"]:
                x0, _, x1, _ = span["bbox"]
                zone = "body"
                if split_margin and x0 >= CODE_MARGIN_RIGHT:
                    zone = "right"
                elif split_margin and x1 <= CODE_MARGIN_LEFT:
                    zone = "left"
                groups.setdefault(zone, []).append(span)
            for spans in groups.values():
                text = "".join(s["text"] for s in spans).strip()
                text = ODD_QUOTES.sub(r"“\1”", text).replace("‘s", "’s").translate(LIGATURES)
                if text:
                    x0 = min(s["bbox"][0] for s in spans)
                    x1 = max(s["bbox"][2] for s in spans)
                    lines.append(Line(page.number, x0, line["bbox"][1], x1, re.sub(r"\s+", " ", text)))
    return sorted(lines, key=lambda l: (round(l.y0 / 3), l.x0))


def is_noise(line: Line) -> bool:
    if DEVANAGARI.search(line.text):
        return True
    if line.y0 < 100 and HEADER.match(line.text):
        return True
    return bool(FOOTER.match(line.text))


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into paragraphs: new paragraph at (1)/(a)/Provided/Explanation."""
    paras: list[str] = []
    for text in lines:
        if not paras or PARA_START.match(text):
            paras.append(text)
        elif paras[-1].endswith("-"):
            paras[-1] += text  # "sub-" + "section"
        else:
            paras[-1] += " " + text
    return "\n".join(p.strip() for p in paras)


def split_number(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z]*)", num)
    return int(m.group(1)), m.group(2)


def is_next_number(num: str, prev: str | None) -> bool:
    """Guard against lines that merely start with 'N.': numbers must go up in order."""
    n, suffix = split_number(num)
    if prev is None:
        return n == 1 and not suffix
    p, psuffix = split_number(prev)
    if suffix:
        return n == p and (not psuffix or suffix > psuffix)
    return p < n <= p + 2


def clean_title(parts: list[str]) -> str:
    title = join_lines(parts).replace("\n", " ")
    return re.sub(r"\s+", " ", title).strip().rstrip(".:").strip()


def heading_case(text: str | None) -> str | None:
    """'DUTIES OF EMPLOYER AND EMPLOYEES, ETC.' -> 'Duties of Employer and Employees, etc.'"""
    if not text:
        return None
    words = text.lower().split()
    return " ".join(
        w if i and w.strip(",.()") in SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words)
    )


# ---------------------------------------------------------------- parsers


def parse_codes(doc: pymupdf.Document, source_id: str) -> list[Record]:
    """Acts in gazette layout: section titles are margin notes beside the section start."""
    body: list[Line] = []
    margin: list[Line] = []
    started = False
    for page in doc:
        for line in page_lines(page, split_margin=True):
            if is_noise(line):
                continue
            if line.x0 >= CODE_MARGIN_RIGHT or line.x1 <= CODE_MARGIN_LEFT:
                margin.append(line)
                continue
            if not started:
                started = bool(CHAPTER.match(line.text)) and line.text.endswith(" I")
            if started:
                body.append(line)

    records = parse_body(body, source_id, kind="section", start_x_max=150)
    assign_margin_titles(records, margin)
    return records


def parse_rules(doc: pymupdf.Document, source_id: str, gsr: str) -> list[Record]:
    """Bilingual gazette rules: skip the Hindi half, start at the English G.S.R. notification."""
    gsr_re = re.compile(r"G\.\s*S\.\s*R\.\s*" + re.escape(gsr.split()[1].split("(")[0]) + r"\s*\(E\)")
    lines: list[Line] = []
    found_gsr = False
    for page in doc:
        for line in page_lines(page):
            if not found_gsr:
                found_gsr = bool(gsr_re.search(line.text)) and not DEVANAGARI.search(line.text)
                continue
            if not is_noise(line):
                lines.append(line)
    # Rule 1 is always "Short title..."; the preamble may contain its own numbered list of
    # superseded rules, so don't just take the first "1.". Include a CHAPTER I heading just above it.
    start = next(i for i, l in enumerate(lines) if re.match(r"^1\.\s*Short\s+title", l.text, re.I))
    for back in range(max(0, start - 3), start):
        if CHAPTER.match(lines[back].text) and lines[back].text.endswith(" I"):
            start = back
            break
    body = lines[start:]
    left = statistics.median(l.x0 for l in body[:200])
    records = parse_body(body, source_id, kind="rule", start_x_max=left + 20)
    for rec in records:
        if rec.kind == "rule":
            text = join_lines([l.text for l in rec.lines])
            m = RULE_TITLE.match(text[:300])
            if m and not text.startswith("("):
                rec.title = clean_title([m.group(1)])
            elif "\n" in text and len(first := text.split("\n", 1)[0]) < 200 and first.endswith("."):
                rec.title = clean_title([first])  # title on its own line, no dash
    return records


def parse_body(lines: list[Line], source_id: str, kind: str, start_x_max: float) -> list[Record]:
    records: list[Record] = []
    chapter = chapter_title = part = part_title = None
    current: Record | None = None
    prev_num: str | None = None
    reading = None  # "chapter" / "part" while collecting an all-caps heading
    skipping = False  # after a signature block, until the next schedule

    for line in lines:
        text = line.text
        if FORM.match(text):
            break  # forms appended to rules are templates, not law
        if m := CHAPTER.match(text):
            chapter, chapter_title, part, part_title = m.group(1), None, None, None
            reading, current = "chapter", None
            continue
        if (m := PART.match(text)) and not (current and current.kind == "schedule"):
            part, part_title, reading, current = m.group(1), None, "part", None
            continue
        if reading:
            if UPPER_TITLE.match(text) and not SECTION_START.match(text):
                if reading == "chapter":
                    chapter_title = f"{chapter_title} {text}" if chapter_title else text
                else:
                    part_title = f"{part_title} {text}" if part_title else text
                continue
            reading = None
        if SCHEDULE.match(text.upper()) and len(text) < 40:
            skipping = False
            current = Record(source_id, "schedule", text.title(), None, None, line)
            records.append(current)
            continue
        if SIGNATURE.match(text):
            skipping, current = True, None
            continue
        if skipping:
            continue
        m = SECTION_START.match(text)
        if m and line.x0 <= start_x_max and is_next_number(m.group(1) + m.group(2), prev_num):
            prev_num = m.group(1) + m.group(2)
            current = Record(source_id, kind, prev_num, chapter, chapter_title, line,
                             part=part, part_title=part_title)
            if m.group(3):
                current.lines.append(Line(line.page, line.x0, line.y0, line.x1, m.group(3)))
            records.append(current)
            continue
        if current is not None:
            current.lines.append(line)
    return records


def assign_margin_titles(records: list[Record], margin: list[Line]) -> None:
    """Group margin lines into notes and attach each title note to the section beside it."""
    notes: list[list[Line]] = []
    open_note: dict[tuple[int, bool], list[Line]] = {}  # (page, is_left_side) -> note in progress
    for line in margin:
        key = (line.page, line.x1 <= CODE_MARGIN_LEFT)
        note = open_note.get(key)
        # Every margin note ends with a full stop; notes on the same side can be packed tightly.
        if note and not note[-1].text.endswith(".") and 0 <= line.y0 - note[-1].y0 < 14:
            note.append(line)
        else:
            open_note[key] = [line]
            notes.append(open_note[key])
    by_page: dict[int, list[Record]] = {}
    for rec in records:
        if rec.kind == "section":
            by_page.setdefault(rec.start.page, []).append(rec)
    for note in notes:
        text = clean_title([l.text for l in note])
        if not text or MARGIN_ACT_REF.match(text + "."):
            continue
        candidates = [r for r in by_page.get(note[0].page, []) if abs(r.start.y0 - note[0].y0) < 20]
        if candidates:
            rec = min(candidates, key=lambda r: abs(r.start.y0 - note[0].y0))
            rec.title = rec.title or text


def parse_explainer(doc: pymupdf.Document, source_id: str) -> list[Record]:
    records = []
    for page in doc:
        lines = [l for l in page_lines(page) if not is_noise(l)]
        if lines:
            rec = Record(source_id, "explainer", f"p{page.number + 1}", None, None, lines[0], lines)
            rec.title = f"Page {page.number + 1}"
            records.append(rec)
    return records


# ---------------------------------------------------------------- output


def to_json(rec: Record, src: dict) -> dict:
    text = join_lines([l.text for l in rec.lines])
    if rec.kind == "rule" and rec.title:  # title is stored separately
        if RULE_TITLE.match(text[:300]):
            text = RULE_TITLE.sub("", text, count=1)
        else:
            text = text.split("\n", 1)[1]
    subsections = []
    for para in text.split("\n"):
        if m := SUBSECTION.match(para):
            subsections.append({"number": m.group(1), "text": para})
        elif subsections:
            subsections[-1]["text"] += "\n" + para
    prefix = {"section": "s", "rule": "r", "schedule": "", "explainer": ""}[rec.kind]
    number_slug = re.sub(r"\W+", "_", rec.number.lower()).strip("_")
    label = {
        "section": f"{src['short']} §{rec.number}",
        "rule": f"{src['short']} r.{rec.number}",
        "schedule": f"{src['short']}, {rec.number}",
        "explainer": f"{src['short']}, {rec.number}",
    }[rec.kind]
    pages = sorted({rec.start.page + 1} | {l.page + 1 for l in rec.lines})
    return {
        "id": f"{src['id']}:{prefix}{number_slug}",
        "source_id": src["id"],
        "source_title": src["title"],
        "source_kind": src["kind"],
        "kind": rec.kind,
        "number": rec.number,
        "label": label,
        "title": rec.title,
        "chapter": rec.chapter,
        "chapter_title": heading_case(rec.chapter_title),
        "part": rec.part,
        "part_title": heading_case(rec.part_title),
        "text": text,
        "subsections": subsections,
        "pages": [pages[0], pages[-1]],
        "url": src["url"],
    }


def extract_definitions(section: dict) -> list[dict]:
    """Split section 2 / rule 2 into one entry per quoted term."""
    defs: list[dict] = []
    for para in section["text"].split("\n"):
        m = DEFINITION.match(para)
        if not m and (s := SUBSECTION.match(para)) and s.group(1) != "1" and defs:
            break  # "(2) Words and expressions used..." ends the definitions
        if m:
            defs.append({
                "term": m.group(2).strip().lower(),
                "clause": m.group(1),
                "label": f"{section['label']}({m.group(1)})",
                "source_id": section["source_id"],
                "section_id": section["id"],
                "text": para,
            })
        elif defs:
            defs[-1]["text"] += "\n" + para
    return defs


def section_gaps(numbers: list[str]) -> list[int]:
    ints = sorted({split_number(n)[0] for n in numbers})
    return [n for n in range(1, ints[-1] + 1) if n not in set(ints)] if ints else []


def main() -> None:
    manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    all_sections, all_defs = [], []

    print(f"{'source':34} {'kind':9} {'items':>5} {'last':>5} {'titled':>7} {'sched':>5} {'defs':>5}  gaps")
    for src in manifest["sources"]:
        path = RAW / f"{src['id']}.pdf"
        if not path.exists() or src["kind"] == "old_act":
            continue
        with pymupdf.open(path) as doc:
            if src["kind"] == "code":
                records = parse_codes(doc, src["id"])
            elif src["kind"] == "rules":
                records = parse_rules(doc, src["id"], src["citation"])
            else:
                records = parse_explainer(doc, src["id"])
        sections = [to_json(r, src) for r in records]
        defs = [d for s in sections if s["kind"] in ("section", "rule") and s["number"] == "2"
                for d in extract_definitions(s)]
        all_sections += sections
        all_defs += defs

        main_items = [s for s in sections if s["kind"] in ("section", "rule")]
        titled = sum(1 for s in main_items if s["title"])
        gaps = section_gaps([s["number"] for s in main_items])
        print(
            f"{src['id']:34} {src['kind']:9} {len(main_items) or len(sections):>5} "
            f"{main_items[-1]['number'] if main_items else '-':>5} "
            f"{titled:>3}/{len(main_items):<3} {sum(s['kind'] == 'schedule' for s in sections):>5} "
            f"{len(defs):>5}  {gaps[:10] if gaps else ''}"
        )

    write_jsonl(OUT / "sections.jsonl", all_sections)
    write_jsonl(OUT / "definitions.jsonl", all_defs)
    print(f"\nWrote {len(all_sections)} records and {len(all_defs)} definitions to {OUT.relative_to(BACKEND)}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
