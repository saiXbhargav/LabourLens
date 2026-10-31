"""Baseline chunking: fixed-size token windows over each document, ignoring its structure.

Each chunk still records which sections it overlaps (by character span), so retrieval can be
scored against the sections a question needs, even though the chunk boundaries ignore them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

# text -> list of (char_start, char_end) for each token, without special tokens
TokenOffsets = Callable[[str], list[tuple[int, int]]]

SECTION_SEPARATOR = "\n\n"


@dataclass
class Chunk:
    id: str
    text: str
    source_id: str
    source_title: str
    section_ids: list[str]
    labels: list[str]  # citation labels of the overlapped sections, e.g. "SS Code §53"
    pages: list[int]  # [first, last]
    url: str
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_size_chunks(
    sections: Iterable[dict], token_offsets: TokenOffsets, size: int = 500, overlap: int = 50
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    by_source: dict[str, list[dict]] = {}
    for s in sections:
        by_source.setdefault(s["source_id"], []).append(s)

    chunks: list[Chunk] = []
    for source_id, secs in by_source.items():
        # Concatenate the document, remembering where each section's text lies.
        doc, spans = "", []
        for s in secs:
            if doc:
                doc += SECTION_SEPARATOR
            spans.append((len(doc), len(doc) + len(s["text"]), s))
            doc += s["text"]

        offsets = token_offsets(doc)
        step = size - overlap
        for n, start in enumerate(range(0, max(len(offsets) - overlap, 1), step)):
            window = offsets[start : start + size]
            if not window:
                break
            c0, c1 = window[0][0], window[-1][1]
            covered = [s for a, b, s in spans if a < c1 and b > c0]
            pages = [p for s in covered for p in s["pages"]]
            chunks.append(Chunk(
                id=f"{source_id}:c{n}",
                text=doc[c0:c1],
                source_id=source_id,
                source_title=secs[0]["source_title"],
                section_ids=[s["id"] for s in covered],
                labels=[s["label"] for s in covered],
                pages=[min(pages), max(pages)] if pages else [],
                url=secs[0]["url"],
                n_tokens=len(window),
            ))
    return chunks
