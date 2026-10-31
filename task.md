# LabourLens — "Know Your Rights" Assistant for India's New Labour Codes

**Decisions:** Next.js frontend · English only (v1) · FastAPI backend · Gemini LLM

## 1. What is this project?

India's **4 new labour codes** came into force on **21 Nov 2025** and replaced **29 older labour laws**. Rules on gratuity, wages, overtime, working hours and layoffs all changed, and employees, freshers and small-business owners are confused by explainers that disagree with each other.

**LabourLens** is a web app where anyone can ask a question in plain language and get an answer that:
- **cites the exact section** of the code or rules it came from
- **does the calculation** (gratuity, overtime) using calculator tools instead of the LLM guessing numbers
- **shows the old law next to the new code** when something changed
- **says "I don't know"** when the codes don't cover the question (for example, state-specific rules)

**Example questions:**
- *"I'm on a 1-year fixed-term contract. Am I eligible for gratuity?"*
- *"My basic is ₹30,000 and I worked 6 years 8 months. How much gratuity do I get?"*
- *"How much overtime pay should I get for 10 extra hours?"*
- *"What changed in gratuity rules compared to the old Gratuity Act?"*

The project is judged by **evaluation**:
- You build a hand-checked test set of questions with known answers.
- You measure accuracy for every change (better chunking, hybrid search, a reranker…).
- You publish the before/after numbers in a results table.

That table, plus a live demo, is what makes this a strong resume project rather than a chatbot demo.

### How it works

```
              ┌──────────────── INDEXING (offline, once) ────────────────┐
 Official PDFs ─► parse into Code → Chapter → Section → Sub-section
 (4 codes, rules,        │
  old acts, FAQs)        ├─► chunks + metadata (code, section no., old/new) ─► vector DB
                         ├─► keyword index (BM25)
                         └─► definitions table ("wages", "employee", "fixed-term")

              ┌──────────────── ANSWERING (per question) ────────────────┐
 user question ─► query rewrite ("PF" → "provident fund") ─► hybrid search ─► reranker
                                                                         │
                         LLM (with tools) ◄── top sections + definitions ◄┘
                          │        │
                          │        └─► calculator tools: gratuity / overtime / wage-50% check
                          ▼
                answer + section citations + "old vs new" note ─► FastAPI ─► web UI

              ┌──────────────── EVALUATION (on every change) ────────────┐
 golden test set (~120 Q&A) ─► run pipeline ─► retrieval + answer + calculation
                                                metrics ─► results table
```

### What you'll have at the end
- [ ] A **live demo URL** that anyone can try
- [ ] A GitHub repo with clean code, tests, Docker, and a README holding an **experiments table**
- [ ] A hand-checked test set of ~120 labour-law questions that you made yourself
- [ ] Resume bullets with real numbers, for example:
  > **LabourLens — RAG assistant for India's new Labour Codes.** Built a hybrid-retrieval RAG pipeline (BM25 + dense + cross-encoder reranking) with section-level citations and LLM tool calling for gratuity/overtime calculations. Improved Recall@5 from 0.xx → 0.xx and answer correctness from xx% → xx% on a 120-question hand-labelled eval set. Deployed with FastAPI, Next.js and Docker, with an eval subset running in CI.

---

## 2. Data sources

Stick to **central** law for v1. State rules are out of scope, and the assistant should say so.

| Source | What it gives you | Where to get it |
|---|---|---|
| Code on Wages, 2019 | Minimum wage, wage definition, overtime, bonus | India Code / Ministry of Labour / eGazette |
| Industrial Relations Code, 2020 | Layoffs, retrenchment, strikes, standing orders | same |
| Code on Social Security, 2020 | Gratuity, PF, ESI, maternity, gig workers | same |
| Occupational Safety, Health & Working Conditions Code, 2020 | Working hours, leave, safety | same |
| Central rules under each code | Procedures, forms, numbers | eGazette / Ministry of Labour (**check the latest notified version in Phase 0**) |
| Old acts (Payment of Gratuity Act 1972, Minimum Wages Act 1948, Payment of Bonus Act 1965, etc.) | The "old law" side of comparisons | India Code (indiacode.nic.in) |
| Ministry of Labour / PIB FAQs | Official plain-language explanations | labour.gov.in, pib.gov.in |

⚠️ **Only use official texts as sources.** Blogs and news articles often get details wrong. Use them only to come up with test questions, never as sources the assistant cites.

---

## 3. Step-by-step tasks

Estimated total: **~4 weeks at 2–3 hrs/day.** Each phase ends with something that works.

### Phase 0 — Set up (Day 1–2) ✅ done 2026-10-30
- [x] Download the official PDFs listed above. Note each one's notification date and save the URLs in `data/SOURCES.md`.
  - 9 documents: the 4 codes, the 4 final central rules (G.S.R. 342–345(E), 8 May 2026) and the PIB explainer. Run `python -m scripts.download_sources`.
- [x] **Confirm the current versions**: which rules are finally notified and which are still drafts. Record the "corpus as of" date, because the UI will show it.
  - All 4 central rules are final (notified 8 May 2026). Corpus as of 2026-10-30.
- [ ] Old acts (Gratuity 1972, Minimum Wages 1948, Bonus 1965, Wages 1936): **download manually** from indiacode.nic.in, which is unreliable for scripts. Only needed for the old-vs-new feature; listed in `SOURCES.md`.
- [x] Python virtual env (3.13 works so far) and `.env.example`. Local git repo initialised.
- [ ] Create the GitHub repo and push (you do this).
- [ ] Copy `backend/.env.example` → `backend/.env` and add your `GEMINI_API_KEY` (you do this).
- [x] Set up the folder structure:
  - `backend/` (Python): `data/`, `src/ingest`, `src/retrieval`, `src/tools`, `src/generation`, `src/api`, `evals/`, `tests/`, `scripts/`
  - `frontend/` (Next.js 16 + TypeScript + Tailwind, dependencies installed)

### Phase 1 — Parse the law into structure (Day 3–5) ✅ done 2026-10-30 (except old-act mapping)
- [x] Extract text from the PDFs (`pymupdf`) and remove headers, footers and page numbers.
  - Also removes the Hindi halves of the rules, the gazette mastheads and the margin cross-references, and fixes odd quote glyphs and ligatures.
- [x] Split each document into its **hierarchy**: Code → Chapter → (Part) → Section → Sub-section. Store the section number and title as metadata.
  - Codes: titles come from the margin notes, matched by position. Rules: titles come from the inline headings. Schedules are separate records; forms are left out.
  - Result: **863 records**. Section counts match the official texts (Wages 69, IR 104, SS 164, OSH 143; rules 54 / 50 / 69 / 186). Only 1 of 954 sections/rules is untitled (SS Rules r.63, which has no title in the original).
- [x] Build a **definitions table** from each code's definitions section (usually Section 2), mapping each term to its definition and source.
  - **288 definitions**, e.g. `Wages Code §2(y)` "wages", `SS Code §2(35)` "gig worker".
- [ ] Map **old act sections to new code sections** for the key topics: gratuity, minimum wage, bonus, overtime, working hours, layoffs. Start with about 20 mappings done by hand.
  - **Blocked** until the old Acts are downloaded (India Code is unreachable by script; see `SOURCES.md`).
- [x] Save as JSONL, one record per section: `data/processed/sections.jsonl` and `definitions.jsonl`. Run `python -m src.ingest.parse`.
- [x] Tests: `tests/test_parse.py` (27 tests: parsing rules and checks on the real output). Run `python -m pytest`.
- **Done when:** you can look up any section by number and its text is clean. ✅

### Phase 2 — Baseline RAG, the simple version (Day 6–7) ✅ done 2026-10-31
- [x] Fixed-size chunking (e.g. 500 tokens with 50 overlap), **ignoring** the structure from Phase 1.
  - 863 sections → **683 chunks** of 500 tokens. Each chunk still records which sections it overlaps, for scoring retrieval in Phase 4.
- [x] Embed with a small open model (`bge-small`), store in Qdrant, retrieve the top-5 chunks and send them to the LLM.
  - `BAAI/bge-small-en-v1.5` (384-d). Qdrant in **local mode** (`data/qdrant`, no server) behind a `VectorStore` interface, so the database can be swapped later.
  - LLM: Gemini **Flash-Lite** (~1.5 s per answer), with Flash as fallback. Flash was overloaded (503/504) during testing.
  - Embedding all chunks takes ~5 min on CPU. Vectors are saved to `vectors_baseline.npy`.
- [x] **Don't optimise yet.** This is the "before" number.
- [x] Tests: `tests/test_retrieval.py` (chunking, store, pipeline with fakes). 38 tests total.
- **Done when:** you can ask a question from the command line and get an answer. ✅
  - `python -m src.ask "Am I eligible for gratuity on a one year fixed-term contract?"` gives a correct answer citing SS Rules r.33.
- **Baseline weaknesses already visible (to measure in Phase 4 and fix in Phase 5):**
  - Chunks span 4–5 sections, so citations are vague ("Wages Rules r.6–r.10").
  - The same rule can fill several of the 5 slots.
  - The PIB explainer is cited next to the actual law.
- **Notes for slow networks:** `python -m scripts.fetch_model` downloads the embedding model through a mirror and checks it against huggingface.co's hashes. After that, set `HF_HUB_OFFLINE=1`.

### Phase 3 — Build the evaluation dataset (Day 8–10) ⭐ most important phase
- [ ] Write about 120 questions, each with the **correct answer** and the **sections that support it**. Mix these types:

  | Type | Example | Approx. count |
  |---|---|---|
  | Definition | "Who counts as a fixed-term employee?" | 15 |
  | Eligibility / yes-no | "Can a gig worker get social security benefits?" | 25 |
  | Calculation | "Gratuity for ₹40k wages, 7 years?" | 20 |
  | Old vs new | "What changed in gratuity eligibility?" | 15 |
  | Multi-section | answers needing 2+ sections | 15 |
  | Everyday wording | "can my company cut my PF?" | 15 |
  | **Out of scope / unanswerable** | "Telangana minimum wage for drivers?" (state rule) | 15 |

- [x] Use an LLM to **draft** questions if you want, but **check every answer against the official text yourself.**
  - `python -m evals.draft_questions` drafts 90 questions with Gemini from chosen sections: 50 single-section (precise and everyday wording alternating), 15 multi-section pairs, 15 definitions, 10 "what changed" from the PIB explainer.
  - Every draft must include a **verbatim quote** from its gold section. `evals/dataset.py` checks the quote and rejects questions that give away the section number.
  - An **AI pre-check** read every draft against the full section text. It corrected 7 drafts (e.g. "lay off" used where the section is about retrenchment; a government power stated as a direct right) and recommends rejecting 6 (4 duplicates, 1 ambiguous provision, 1 bad quote). Each draft shows its note in the review tool.
  - "Old vs new" became **"what changed"**, sourced from the PIB explainer, because the old Acts couldn't be downloaded.
- [x] For calculation questions, work out the expected numbers by hand using the formulas **from the official text**.
  - 17 calculations in `evals/handwritten.jsonl`: gratuity (÷26×15 rule, the more-than-6-months rounding, seasonal), the 50% wages rule, overtime, bonus min/max, lay-off, retrenchment, annual leave and closure cap. Plus 15 out-of-scope questions, each checked to be absent from the corpus.
- [ ] **YOUR TURN: review every question** (about 1–2 hours). This is what makes "hand-checked" true on your resume:
  ```
  cd backend
  .venv\Scripts\python -m evals.review            # a=accept r=reject q/w=edit t=full text x=exit
  .venv\Scripts\python -m evals.review --status   # progress
  .venv\Scripts\python -m evals.review --export   # writes evals/golden_set.jsonl
  ```
  - 122 candidates; the pre-check recommends 116 to accept. Decisions save immediately, so you can stop and continue later.
- [ ] Freeze it as `evals/golden_set.jsonl`. Don't change it once experiments start.
- **Done when:** you have a versioned test set that you trust.

### Phase 4 — Evaluation harness + baseline numbers (Day 11–12)
- [ ] **Retrieval:** Recall@5 and MRR, i.e. did the right section show up, and how high.
- [ ] **Answers:**
  - correctness against the reference (LLM-as-judge, spot-checked by you on ~30 answers)
  - **citation accuracy**: does each cited section really support the claim
  - **refusal accuracy** on the out-of-scope questions
- [ ] **Calculations:** exact match within ₹1.
- [ ] **Also log:** latency and cost per question.
- [ ] One command, `python -m evals.run --config baseline.yaml`, writes results to CSV and W&B.
- **Done when:** the baseline row of your results table exists.

### Phase 5 — Improve retrieval, one change at a time (Day 13–17)
Run the full evaluation after each change and keep one row per experiment:

| Experiment | Recall@5 | MRR | Correctness | Latency |
|---|---|---|---|---|
| Baseline (fixed chunks, dense only) | | | | |
| + section-aware chunking (from Phase 1) | | | | |
| + better embedding model (`bge-m3` or similar) | | | | |
| + hybrid search (BM25 + dense) | | | | |
| + cross-encoder reranker | | | | |
| + auto-attach definitions of terms in retrieved sections | | | | |
| + query rewriting (everyday words → legal terms) | | | | |

- [ ] Write 2–3 lines on **why** each change helped or didn't. Interviewers will ask.

### Phase 6 — Calculator tools (Day 18–19)
- [ ] Write plain Python functions with **unit tests**:
  - `gratuity(last_wage, years, months, employment_type)`
  - `overtime_pay(wage, hours)`
  - `wage_share_check(basic, allowances, ctc)` (checks the rule that "wages" must be at least 50% of pay)
- [ ] Take every formula and rule **from the official sections** and cite the section in the docstring.
- [ ] Expose them to the LLM through **function/tool calling** (Gemini supports this).
- [ ] Measure: does the model pick the right tool, with the right inputs? Are the final numbers right on the 20 calculation questions?

### Phase 7 — Answer quality (Day 20–21)
- [ ] Prompt that requires **inline citations** like `[SS Code §53(1)]`, plus a check that each cited section exists in the retrieved text.
- [ ] An "old vs new" block whenever the retrieved sections have a mapping from Phase 1.
- [ ] Return "not covered by central codes, check your state rules" when retrieval confidence is low or the question is state-specific.
- [ ] A fixed disclaimer: *informational only, not legal advice*, plus the "corpus as of" date.
- [ ] English only for v1. Hindi/Telugu can be a later add-on.

### Phase 8 — API + UI (Day 22–24)
- [ ] FastAPI endpoints `/ask` (streams the answer, returns sources and tool calls), `/health` and `/docs`.
- [ ] Next.js UI (App Router + TypeScript + Tailwind), streaming answers from `/ask`:
  - a question box
  - the answer
  - **clickable citation cards** that show the original section text
  - a calculator result card
  - a few example questions on the home screen
- [ ] Basic logging of questions, latency and errors, without storing personal details.

### Phase 9 — Deploy + CI (Day 25–26)
- [ ] Dockerfile + docker-compose (backend + Qdrant; frontend optional for local runs).
- [ ] Deploy:
  - **frontend** (Next.js) to Vercel's free tier
  - **backend** (FastAPI) to Render, Railway or Hugging Face Spaces
  - Qdrant Cloud free tier for the vector DB
- [ ] GitHub Actions on every push:
  - unit tests (calculators)
  - a **small evaluation subset** of about 20 questions; fail the build if quality drops below a set minimum

### Phase 10 — Package it for your resume (Day 27–28)
- [ ] README with:
  - a demo GIF and the live link
  - an architecture diagram
  - the **experiments table** and key findings
  - how to run it
  - the data sources and corpus date
- [ ] Fill in the resume bullet with your real numbers.
- [ ] Optional: share it with friends and batchmates, collect feedback, and turn real user questions into new test cases.
- [ ] Optional: a LinkedIn post, e.g. *"5 things that changed for employees under the new labour codes, answered by a RAG system I built."*

---

## 4. Resources needed

### Accounts & keys
| What | Why | Cost |
|---|---|---|
| Gemini API key (you already have one from tech_news) | Answers, tool calling, LLM judge | Free tier is likely enough |
| GitHub | Repo + Actions CI | Free |
| Weights & Biases | Experiment tracking | Free |
| Hugging Face | Models + hosting the demo on Spaces | Free |
| Qdrant Cloud *(optional)* | Hosted vector DB for the live demo | Free tier |

### Compute
**Your laptop's CPU is enough.** The documents are small: embedding them takes minutes, and the LLM runs through the API. You don't need a GPU.

### Tech stack
| Layer | Tool |
|---|---|
| PDF parsing | `pymupdf` (+ regex for section splitting) |
| Embeddings | `sentence-transformers` (`bge-small` → `bge-m3`) |
| Vector DB | Qdrant (Docker locally) |
| Keyword search | `rank_bm25` |
| Reranker | `bge-reranker` cross-encoder |
| LLM + tools | Gemini (Flash) with function calling |
| Evaluation | Your own harness (+ optionally RAGAS) |
| Tracking | Weights & Biases |
| Backend | FastAPI |
| Frontend | Next.js (App Router) + TypeScript + Tailwind |
| Testing | `pytest` |
| Deploy | Vercel (frontend) + Docker on Render / HF Spaces (backend) |

Write the core pipeline without LangChain/LlamaIndex. You'll understand every piece, and that gives you more to talk about in interviews.

### What to read
- Embeddings & search: [sbert.net](https://sbert.net), [Qdrant docs](https://qdrant.tech/documentation/), [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Better retrieval: Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) post
- Evaluation: [RAGAS docs](https://docs.ragas.io)
- Tool calling: Gemini API function-calling docs
- API: [FastAPI tutorial](https://fastapi.tiangolo.com)

---

## 5. Risks & how to handle them
| Risk | Mitigation |
|---|---|
| Wrong legal answers | Cite only official texts, check citations automatically, show a disclaimer, prefer refusing over guessing |
| Rules still changing (central rules, state rules) | Record the "corpus as of" date, keep v1 to central law, make re-indexing one command |
| Calculations wrong | Formulas come from the official sections, are unit-tested, and the LLM never does arithmetic itself |
| Small document set makes experiments look similar | Use everyday-wording and multi-section questions, which show the differences between methods |
| Free API rate limits | Cache LLM calls and run the small evaluation subset during development |

---

## 6. Decisions (made)
1. **UI:** Next.js
2. **Languages:** English only for v1
3. **Project name:** LabourLens
