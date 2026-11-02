# LabourLens

A "know your rights" assistant for India's four new Labour Codes. It answers questions with section-level citations, runs gratuity and overtime calculations through tools, and is measured against a hand-checked evaluation set.

> 🚧 Work in progress. See [task.md](task.md) for the full plan.

## Repo layout

```
backend/            Python: ingestion, retrieval, tools, generation, API, evals
  data/sources.json   list of official source documents
  data/SOURCES.md     generated: what was downloaded, checksums, corpus date
  scripts/            one-off scripts (e.g. download_sources.py)
frontend/           Next.js (App Router, TypeScript, Tailwind)
task.md             project plan
```

## Setup

### Backend
```powershell
cd backend
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env          # fill in GEMINI_API_KEY
.venv\Scripts\python -m scripts.download_sources   # official PDFs -> data/raw/
.venv\Scripts\python -m src.ingest.parse           # -> data/processed/sections.jsonl, definitions.jsonl
.venv\Scripts\python -m scripts.fetch_model        # embedding model (only if huggingface.co is slow for you)
.venv\Scripts\python -m src.retrieval.index        # chunk + embed + store in Qdrant (local, data/qdrant)
.venv\Scripts\python -m src.ask "Am I eligible for gratuity on a fixed-term contract?"
.venv\Scripts\python -m pytest
```

### Evaluation set
```powershell
.venv\Scripts\python -m evals.draft_questions   # LLM-drafted candidates -> evals/drafts.jsonl
.venv\Scripts\python -m evals.review            # human review, one question at a time
.venv\Scripts\python -m evals.review --export   # -> evals/golden_set.jsonl
.venv\Scripts\python -m evals.run --config evals/configs/baseline.yaml   # -> evals/results/RESULTS.md
.venv\Scripts\python -m evals.spotcheck baseline   # blind-grade ~30 answers to check the LLM judge
```

Results: [backend/evals/results/RESULTS.md](backend/evals/results/RESULTS.md)

### Frontend
```powershell
cd frontend
npm install
npm run dev
```

## Disclaimer
LabourLens is for information only and is not legal advice. It covers central law only; state rules may differ.
