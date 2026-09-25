# Zepto Data & AI Platform

A three-part learning project: a books data pipeline, Titanic analytics/modeling, and an offline-first policy support assistant.

## Setup
Python 3.11 recommended.
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Module 1 — Books data pipeline
Run from the repository root:
```bash
python data_pipeline/scraper.py
python data_pipeline/cleaning.py
python data_pipeline/database.py
python data_pipeline/queries.py
```
Scraper discovers catalog categories and gathers up to 25 books per category (three categories, target at least 60). It requires network access. SQL statements/results are saved under `query_outputs/`; SQL JOIN and category-count results are checked against pandas equivalents.

## Module 2 — Titanic analytics
Place the Titanic CSV at `analytics/titanic.csv` or run the EDA script once with network access to fetch the seaborn Titanic dataset.
```bash
python analytics/analysis_starter.py
python analytics/modeling_starter.py
```
Charts, tables, metrics, and fitted pipelines are saved under `analytics/outputs/`. The classifier split is stratified and performed before fitting preprocessing. SMOTE is optional and runs only on training data when imbalanced-learn is available.

## Module 3 — Support assistant
Run from `support_assistant/`:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```
`GET /health` checks service status; `POST /ask` accepts `{"query":"What is the return policy?"}`. The workflow uses local sentence-transformer embeddings and a persistent ChromaDB collection, then returns the best-matching supplied policy passage and source file. First run may download the embedding model. No hosted LLM or API key is used. Build the container from the repository root:
```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run -p 7860:7860 zepto-support
```

## Notes
- Scraping requires internet and is subject to the target site's availability and terms.
- Model metrics are generated from the local dataset when scripts are run; no performance numbers are hard-coded.
- This is an educational demonstration, not an official Zepto service.
