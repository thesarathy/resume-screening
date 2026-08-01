# AI Resume Screening & Candidate Ranking System

An AI-powered **Resume Screening & Candidate Ranking System** built with **Python** and **Flask**. Upload resumes and a job description, and the app parses each resume, extracts candidate facts and skills with spaCy, scores how well each matches the JD, and ranks the candidates — with history, a candidate detail view, and CSV export.

---

## Features

- **Resume upload** — PDF, DOCX, TXT (drag-and-drop, multiple files)
- **Text extraction** — pdfplumber / PyMuPDF / python-docx
- **Entity extraction** — name, email, phone, education, years of experience (spaCy NER + regex)
- **Skill extraction** — curated vocabulary + PhraseMatcher, plus candidate-skill suggestions
- **Two ranking methods** (switchable in the UI):
  - **TF-IDF** — fast, fully offline, interpretable
  - **Sentence Transformer** (semantic) — `all-MiniLM-L6-v2`, better at paraphrases
- **Ranking history** — every run is persisted and re-openable
- **Candidate detail view** — full facts + resume-text preview
- **CSV export** — download any ranking for Excel
- **Database** — SQLite locally, PostgreSQL via `DATABASE_URL` (for deployment)

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Backend | Flask, Flask-SQLAlchemy |
| NLP | spaCy |
| ML | scikit-learn, Sentence Transformers |
| Similarity | TF-IDF, Cosine Similarity |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |
| Hosting (optional) | Render / Railway (gunicorn) |

---

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the app

```bash
python app.py
```

Then open <http://localhost:5000>.

> **Windows PowerShell** — run the venv interpreter directly:
> `.\venv\Scripts\python.exe app.py`

### 5. Run the tests

```bash
pytest
```

---

## Usage

1. Open the dashboard at `/` and **paste a job description**.
2. **Add one or more resumes** (drag & drop, or browse). Select several at once or click **“Add another resume”**.
3. Pick a **ranking method**: *TF-IDF* (fast) or *Semantic* (Sentence Transformer; downloads the model on first use).
4. Click **Rank candidates** — you're taken to the results page sorted by match.
5. Use **🗂 History** to see past rankings, open any candidate for a **detail view**, or **⬇ Download CSV** to export.

### Routes

| Route | Description |
|-------|-------------|
| `GET /` | Upload form / dashboard |
| `POST /rank` | Run a ranking (redirects to the result) |
| `GET /rankings` | Ranking history |
| `GET /rankings/<id>` | View a ranking's results |
| `GET /rankings/<id>/candidates/<cid>` | Candidate detail |
| `GET /rankings/<id>/export.csv` | CSV export |
| `GET /health` | Health check |

---

## Project Structure

```text
resume-screening/
├── app/
│   ├── templates/        # HTML pages (index, results, history, detail)
│   ├── static/css/       # Stylesheet
│   ├── data/skills.json  # Skill taxonomy
│   ├── __init__.py       # Flask app factory + routes
│   ├── config.py         # Environment-based config (SQLite/Postgres)
│   ├── factory.py        # Shared spaCy + scorer construction
│   ├── models.py         # Ranking + Candidate ORM models
│   ├── parser.py         # PDF/DOCX/TXT text extraction
│   ├── preprocessing.py  # Text cleaning / normalization
│   ├── entity_extractor.py
│   ├── skill_extractor.py
│   ├── similarity.py     # TF-IDF + Sentence-Transformer scorers
│   └── ranking.py        # Orchestration: parse → extract → score → rank
├── tests/                # Pytest suite
├── app.py                # Local dev entry point
├── wsgi.py               # Production (gunicorn) entry point
├── render.yaml           # Optional Render deploy config
├── railway.json          # Optional Railway deploy config
└── requirements.txt
```

---

## Deployment (optional)

The repo ships ready-to-host configs, but the app runs fully locally without them.

### Render

Push to GitHub, create a **New → Blueprint** from this repo (it reads `render.yaml`). Set a `DATABASE_URL` (e.g. a free Neon/Supabase Postgres) so history persists — Render's free-tier disk is ephemeral. Build installs deps and downloads the spaCy model automatically; the service runs via `gunicorn wsgi:app`.

### Railway

Deploy the repo on Railway; `railway.json` sets the correct start command (`gunicorn wsgi:app`) and the spaCy build step. Add a Postgres plugin and set `DATABASE_URL`.

> **Memory note:** Render/Railway free tiers have ~512MB. Both heavy models (spaCy and the Sentence Transformer) load **lazily** on first use, not at startup. If a deploy still runs out of memory during the *build*, it's usually the `sentence-transformers`/PyTorch install — consider a larger plan or removing that dependency.

---

## API

### Health Check

```
GET /health
```

```json
{ "status": "ok", "service": "resume-screening" }
```

---

## License

MIT License.

---

## Author

**Sarathy** — B.Tech Computer Science Engineering (AI/ML). Passionate about Machine Learning, NLP, and AI-powered applications.
