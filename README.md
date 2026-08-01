# AI Resume Screening & Candidate Ranking System

An AI-powered Resume Screening & Candidate Ranking System built with **Python** and **Flask**. This project aims to automate resume screening by extracting information from resumes, comparing them against job descriptions using NLP techniques, and ranking candidates based on semantic similarity.

> **Project Status:** Under Development

---

## Overview

Recruiters often spend significant time manually reviewing resumes. This project streamlines that process by leveraging Natural Language Processing (NLP) to parse resumes, extract relevant skills, and rank candidates based on how well they match a given job description.

---

## Planned Features

- Resume upload (PDF, DOCX, TXT)
- Resume text extraction
- NLP preprocessing
- Skill extraction using spaCy
- TF-IDF based similarity scoring
- Sentence Transformer semantic matching
- Candidate ranking dashboard
- CSV export of ranked candidates
- Recruiter-friendly interface

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Backend | Flask |
| NLP | spaCy, NLTK |
| Machine Learning | scikit-learn, Sentence Transformers |
| Similarity | TF-IDF, Cosine Similarity |
| Frontend | HTML, Bootstrap, JavaScript |
| Testing | Pytest |
| Version Control | Git, GitHub |

---

## Project Structure

```text
resume-screening/
│
├── app/
├── models/
├── notebooks/
├── static/
├── tests/
├── uploads/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Current Progress

- Flask application factory
- Configuration management
- Logging setup
- Health check API (`/health`)
- Project structure
- Unit tests
- GitHub repository setup

---

## Current API

### Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "resume-screening"
}
```

---

##  Getting Started

### Clone the repository

```bash
git clone https://github.com/<your-username>/resume-screening.git
cd resume-screening
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

---

##  Run Tests

```bash
pytest
```

---

##  Roadmap

- [x] Project initialization
- [x] Flask backend setup
- [x] Health API
- [ ] Resume upload API
- [ ] Resume parsing
- [ ] Skill extraction
- [ ] TF-IDF ranking
- [ ] Sentence Transformer ranking
- [ ] Recruiter dashboard
- [ ] CSV export
- [ ] Deployment on Render

---

## License

This project is licensed under the MIT License.

---

## Author

**Sarathy**

B.Tech Computer Science Engineering (AI/ML)

Passionate about Machine Learning, NLP, and AI-powered applications.
