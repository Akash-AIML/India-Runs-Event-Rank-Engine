# Redrob Intelligent Candidate Ranking System

This repository implements a complete, submission-ready candidate ranking pipeline for the Redrob "Intelligent Candidate Discovery & Ranking Challenge."

## Approach (TL;DR)
The pipeline combines structured feature extraction (title relevance, production deployment evidence, company type classification, skills corroboration), local BM25 semantic scoring, rule-based hard disqualifiers (keyword stuffers, honeypots, services-only), and a multiplicative behavioral availability multiplier (from `redrob_signals`: notice period, recruiter engagement, open_to_work, location). The design explicitly avoids keyword-only matching by scoring title-career corroboration and rewarding plain-language descriptions of production search/retrieval work over buzzword-stuffed skill lists.

## Repository Structure
```
redrob-ranker/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── submission.csv                   # Final ranked output (top 100)
├── validate_submission.py           # Official validator
├── data/                            # Hackathon bundle files
│   ├── candidates.jsonl             # 100K candidates
│   ├── job_description.docx
│   ├── candidate_schema.json
│   ├── redrob_signals_doc.docx
│   └── submission_spec.docx
├── artifacts/                       # Precomputed (offline step)
│   ├── candidate_embeddings.npy     # Dummy embeddings matrix (100K x 384)
│   ├── candidate_features.csv       # Precomputed semantic scores
│   ├── candidate_ids.csv            # Precomputed candidate IDs
│   ├── parsed_jd.json               # Parsed JD keywords
│   └── precomputed_reasonings.json  # Precomputed reasonings for top candidates
├── src/
│   ├── parse_jd.py                  # JD docx -> structured requirements
│   ├── features.py                  # Candidate feature extraction
│   ├── embeddings.py                # Local BM25 scoring model
│   ├── precompute.py                # Offline artifact generation
│   ├── scoring.py                   # Full composite scoring
│   ├── reasoning.py                 # Candidate-specific reasoning strings
│   └── rank.py                      # CLI entrypoint
├── tests/
│   └── test_scoring.py              # Unit tests covering all trap archetypes
└── sandbox/
    └── app.py                       # Streamlit sandbox app (placeholder)
```

## Setup
```bash
pip install -r requirements.txt
```

## Precompute (One-Time Offline Step)
```bash
python src/precompute.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out-dir ./artifacts
```
This step fits the BM25 model, scores all candidates, and generates the precomputed artifacts in `./artifacts/`.

## Reproduce submission.csv (Single Command)
```bash
python src/rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./submission.csv --artifacts-dir ./artifacts
```
When precomputed artifacts are present, this loads them directly and runs in **less than 2 seconds** on CPU.

## Validate Output
```bash
python validate_submission.py ./submission.csv
```

## Run Unit Tests
```bash
python -m unittest tests/test_scoring.py
```

## Compute Environment
* **CPU only**: Yes (no GPU required).
* **RAM target**: < 100 MB.
* **Ranking runtime (with precompute cached)**: < 2 seconds.
* **Precompute runtime**: ~10 seconds.
* **Python version**: 3.11/3.12.
