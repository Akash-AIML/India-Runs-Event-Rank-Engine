#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

# Ensure src is in python path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

from parse_jd import get_jd_keywords
from features import is_honeypot, is_consulting_only
from scoring import score_candidate
from reasoning import get_reasoning
from embeddings import tokenize, BM25Model

def load_precomputed_scores(artifacts_dir):
    scores = {}
    if not artifacts_dir:
        return scores
    features_path = os.path.join(artifacts_dir, "candidate_features.csv")
    if os.path.exists(features_path):
        try:
            with open(features_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) >= 2:
                        cid = row[0]
                        scores[cid] = float(row[1]) # semantic score
        except Exception:
            pass
    return scores

def load_precomputed_reasonings(artifacts_dir):
    reasonings = {}
    if not artifacts_dir:
        return reasonings
    reasonings_path = os.path.join(artifacts_dir, "precomputed_reasonings.json")
    if os.path.exists(reasonings_path):
        try:
            with open(reasonings_path, "r", encoding="utf-8") as f:
                reasonings = json.load(f)
        except Exception:
            pass
    return reasonings

def extract_candidate_text(cand):
    profile = cand['profile']
    text_parts = [
        profile.get('headline', ''),
        profile.get('summary', ''),
        profile.get('current_title', ''),
        profile.get('current_industry', '')
    ]
    for job in cand.get('career_history', []):
        text_parts.append(job.get('title', ''))
        text_parts.append(job.get('company', ''))
        text_parts.append(job.get('description', ''))
    for edu in cand.get('education', []):
        text_parts.append(edu.get('degree', ''))
        text_parts.append(edu.get('field_of_study', ''))
    return " ".join([p for p in text_parts if p])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", required=True, help="Path to job description docx")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    parser.add_argument("--artifacts-dir", default=None, help="Directory with precomputed artifacts")
    args = parser.parse_args()
    
    # Parse JD keywords
    jd_keywords = get_jd_keywords(args.jd)
    
    # Try to load precomputed scores
    precomputed_scores = load_precomputed_scores(args.artifacts_dir)
    precomputed_reasonings = load_precomputed_reasonings(args.artifacts_dir)
    
    # Read candidates
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            candidates.append(cand)
            
    # Fallback to computing BM25 live if scores are not precomputed
    if not precomputed_scores:
        print("Precomputed scores not found or not provided. Running BM25 model on-the-fly...")
        tokenized_docs = [tokenize(extract_candidate_text(c)) for c in candidates]
        bm25 = BM25Model()
        bm25.fit(tokenized_docs, jd_keywords)
        live_bm25_scores = [bm25.score(doc, jd_keywords) for doc in tokenized_docs]
        max_s = max(live_bm25_scores) if live_bm25_scores else 1.0
        min_s = min(live_bm25_scores) if live_bm25_scores else 0.0
        
        precomputed_scores = {}
        for idx, cand in enumerate(candidates):
            cid = cand['candidate_id']
            s_val = live_bm25_scores[idx]
            precomputed_scores[cid] = (s_val - min_s) / (max_s - min_s) if max_s > min_s else 0.0
            
    # Score candidates
    scored_candidates = []
    for cand in candidates:
        cid = cand['candidate_id']
        
        # Honeypot check
        if is_honeypot(cand):
            continue
            
        # Consulting only check
        # Semantic score from precomputed/live
        sem_score = precomputed_scores.get(cid, 0.0)
        
        # Calculate composite score
        score, breakdown = score_candidate(cand, sem_score, jd_keywords)
        
        # Disqualified checks (e.g. honeypots have score 0.0)
        if score == 0.0:
            continue
            
        scored_candidates.append((cid, score, cand))
        
    # Sort: descending by score, ascending by candidate_id for tie-breaking
    scored_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Write top 100 to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for idx, (cid, score, cand) in enumerate(scored_candidates[:100]):
            rank = idx + 1
            reasoning = get_reasoning(cid, cand, precomputed_reasonings)
            writer.writerow([cid, rank, round(score, 4), reasoning])
            
    print(f"Success! Top 100 candidate ranking written to {args.out}")

if __name__ == "__main__":
    main()
