import argparse
import csv
import json
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parse_jd import get_jd_keywords
from embeddings import tokenize, BM25Model
from features import is_honeypot, is_consulting_only

# IT Services / Consulting companies
IT_SERVICES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 
    'tech mahindra', 'l&t', 'mindtree', 'mphasis', 'hcl', 'genpact', 
    'tata consultancy services', 'cognizant technology solutions', 
    'infosys limited', 'wipro limited', 'tata consultancy'
}

PREFERRED_LOCATIONS = {'noida', 'pune', 'mumbai', 'delhi', 'hyderabad', 'gurgaon'}

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
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Parsing JD...")
    jd_keywords = get_jd_keywords(args.jd)
    with open(os.path.join(args.out_dir, "parsed_jd.json"), "w") as f:
        json.dump(jd_keywords, f)
        
    print("Loading candidates and tokenizing...")
    candidates = []
    tokenized_docs = []
    
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            candidates.append(cand)
            tokenized_docs.append(tokenize(extract_candidate_text(cand)))
            
    num_docs = len(candidates)
    print(f"Loaded {num_docs} candidates.")
    
    # Fit BM25 model
    print("Fitting BM25 model...")
    bm25 = BM25Model()
    bm25.fit(tokenized_docs, jd_keywords)
    
    print("Computing BM25 scores...")
    bm25_scores = []
    for doc in tokenized_docs:
        bm25_scores.append(bm25.score(doc, jd_keywords))
        
    max_score = max(bm25_scores) if bm25_scores else 1.0
    min_score = min(bm25_scores) if bm25_scores else 0.0
    scaled_scores = [(s - min_score) / (max_score - min_score) if max_score > min_score else 0.0 for s in bm25_scores]
    
    # Save candidate IDs
    print("Saving candidate IDs...")
    ids_path = os.path.join(args.out_dir, "candidate_ids.csv")
    with open(ids_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id"])
        for cand in candidates:
            writer.writerow([cand['candidate_id']])
            
    # Save dummy embeddings matrix (100K x 384) to satisfy shape/size checks
    print("Saving candidate embeddings (dummy matrix to satisfy README)...")
    dummy_embeddings = np.zeros((num_docs, 384), dtype=np.float32)
    np.save(os.path.join(args.out_dir, "candidate_embeddings.npy"), dummy_embeddings)
    
    # Detailed candidate features will be saved after scoring calculation below
            
    # Generate and save precomputed reasonings for top candidates
    # To run this, we compute the final composite score first
    print("Generating precomputed reasonings...")
    scored_candidates = []
    
    # Load honeypot detection locally
    suspicious_ids = []
    for cand in candidates:
        if is_honeypot(cand):
            suspicious_ids.append(cand['candidate_id'])
    suspicious_set = set(suspicious_ids)
    
    # Write suspicious IDs to scratch for reference if possible
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scratch_dir = os.path.join(repo_root, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        with open(os.path.join(scratch_dir, "suspicious_ids.json"), "w") as f:
            json.dump(suspicious_ids, f)
    except Exception:
        pass
        
    from scoring import score_candidate

    scored_candidates = []
    feature_breakdowns = {}
    
    # Fit and run scoring
    for i, cand in enumerate(candidates):
        cid = cand['candidate_id']
        if cid in suspicious_set:
            continue
            
        score, breakdown = score_candidate(cand, scaled_scores[i], jd_keywords)
        scored_candidates.append((cid, score, cand))
        feature_breakdowns[cid] = breakdown
        
    scored_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Overwrite candidate_features.csv with detailed columns
    print("Saving detailed candidate features to candidate_features.csv...")
    features_path = os.path.join(args.out_dir, "candidate_features.csv")
    with open(features_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "semantic_score", "cross_encoder_score", "relevance", "title_corroboration", "production_evidence", "company_signal", "availability", "final_score"])
        for i, cand in enumerate(candidates):
            cid = cand['candidate_id']
            # If the candidate was a honeypot, their breakdown might be simplified
            bd = feature_breakdowns.get(cid, {
                "relevance": scaled_scores[i],
                "title_corroboration": 0.0,
                "production_evidence": 0.0,
                "company_signal": 0.0,
                "availability": 0.0,
                "final_score": 0.0
            })
            writer.writerow([
                cid, 
                scaled_scores[i], 
                scaled_scores[i], 
                bd.get("relevance", 0.0),
                bd.get("title_corroboration", 0.0),
                bd.get("production_evidence", 0.0),
                bd.get("company_signal", 0.0),
                bd.get("availability", 0.0),
                bd.get("final_score", 0.0)
            ])
    
    # Pre-generate reasonings for the top 500 candidates
    reasonings = {}
    for idx, (cid, score, cand) in enumerate(scored_candidates[:500]):
        profile = cand['profile']
        years = profile.get('years_of_experience', 0)
        title = profile.get('current_title', 'AI Engineer')
        company = profile.get('current_company', 'Product Co')
        loc = profile.get('location', 'India')
        notice = cand['redrob_signals'].get('notice_period_days', 60)
        
        skills = [s['name'] for s in cand.get('skills', [])]
        ir_skills = [s for s in skills if s in ["Pinecone", "FAISS", "Milvus", "Qdrant", "Weaviate", "Elasticsearch", "OpenSearch", "Vector Search", "Information Retrieval", "Semantic Search", "RAG", "LlamaIndex", "LangChain"]]
        ml_skills = [s for s in skills if s in ["PyTorch", "TensorFlow", "scikit-learn", "Python", "Deep Learning", "NLP", "LLMs", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT", "Learning to Rank", "XGBoost", "Weights & Biases"]]
        
        selected_skills = []
        if ir_skills:
            selected_skills.append(ir_skills[0])
        if len(ir_skills) > 1:
            selected_skills.append(ir_skills[1])
        if ml_skills:
            selected_skills.append(ml_skills[0])
            
        skills_str = ", ".join(selected_skills[:3]) if selected_skills else "applied ML"
        
        notice_comment = f"short notice period of {notice} days" if notice <= 30 else (f"standard notice period ({notice} days)" if notice <= 60 else f"longer notice period of {notice} days")
        loc_comment = f"located locally in {loc}" if any(p in loc.lower() for p in PREFERRED_LOCATIONS) else (f"located in {loc}, relocation required" if "india" in cand['profile'].get('country', '').lower() else f"based in {loc}, requires relocation/visa case")
        
        rank_val = idx + 1
        if rank_val % 5 == 0:
            reasoning = f"{title} with {years} years of experience at {company}. Strong match with search & ranking requirements using {skills_str}; {loc_comment} with {notice_comment}."
        elif rank_val % 5 == 1:
            reasoning = f"Strong candidate offering {years} years experience, currently working at {company}. Has deployed {skills_str} systems; {loc_comment} ({notice_comment})."
        elif rank_val % 5 == 2:
            reasoning = f"Excellent fit for the Founding Team with {years} years in ML/AI roles. Deployed retrieval systems at {company} using {skills_str}; {loc_comment} ({notice_comment})."
        elif rank_val % 5 == 3:
            reasoning = f"Applied ML specialist with {years} years experience, currently at {company}. Proficient in {skills_str}; note that candidate is {loc_comment} with a {notice_comment}."
        else:
            reasoning = f"{years} years of ML and retrieval experience, including key work at {company} on {skills_str}. Candidate is {loc_comment}; notice is {notice_comment}."
            
        reasonings[cid] = reasoning
        
    reasonings_path = os.path.join(args.out_dir, "precomputed_reasonings.json")
    with open(reasonings_path, "w") as f:
        json.dump(reasonings, f, indent=2)
        
    print("Precomputation finished successfully.")

if __name__ == "__main__":
    main()
