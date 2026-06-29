import sys
import os
from datetime import datetime

# Add src to python path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import get_experience_score, get_location_score, get_notice_score, get_redrob_signals_multiplier, get_stuffing_penalty, is_honeypot, is_consulting_only, parse_date

def get_title_corroboration_score(cand):
    profile = cand.get('profile', {})
    current_title = profile.get('current_title', '').lower()
    
    career = cand.get('career_history', [])
    past_titles = [job.get('title', '').lower() for job in career]
    all_titles = [current_title] + past_titles
    
    # Check current title
    if any(kw in current_title for kw in ["search engineer", "retrieval engineer", "ranking engineer", "search & ranking"]):
        return 1.0
    elif any(kw in current_title for kw in ["senior machine learning", "senior ml", "lead ml", "lead machine learning", "principal ml"]):
        return 0.9
    elif any(kw in current_title for kw in ["machine learning engineer", "ml engineer", "nlp engineer", "ai engineer", "applied scientist"]):
        return 0.8
    elif any(kw in current_title for kw in ["data scientist", "research engineer"]):
        return 0.6
        
    # Check past titles
    has_past_search = any(any(kw in t for kw in ["search", "retrieval", "ranking"]) for t in past_titles)
    has_past_ml = any(any(kw in t for kw in ["ml", "machine learning", "nlp", "ai"]) for t in past_titles)
    
    if has_past_search:
        return 0.7
    elif has_past_ml:
        return 0.5
    elif any(kw in current_title for kw in ["software engineer", "developer", "engineer"]):
        return 0.3
    else:
        return 0.1

def get_production_evidence_score(cand):
    career = cand.get('career_history', [])
    evidence_verbs = [
        "deployed", "production", "served at scale", "latency", "throughput", 
        "pipeline", "a/b test", "a/b tested", "a/b testing", "ndcg", "mrr", 
        "map", "scale", "qps", "online inference", "retrieval pipeline", 
        "search latency", "vector index", "hybrid retrieval"
    ]
    
    matches_count = 0
    total_words = 0
    
    for job in career:
        desc = job.get('description', '').lower()
        title = job.get('title', '').lower()
        full_text = desc + " " + title
        total_words += len(full_text.split())
        for verb in evidence_verbs:
            if verb in full_text:
                matches_count += 1
                
    if total_words == 0:
        return 0.0
        
    # Calibrate matching count (5 distinct terms represents very strong evidence)
    evidence_score = min(matches_count / 5.0, 1.0)
    return evidence_score

def get_skill_stuffing_penalty(cand):
    skills = cand.get('skills', [])
    skills_names = [s.get('name', '').lower() for s in skills]
    
    # List of key vector DBs and retrieval tools
    vector_dbs = ["pinecone", "milvus", "qdrant", "weaviate", "faiss", "elasticsearch", "opensearch", "pgvector"]
    listed_dbs = [db for db in vector_dbs if db in skills_names]
    
    # If they list 4+ different vector databases, check if they actually describe using them in their career history
    if len(listed_dbs) >= 4:
        career = cand.get('career_history', [])
        history_text = " ".join([job.get('description', '').lower() + " " + job.get('title', '').lower() for job in career])
        
        matches_in_history = sum(1 for db in listed_dbs if db in history_text)
        # If they list many DBs but use none of them in descriptions, it is a stuffing trap
        if matches_in_history == 0:
            return 0.1
            
    # Also check total skills vs description length
    career = cand.get('career_history', [])
    total_desc_len = sum(len(job.get('description', '')) for job in career)
    if len(skills) > 15 and total_desc_len < 150:
        return 0.2
        
    return 1.0

def score_candidate(cand, semantic_score, jd_keywords):
    if is_honeypot(cand):
        return 0.0, {
            "relevance": 0.0,
            "title_corroboration": 0.0,
            "production_evidence": 0.0,
            "company_signal": 0.0,
            "availability": 0.0,
            "final_score": 0.0,
            "disqualified": True,
            "reason": "Honeypot/Date Contradiction"
        }
        
    # 1. Relevance Score (Lexical BM25, 0-1)
    relevance = semantic_score
    
    # 2. Title Corroboration Score (0-1)
    title_corrob = get_title_corroboration_score(cand)
    
    # 3. Production Evidence Score (0-1)
    prod_evidence = get_production_evidence_score(cand)
    
    # 4. Company Signal (0-1)
    if is_consulting_only(cand):
        # Medium confidence penalty instead of hard exclusion
        company_signal = 0.1
    else:
        company_signal = 1.0
        
    # Compute Core Fit Score (0 to 100)
    core_fit = (0.25 * relevance + 0.35 * title_corrob + 0.25 * prod_evidence + 0.15 * company_signal) * 100.0
    
    # Apply Skill Stuffing Penalty
    stuffing_penalty = get_skill_stuffing_penalty(cand)
    core_fit *= stuffing_penalty
    
    # 5. Availability Score (0-1)
    notice_score = get_notice_score(cand)
    loc_score = get_location_score(cand)
    
    signals = cand.get('redrob_signals', {})
    
    # Active score
    last_act = parse_date(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 7, 2) - last_act).days
        active_score = 1.0 if days_inactive <= 30 else (0.8 if days_inactive <= 90 else (0.5 if days_inactive <= 180 else 0.1))
    else:
        active_score = 0.5
        
    # Response rate score
    resp_rate_score = signals.get("recruiter_response_rate", 0.5)
    
    # Bounded availability score
    availability = (0.3 * notice_score + 0.3 * loc_score + 0.2 * active_score + 0.2 * resp_rate_score)
    
    # Open to work boost
    if signals.get("open_to_work_flag", False):
        availability = min(availability * 1.1, 1.0)
        
    # 6. Combined Bounded Final Score
    final_score = core_fit * (0.75 + 0.25 * availability)
    
    breakdown = {
        "relevance": round(relevance, 4),
        "title_corroboration": round(title_corrob, 4),
        "production_evidence": round(prod_evidence, 4),
        "company_signal": round(company_signal, 4),
        "availability": round(availability, 4),
        "final_score": round(final_score, 4),
        "disqualified": False
    }
    
    return final_score, breakdown
