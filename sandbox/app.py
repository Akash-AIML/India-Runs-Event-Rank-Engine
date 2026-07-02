import streamlit as st
import pandas as pd
import json
import os
import sys
import csv
from datetime import datetime

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from parse_jd import get_jd_keywords
from features import is_honeypot, is_consulting_only
from scoring import score_candidate
from reasoning import get_reasoning
from embeddings import tokenize, BM25Model

st.set_page_config(
    page_title="Redrob Candidate Discovery Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Harmonious CSS styling
st.markdown("""
<style>
    .main-title {
        color: #ff4b4b;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6d7a8a;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Redrob Intelligent Candidate Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Calibrated Ranking Pipeline & Dynamic Feature Audit Console</div>', unsafe_allow_html=True)

# Default paths
DEFAULT_CANDIDATES_PATH = "./data/candidates.jsonl"
DEFAULT_JD_PATH = "./data/job_description.docx"
DEFAULT_ARTIFACTS_DIR = "./artifacts"

# Sidebar controls
st.sidebar.header("📁 Data Sourcing")
use_default = st.sidebar.checkbox("Use default dataset (data/candidates.jsonl)", value=True)

candidates_file = None
if not use_default:
    candidates_file = st.sidebar.file_uploader("Upload candidates.jsonl", type=["jsonl"])

jd_file = st.sidebar.file_uploader("Upload job_description.docx (Optional)", type=["docx"])

# Weights customizer
st.sidebar.header("🎛️ Dynamic Weights Tuning")
w_title = st.sidebar.slider("Title Corroboration Weight", 0.0, 1.0, 0.35, 0.05)
w_prod = st.sidebar.slider("Production Evidence Weight", 0.0, 1.0, 0.25, 0.05)
w_rel = st.sidebar.slider("Lexical Relevance (BM25) Weight", 0.0, 1.0, 0.25, 0.05)
w_comp = st.sidebar.slider("Company Signal Weight", 0.0, 1.0, 0.15, 0.05)

st.sidebar.header("🛫 Logistic Constraints")
w_availability = st.sidebar.slider("Availability & Relocation Factor", 0.0, 1.0, 0.25, 0.05)

run_button = st.sidebar.button("Run Ranking Discovery", type="primary")

# Load data
@st.cache_data
def load_candidates(path_or_file):
    candidates = []
    if isinstance(path_or_file, str):
        if not os.path.exists(path_or_file):
            return []
        with open(path_or_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        # File uploader
        for line in path_or_file:
            if line.strip():
                candidates.append(json.loads(line.decode("utf-8")))
    return candidates

# Execution logic
candidates = []
if use_default:
    candidates = load_candidates(DEFAULT_CANDIDATES_PATH)
elif candidates_file is not None:
    candidates = load_candidates(candidates_file)

if not candidates:
    st.info("Please upload a candidate dataset or ensure data/candidates.jsonl exists in the directory.")
else:
    # Resolve JD path
    jd_path = DEFAULT_JD_PATH
    if jd_file is not None:
        # Save temp file
        temp_jd_path = "./data/temp_jd.docx"
        os.makedirs("./data", exist_ok=True)
        with open(temp_jd_path, "wb") as f:
            f.write(jd_file.getbuffer())
        jd_path = temp_jd_path
        
    # Run pipeline
    if run_button or 'ranked_df' not in st.session_state:
        with st.spinner("Processing matching candidates and generating scoring breakdowns..."):
            jd_keywords = get_jd_keywords(jd_path)
            
            # Retrieve cached scores if available
            precomputed_scores = {}
            features_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "candidate_features.csv")
            if os.path.exists(features_path):
                try:
                    with open(features_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            if len(row) >= 2:
                                precomputed_scores[row[0]] = float(row[1])
                except Exception:
                    pass
            
            # If no cached scores, calculate live BM25
            if not precomputed_scores:
                tokenized_docs = []
                for cand in candidates:
                    profile = cand.get('profile', {})
                    text_parts = [profile.get('headline', ''), profile.get('summary', ''), profile.get('current_title', '')]
                    for job in cand.get('career_history', []):
                        text_parts.append(job.get('title', '') + " " + job.get('description', ''))
                    tokenized_docs.append(tokenize(" ".join(text_parts)))
                    
                bm25 = BM25Model()
                bm25.fit(tokenized_docs, jd_keywords)
                live_scores = [bm25.score(doc, jd_keywords) for doc in tokenized_docs]
                max_s = max(live_scores) if live_scores else 1.0
                min_s = min(live_scores) if live_scores else 0.0
                
                for idx, cand in enumerate(candidates):
                    cid = cand['candidate_id']
                    s_val = live_scores[idx]
                    precomputed_scores[cid] = (s_val - min_s) / (max_s - min_s) if max_s > min_s else 0.0
            
            # Load reasonings
            precomputed_reasonings = {}
            reasonings_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "precomputed_reasonings.json")
            if os.path.exists(reasonings_path):
                try:
                    with open(reasonings_path, "r", encoding="utf-8") as f:
                        precomputed_reasonings = json.load(f)
                except Exception:
                    pass
                    
            # Compute scores and filters
            processed_list = []
            honeypot_count = 0
            consulting_count = 0
            
            for cand in candidates:
                cid = cand['candidate_id']
                if is_honeypot(cand):
                    honeypot_count += 1
                    continue
                if is_consulting_only(cand):
                    consulting_count += 1
                    
                sem_score = precomputed_scores.get(cid, 0.0)
                
                # Dynamic scoring calculation with custom sliders weights
                # Get raw sub-features
                from scoring import get_title_corroboration_score, get_production_evidence_score, get_skill_stuffing_penalty
                from features import get_notice_score, get_location_score, get_redrob_signals_multiplier, parse_date
                
                title_c = get_title_corroboration_score(cand)
                prod_e = get_production_evidence_score(cand)
                comp_s = 0.1 if is_consulting_only(cand) else 1.0
                
                # Combine core fit using slider weights
                total_w = w_rel + w_title + w_prod + w_comp
                norm_w_rel = w_rel / total_w
                norm_w_title = w_title / total_w
                norm_w_prod = w_prod / total_w
                norm_w_comp = w_comp / total_w
                
                core_fit = (norm_w_rel * sem_score + norm_w_title * title_c + norm_w_prod * prod_e + norm_w_comp * comp_s) * 100.0
                core_fit *= get_skill_stuffing_penalty(cand)
                
                # Availability
                notice_s = get_notice_score(cand)
                loc_s = get_location_score(cand)
                
                signals = cand.get('redrob_signals', {})
                last_act = parse_date(signals.get("last_active_date"))
                if last_act:
                    days_inactive = (datetime(2026, 7, 2) - last_act).days
                    active_s = 1.0 if days_inactive <= 30 else (0.8 if days_inactive <= 90 else (0.5 if days_inactive <= 180 else 0.1))
                else:
                    active_s = 0.5
                resp_s = signals.get("recruiter_response_rate", 0.5)
                
                availability = (0.3 * notice_s + 0.3 * loc_s + 0.2 * active_s + 0.2 * resp_s)
                if signals.get("open_to_work_flag", False):
                    availability = min(availability * 1.1, 1.0)
                    
                # Bounded score
                final_score = core_fit * ((1.0 - w_availability) + w_availability * availability)
                
                reasoning = get_reasoning(cid, cand, precomputed_reasonings)
                
                processed_list.append({
                    "candidate_id": cid,
                    "name": cand['profile'].get('anonymized_name', 'Secret Candidate'),
                    "score": round(final_score, 4),
                    "title": cand['profile'].get('current_title', 'AI Engineer'),
                    "company": cand['profile'].get('current_company', 'Product Co'),
                    "location": cand['profile'].get('location', 'India'),
                    "notice_days": cand['redrob_signals'].get('notice_period_days', 60),
                    "reasoning": reasoning,
                    "relevance": round(sem_score, 2),
                    "title_corrob": round(title_c, 2),
                    "prod_evidence": round(prod_e, 2),
                    "company_signal": round(comp_s, 2),
                    "availability": round(availability, 2),
                    "raw_cand": cand
                })
                
            # Sort
            processed_list.sort(key=lambda x: (-x['score'], x['candidate_id']))
            
            # Store in session
            st.session_state.ranked_list = processed_list
            st.session_state.honeypot_count = honeypot_count
            st.session_state.consulting_count = consulting_count
            
        st.success("Ranking successfully completed!")
        
    # Render layout
    ranked_list = st.session_state.ranked_list
    
    # 1. Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{len(candidates)}</div><div class="metric-label">Total Pool</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{len(ranked_list)}</div><div class="metric-label">Valid Scored</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #ff4b4b;">{st.session_state.honeypot_count}</div><div class="metric-label">Disqualified Honeypots</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #f59e0b;">{st.session_state.consulting_count}</div><div class="metric-label">IT Services Flagged</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render candidate list
    col_table, col_inspect = st.columns([3, 2])
    
    with col_table:
        st.subheader("🏆 Candidate Leaderboard (Top 100)")
        
        # Build pandas dataframe for top 100
        display_data = []
        for idx, item in enumerate(ranked_list[:100]):
            display_data.append({
                "Rank": idx + 1,
                "Candidate ID": item["candidate_id"],
                "Name": item["name"],
                "Score": item["score"],
                "Current Title": item["title"],
                "Company": item["company"]
            })
            
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download block
        csv_buffer = []
        csv_buffer.append(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(ranked_list[:100]):
            csv_buffer.append([item["candidate_id"], idx + 1, item["score"], item["reasoning"]])
            
        csv_string = ""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(csv_buffer)
        csv_string = output.getvalue()
        
        st.download_button(
            label="💾 Download submission.csv",
            data=csv_string,
            file_name="submission.csv",
            mime="text/csv",
            type="primary"
        )
        
    with col_inspect:
        st.subheader("🔍 Profile Audit & Inspection Console")
        selected_index = st.selectbox(
            "Select Candidate to inspect:",
            range(min(len(ranked_list), 100)),
            format_func=lambda x: f"Rank {x+1}: {ranked_list[x]['name']} ({ranked_list[x]['candidate_id']})"
        )
        
        if selected_index is not None and len(ranked_list) > 0:
            cand_data = ranked_list[selected_index]
            raw = cand_data["raw_cand"]
            profile = raw.get('profile', {})
            
            st.markdown(f"### **{cand_data['name']}**")
            st.markdown(f"**Current Role**: {cand_data['title']} at *{cand_data['company']}*")
            st.markdown(f"**Location**: {cand_data['location']} | **Notice Period**: {cand_data['notice_days']} days")
            
            st.divider()
            
            # Score feature breakdown visualizer
            st.markdown("#### **Calibrated Feature Score Breakdown**")
            
            st.markdown(f"**BM25 Relevance Match**: `{cand_data['relevance']}`")
            st.progress(float(cand_data['relevance']))
            
            st.markdown(f"**Title Corroboration**: `{cand_data['title_corrob']}`")
            st.progress(float(cand_data['title_corrob']))
            
            st.markdown(f"**Production Evidence**: `{cand_data['prod_evidence']}`")
            st.progress(float(cand_data['prod_evidence']))
            
            st.markdown(f"**Availability Score**: `{cand_data['availability']}`")
            st.progress(float(cand_data['availability']))
            
            st.divider()
            
            st.markdown("#### **RAG Reasoning Justification**")
            st.info(cand_data["reasoning"])
            
            st.divider()
            
            st.markdown("#### **Career History Highlights**")
            for job in raw.get('career_history', []):
                st.markdown(f"📌 **{job.get('title')}** at **{job.get('company')}** ({job.get('duration_months')} months)")
                st.caption(job.get('description'))
                st.markdown("")
