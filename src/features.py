import json
from datetime import datetime

# IT Services / Consulting companies to exclude
IT_SERVICES = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 
    'tech mahindra', 'l&t', 'mindtree', 'mphasis', 'hcl', 'genpact', 
    'tata consultancy services', 'cognizant technology solutions', 
    'infosys limited', 'wipro limited', 'tata consultancy'
}

PREFERRED_LOCATIONS = {'noida', 'pune', 'mumbai', 'delhi', 'hyderabad', 'gurgaon'}

def parse_date(date_str):
    if not date_str:
        return datetime(2026, 7, 2)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def is_honeypot(cand):
    # Check 1: Job Date/Duration mismatch
    career = cand.get("career_history", [])
    total_duration_months = 0
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        dur = job.get("duration_months", 0)
        total_duration_months += dur
        
        if start and end and start > end:
            return True
            
        if start and end:
            diff_months = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(diff_months - dur) > 3:
                return True
                
    # Check 2: School dates mismatch
    education = cand.get("education", [])
    for edu in education:
        start_yr = edu.get("start_year")
        end_yr = edu.get("end_year")
        if start_yr and end_yr and start_yr > end_yr:
            return True
            
    # Check 3: Expert/advanced skill with 0 duration
    skills = cand.get("skills", [])
    expert_zero = [s for s in skills if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 0) == 0]
    if len(expert_zero) > 0:
        return True
        
    # Check 4: Experience vs career duration mismatch
    years_exp = cand['profile'].get('years_of_experience', 0)
    if abs(years_exp - (total_duration_months / 12.0)) > 5.0:
        return True
        
    return False

def is_consulting_only(cand):
    career = cand.get("career_history", [])
    all_consulting = True
    has_any_job = False
    for job in career:
        has_any_job = True
        comp = job.get("company", "").lower().strip()
        is_services = False
        for svc in IT_SERVICES:
            if svc in comp:
                is_services = True
                break
        if not is_services:
            all_consulting = False
    return has_any_job and all_consulting

def get_experience_score(cand):
    years = cand['profile'].get('years_of_experience', 0)
    if 5.0 <= years <= 9.0:
        return 1.0
    elif 4.0 <= years < 5.0:
        return 0.8
    elif 9.0 < years <= 12.0:
        return 0.8
    elif 3.0 <= years < 4.0:
        return 0.5
    elif 12.0 < years <= 15.0:
        return 0.5
    else:
        return 0.1

def get_location_score(cand):
    profile = cand['profile']
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = profile.get("willing_to_relocate", False)
    
    is_pref_loc = any(p in loc for p in PREFERRED_LOCATIONS)
    is_india = (country == "india")
    
    if is_india:
        if is_pref_loc:
            return 1.0
        elif willing:
            return 0.8
        else:
            return 0.3
    else:
        if willing:
            return 0.1
        else:
            return 0.0

def get_notice_score(cand):
    notice = cand['redrob_signals'].get('notice_period_days', 180)
    if notice <= 30:
        return 1.0
    elif notice <= 60:
        return 0.8
    elif notice <= 90:
        return 0.5
    elif notice <= 120:
        return 0.2
    else:
        return 0.0

def get_redrob_signals_multiplier(cand):
    m = 1.0
    signals = cand['redrob_signals']
    
    # Active check
    last_act = parse_date(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 7, 2) - last_act).days
        if days_inactive <= 30:
            m *= 1.2
        elif days_inactive <= 90:
            m *= 1.0
        elif days_inactive <= 180:
            m *= 0.7
        else:
            m *= 0.3
            
    # Response rate
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    if resp_rate >= 0.8:
        m *= 1.2
    elif resp_rate < 0.2:
        m *= 0.5
        
    if signals.get("open_to_work_flag", False):
        m *= 1.1
        
    gh = signals.get("github_activity_score", -1)
    if gh > 50:
        m *= 1.15
    elif gh > 20:
        m *= 1.05
    elif gh == -1:
        m *= 0.9
        
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        m *= 1.1
        
    completion = signals.get("interview_completion_rate", 0.0)
    if completion >= 0.8:
        m *= 1.1
    elif completion < 0.4:
        m *= 0.7
        
    return m

def get_stuffing_penalty(cand, matching_skills):
    current_title = cand['profile'].get("current_title", "").lower()
    is_unrelated_title = any(un in current_title for un in ["marketing", "accountant", "hr", "operations", "sales", "finance", "support", "writer"])
    if is_unrelated_title and matching_skills >= 3:
        return 0.05
    return 1.0
