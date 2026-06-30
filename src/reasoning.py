import os
import json

def get_reasoning(cid, cand, precomputed_reasonings=None):
    if precomputed_reasonings and cid in precomputed_reasonings:
        return precomputed_reasonings[cid]
        
    # Fallback reasoning generator using candidate facts
    profile = cand.get('profile', {})
    title = profile.get('current_title', 'AI Engineer')
    years = profile.get('years_of_experience', 0)
    company = profile.get('current_company', 'Product Co')
    location = profile.get('location', 'India')
    notice = cand.get('redrob_signals', {}).get('notice_period_days', 60)
    
    return f"{title} with {years} years of experience at {company}, matching key requirements. Located in {location} ({notice} days notice)."
