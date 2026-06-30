import unittest
import json
from datetime import datetime
from src.features import is_honeypot, parse_date, IT_SERVICES

class TestCandidateRanker(unittest.TestCase):
    
    def setUp(self):
        # Sample normal candidate template
        self.normal_candidate = {
            "candidate_id": "CAND_9999999",
            "profile": {
                "anonymized_name": "Test Candidate",
                "headline": "AI Engineer",
                "summary": "Building retrieval systems.",
                "years_of_experience": 5.0,
                "current_title": "AI Engineer",
                "current_company": "Pied Piper",
                "location": "Pune, Maharashtra",
                "country": "India"
            },
            "career_history": [
                {
                    "company": "Pied Piper",
                    "title": "AI Engineer",
                    "start_date": "2022-07-02",
                    "end_date": None,
                    "duration_months": 48, # 4 years = 48 months
                    "is_current": True,
                    "description": "Building RAG search."
                }
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2018,
                    "end_year": 2022
                }
            ],
            "skills": [
                {
                    "name": "Python",
                    "proficiency": "expert",
                    "endorsements": 10,
                    "duration_months": 48
                }
            ],
            "redrob_signals": {
                "notice_period_days": 30,
                "last_active_date": "2026-06-20",
                "recruiter_response_rate": 0.8,
                "open_to_work_flag": True
            }
        }

    def test_normal_candidate_is_not_honeypot(self):
        self.assertFalse(is_honeypot(self.normal_candidate))

    def test_honeypot_with_date_mismatch(self):
        # Mismatch between start_date/end_date and duration_months
        bad_cand = json.loads(json.dumps(self.normal_candidate))
        bad_cand["career_history"][0]["start_date"] = "2024-07-02" # 2 years calculated, but duration is 48 months
        self.assertTrue(is_honeypot(bad_cand))

    def test_honeypot_with_expert_skill_zero_duration(self):
        # Expert skill with 0 duration
        bad_cand = json.loads(json.dumps(self.normal_candidate))
        bad_cand["skills"].append({
            "name": "RAG",
            "proficiency": "expert",
            "endorsements": 0,
            "duration_months": 0
        })
        self.assertTrue(is_honeypot(bad_cand))

    def test_honeypot_with_experience_mismatch(self):
        # years_of_experience is 15.0 but career history duration is only 4.0 years (48 months)
        bad_cand = json.loads(json.dumps(self.normal_candidate))
        bad_cand["profile"]["years_of_experience"] = 15.0
        self.assertTrue(is_honeypot(bad_cand))

    def test_honeypot_with_school_dates_mismatch(self):
        # start_year > end_year
        bad_cand = json.loads(json.dumps(self.normal_candidate))
        bad_cand["education"][0]["start_year"] = 2023
        bad_cand["education"][0]["end_year"] = 2022
        self.assertTrue(is_honeypot(bad_cand))

    def test_it_services_company_check(self):
        # Verifies that IT_SERVICES set includes common services companies
        self.assertIn("tcs", IT_SERVICES)
        self.assertIn("infosys", IT_SERVICES)
        self.assertIn("wipro", IT_SERVICES)
        self.assertIn("accenture", IT_SERVICES)

    def test_parse_date_correctness(self):
        self.assertEqual(parse_date("2026-07-02"), datetime(2026, 7, 2))
        self.assertIsNone(parse_date("invalid-date"))

if __name__ == "__main__":
    unittest.main()
