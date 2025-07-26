from pydantic import BaseModel
from typing import Optional, List

class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    experience: str
    education: str
    target_job: Optional[str] = None

class JobPosting(BaseModel):
    title: str
    description: str
    required_skills: List[str]
    salary_range: Optional[str] = None

class ResumeMatchResult(BaseModel):
    score: float
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]