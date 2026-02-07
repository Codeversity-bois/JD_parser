from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime


class Project(BaseModel):
    """Model for candidate project"""
    name: str = Field(..., description="Project name")
    description: str = Field(..., max_length=500, description="Project description (max 500 words)")
    github_link: HttpUrl = Field(..., description="GitHub repository link")
    technologies: Optional[List[str]] = Field(default_factory=list, description="Technologies used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "E-commerce Platform",
                "description": "Built a full-stack e-commerce platform with React, Node.js, and MongoDB...",
                "github_link": "https://github.com/username/ecommerce-platform",
                "technologies": ["React", "Node.js", "MongoDB", "Express"]
            }
        }


class Education(BaseModel):
    """Model for candidate education"""
    degree: str = Field(..., description="Degree name")
    field_of_study: str = Field(..., description="Field of study")
    institution: str = Field(..., description="Institution name")
    graduation_year: Optional[int] = Field(None, description="Year of graduation")
    gpa: Optional[float] = Field(None, description="GPA or percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "degree": "Bachelor of Technology",
                "field_of_study": "Computer Science",
                "institution": "ABC University",
                "graduation_year": 2023,
                "gpa": 3.8
            }
        }


class CandidateProfile(BaseModel):
    """Complete candidate profile model"""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    leetcode_username: str = Field(..., description="LeetCode username")
    leetcode_stats: Optional[Dict[str, Any]] = Field(None, description="LeetCode profile stats")
    projects: List[Project] = Field(..., min_length=2, description="At least 2 projects")
    resume_text: str = Field(..., description="Resume content as text")
    education: List[Education] = Field(..., min_length=1, description="Education details")
    interview_questions: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Interview-specific questions and answers"
    )
    
    # Metadata
    embeddings: Optional[Dict[str, List[float]]] = Field(None, description="Embeddings for different sections")
    faiss_indices: Optional[Dict[str, int]] = Field(None, description="FAISS index IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "candidate_001",
                "leetcode_username": "john_doe",
                "projects": [
                    {
                        "name": "ML Pipeline",
                        "description": "Built an automated ML pipeline...",
                        "github_link": "https://github.com/john/ml-pipeline",
                        "technologies": ["Python", "TensorFlow", "Docker"]
                    }
                ],
                "resume_text": "Experienced software engineer with 5 years...",
                "education": [
                    {
                        "degree": "B.Tech",
                        "field_of_study": "Computer Science",
                        "institution": "XYZ University",
                        "graduation_year": 2020,
                        "gpa": 3.7
                    }
                ],
                "interview_questions": {
                    "Why this company?": "I am passionate about...",
                    "Career goals": "I aim to become..."
                }
            }
        }


class CandidateInput(BaseModel):
    """Input schema for candidate profile submission"""
    leetcode_username: str = Field(..., description="LeetCode username")
    projects: List[Project] = Field(..., min_length=2, description="At least 2 projects")
    resume_text: str = Field(..., description="Resume content as text")
    education: List[Education] = Field(..., min_length=1, description="Education details")
    interview_questions: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Interview-specific questions and answers"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "leetcode_username": "john_doe",
                "projects": [
                    {
                        "name": "ML Pipeline",
                        "description": "Built an automated ML pipeline using Python and TensorFlow...",
                        "github_link": "https://github.com/john/ml-pipeline",
                        "technologies": ["Python", "TensorFlow"]
                    },
                    {
                        "name": "Web Dashboard",
                        "description": "Created a real-time analytics dashboard...",
                        "github_link": "https://github.com/john/dashboard",
                        "technologies": ["React", "FastAPI"]
                    }
                ],
                "resume_text": "Software Engineer with 5 years of experience...",
                "education": [
                    {
                        "degree": "Bachelor of Technology",
                        "field_of_study": "Computer Science",
                        "institution": "ABC University",
                        "graduation_year": 2020,
                        "gpa": 3.7
                    }
                ],
                "interview_questions": {
                    "Why this company?": "I am passionate about innovation...",
                    "Career goals": "I aim to become a tech lead..."
                }
            }
        }


class MatchResult(BaseModel):
    """Model for job matching result"""
    job_id: str
    job_title: str
    company: Optional[str]
    location: Optional[str]
    similarity_score: float
    matched_skills: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    

class ShortlistResult(BaseModel):
    """Model for shortlisting result"""
    candidate_id: str
    total_matches: int
    shortlisted_jobs: List[MatchResult]
    eliminated_count: int
    status: str = "shortlisted"


class FinalEvaluationResult(BaseModel):
    """Model for final LLM evaluation result"""
    candidate_id: str
    job_id: str
    job_title: str
    similarity_score: float
    llm_evaluation: Dict[str, Any]
    final_score: float
    recommendation: str
    reasoning: str
    proceed_to_oa: bool


class CandidateEvaluationResponse(BaseModel):
    """Response for complete candidate evaluation"""
    candidate_id: str
    leetcode_username: str
    total_jobs_analyzed: int
    initial_matches: int
    after_60_percent_filter: int
    final_recommendations: List[FinalEvaluationResult]
    message: str
