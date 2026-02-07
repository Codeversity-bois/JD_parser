from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    """Model for Job Description data"""
    job_id: str = Field(..., description="Unique identifier for the job")
    title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    description: str = Field(..., description="Full job description text")
    
    # Extracted fields
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    experience_years: Optional[str] = Field(None, description="Required years of experience")
    education: Optional[str] = Field(None, description="Education requirements")
    responsibilities: List[str] = Field(default_factory=list, description="Job responsibilities")
    benefits: List[str] = Field(default_factory=list, description="Benefits offered")
    job_type: Optional[str] = Field(None, description="Job type (Full-time, Part-time, etc.)")
    salary_range: Optional[str] = Field(None, description="Salary range if mentioned")
    
    # Metadata
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    faiss_index_id: Optional[int] = Field(None, description="FAISS index ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "title": "Senior Python Developer",
                "company": "Tech Corp",
                "location": "Remote",
                "description": "We are looking for a senior Python developer...",
                "required_skills": ["Python", "FastAPI", "MongoDB"],
                "experience_years": "5+",
                "job_type": "Full-time"
            }
        }


class SearchQuery(BaseModel):
    """Model for search queries"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, description="Number of results to return")
    
    
class SearchResult(BaseModel):
    """Model for search results"""
    job_id: str
    title: str
    company: Optional[str]
    location: Optional[str]
    description: str
    score: float
    matched_skills: List[str] = Field(default_factory=list)
