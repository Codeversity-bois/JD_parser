from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class JobDescriptionInput(BaseModel):
    """Input schema for job description parsing"""
    job_description: str = Field(..., description="Raw job description text")
    job_title: Optional[str] = Field(None, description="Job title (optional)")
    company: Optional[str] = Field(None, description="Company name (optional)")
    location: Optional[str] = Field(None, description="Job location (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "We are looking for a Senior Python Developer with 5+ years of experience...",
                "job_title": "Senior Python Developer",
                "company": "Tech Corp",
                "location": "Remote"
            }
        }


class ParsedJobResponse(BaseModel):
    """Response schema for parsed job description"""
    job_id: str
    title: str
    company: Optional[str]
    location: Optional[str]
    
    # Parsed fields with their collection IDs
    skills_id: str
    experience_id: str
    education_id: Optional[str]
    responsibilities_id: str
    benefits_id: Optional[str]
    job_details_id: str
    
    # Actual parsed data
    parsed_data: Dict[str, Any]
    
    message: str = "Job description parsed and stored successfully"
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "title": "Senior Python Developer",
                "company": "Tech Corp",
                "location": "Remote",
                "skills_id": "skills_xyz789",
                "experience_id": "exp_def456",
                "education_id": "edu_ghi789",
                "responsibilities_id": "resp_jkl012",
                "benefits_id": "ben_mno345",
                "job_details_id": "details_pqr678",
                "parsed_data": {
                    "required_skills": ["Python", "FastAPI"],
                    "experience_years": "5+",
                    "responsibilities": ["Design systems", "Write code"],
                    "benefits": ["Health insurance", "Remote work"]
                },
                "message": "Job description parsed and stored successfully"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    database: str
    faiss_vectors: int
