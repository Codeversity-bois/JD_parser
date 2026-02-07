import re
import json
from typing import Dict, Any, List
import logging
import requests

from config import config
from models import JobDescription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JDParser:
    """Parser for extracting structured data from job descriptions"""
    
    def __init__(self):
        """Initialize the parser"""
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
    
    def parse_with_llm(self, job_description: str) -> Dict[str, Any]:
        """
        Parse job description using LLM (OpenRouter)
        
        Args:
            job_description: Raw job description text
            
        Returns:
            Structured job data dictionary
        """
        try:
            url = f"{self.base_url}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
Extract the following information from the job description below. Return the data as a valid JSON object.

Required fields:
- title: Job title
- company: Company name (if mentioned)
- location: Job location (if mentioned)
- required_skills: List of required skills
- preferred_skills: List of preferred/nice-to-have skills
- experience_years: Required years of experience (e.g., "3-5 years", "5+", etc.)
- education: Education requirements
- responsibilities: List of main responsibilities
- benefits: List of benefits offered
- job_type: Type of job (Full-time, Part-time, Contract, etc.)
- salary_range: Salary range if mentioned

Job Description:
{job_description}

Return only the JSON object, no additional text.
"""
            
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from job descriptions. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON response
            parsed_data = json.loads(content)
            
            logger.info("Successfully parsed job description with LLM")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing with LLM: {e}")
            # Fallback to rule-based parsing
            return self.parse_with_rules(job_description)
    
    def parse_with_rules(self, job_description: str) -> Dict[str, Any]:
        """
        Fallback rule-based parsing
        
        Args:
            job_description: Raw job description text
            
        Returns:
            Structured job data dictionary
        """
        logger.info("Using rule-based parsing as fallback")
        
        # Basic extraction using regex patterns
        data = {
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": None,
            "education": None,
            "responsibilities": [],
            "benefits": [],
            "job_type": None,
            "salary_range": None
        }
        
        # Extract skills (common programming languages and technologies)
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin)\b',
            r'\b(React|Angular|Vue|Node\.js|Django|Flask|FastAPI|Spring|Express)\b',
            r'\b(MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|DynamoDB)\b',
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|CI/CD)\b',
            r'\b(Git|GitHub|GitLab|Jira|Agile|Scrum)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            data["required_skills"].extend(matches)
        
        # Remove duplicates
        data["required_skills"] = list(set(data["required_skills"]))
        
        # Extract experience
        exp_pattern = r'(\d+[\+\-]?\s*(?:to|-)?\s*\d*\s*(?:years?|yrs?))'
        exp_match = re.search(exp_pattern, job_description, re.IGNORECASE)
        if exp_match:
            data["experience_years"] = exp_match.group(1)
        
        # Extract education
        edu_patterns = [
            r"(Bachelor'?s?|Master'?s?|PhD|Doctorate) (?:degree )?(?:in )?([\w\s]+)",
            r"(BS|MS|MBA|PhD) in ([\w\s]+)"
        ]
        for pattern in edu_patterns:
            edu_match = re.search(pattern, job_description, re.IGNORECASE)
            if edu_match:
                data["education"] = edu_match.group(0)
                break
        
        # Extract job type
        job_type_pattern = r'\b(Full-time|Part-time|Contract|Freelance|Remote|Hybrid)\b'
        job_type_match = re.search(job_type_pattern, job_description, re.IGNORECASE)
        if job_type_match:
            data["job_type"] = job_type_match.group(1)
        
        # Extract salary range
        salary_pattern = r'\$\s*\d+[,\d]*\s*(?:-|to)\s*\$?\s*\d+[,\d]*'
        salary_match = re.search(salary_pattern, job_description)
        if salary_match:
            data["salary_range"] = salary_match.group(0)
        
        return data
    
    def create_job_object(
        self,
        job_id: str,
        title: str,
        description: str,
        company: str = None,
        location: str = None,
        use_llm: bool = True
    ) -> JobDescription:
        """
        Create a JobDescription object from raw data
        
        Args:
            job_id: Unique job identifier
            title: Job title
            description: Full job description
            company: Company name
            location: Job location
            use_llm: Whether to use LLM for parsing (default: True)
            
        Returns:
            JobDescription object
        """
        try:
            # Parse the description
            if use_llm:
                parsed_data = self.parse_with_llm(description)
            else:
                parsed_data = self.parse_with_rules(description)
            
            # Create JobDescription object
            job = JobDescription(
                job_id=job_id,
                title=title,
                company=company or parsed_data.get('company'),
                location=location or parsed_data.get('location'),
                description=description,
                required_skills=parsed_data.get('required_skills', []),
                preferred_skills=parsed_data.get('preferred_skills', []),
                experience_years=parsed_data.get('experience_years'),
                education=parsed_data.get('education'),
                responsibilities=parsed_data.get('responsibilities', []),
                benefits=parsed_data.get('benefits', []),
                job_type=parsed_data.get('job_type'),
                salary_range=parsed_data.get('salary_range')
            )
            
            logger.info(f"Created JobDescription object for {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating job object: {e}")
            raise
