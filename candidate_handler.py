from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from datetime import datetime
import logging
import uuid

from config import config
from candidate_models import CandidateProfile, Project, Education

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateHandler:
    """Handler for managing candidate profiles in MongoDB"""
    
    def __init__(self):
        """Initialize MongoDB connection for candidates"""
        try:
            self.client = MongoClient(config.MONGODB_URL)
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas for candidates")
            
            self.db = self.client[config.MONGODB_DB_NAME]
            
            # Define candidate collections
            self.candidates_collection = self.db['candidates']
            self.candidate_projects_collection = self.db['candidate_projects']
            self.candidate_education_collection = self.db['candidate_education']
            self.candidate_skills_collection = self.db['candidate_skills']
            
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for candidate collections"""
        try:
            # Main candidates collection
            self.candidates_collection.create_index("candidate_id", unique=True)
            self.candidates_collection.create_index("leetcode_username")
            
            # Other collections
            for collection in [
                self.candidate_projects_collection,
                self.candidate_education_collection,
                self.candidate_skills_collection
            ]:
                collection.create_index("candidate_id")
                collection.create_index("created_at")
            
            logger.info("Candidate indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating candidate indexes: {e}")
    
    def store_candidate(
        self,
        candidate_id: str,
        leetcode_username: str,
        leetcode_stats: Optional[Dict[str, Any]],
        projects: List[Project],
        resume_text: str,
        education: List[Education],
        interview_questions: Dict[str, str],
        embeddings_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Store candidate profile across multiple collections with embeddings
        
        Args:
            leetcode_stats: LeetCode profile statistics
            embeddings_data: Dictionary containing embeddings and FAISS IDs for each section
                            Format: {"projects": {"embedding": [...], "faiss_id": 1}, ...}
        
        Returns:
            Dictionary with all collection IDs
        """
        try:
            timestamp = datetime.utcnow()
            
            # Store main candidate profile
            main_candidate_doc = {
                "candidate_id": candidate_id,
                "leetcode_username": leetcode_username,
                "leetcode_stats": leetcode_stats,
                "resume_text": resume_text,
                "interview_questions": interview_questions,
                "resume_embedding": embeddings_data.get('resume', {}).get('embedding'),
                "resume_faiss_id": embeddings_data.get('resume', {}).get('faiss_id'),
                "overall_embedding": embeddings_data.get('overall', {}).get('embedding'),
                "overall_faiss_id": embeddings_data.get('overall', {}).get('faiss_id'),
                "created_at": timestamp,
                "updated_at": timestamp
            }
            self.candidates_collection.insert_one(main_candidate_doc)
            logger.info(f"Stored main candidate profile: {candidate_id}")
            
            # Store projects with embeddings
            project_ids = []
            for idx, project in enumerate(projects):
                project_doc = {
                    "project_id": f"proj_{uuid.uuid4().hex[:8]}",
                    "candidate_id": candidate_id,
                    "name": project.name,
                    "description": project.description,
                    "github_link": str(project.github_link),
                    "technologies": project.technologies or [],
                    "embedding": embeddings_data.get(f'project_{idx}', {}).get('embedding'),
                    "faiss_index_id": embeddings_data.get(f'project_{idx}', {}).get('faiss_id'),
                    "created_at": timestamp
                }
                self.candidate_projects_collection.insert_one(project_doc)
                project_ids.append(project_doc['project_id'])
            logger.info(f"Stored {len(project_ids)} projects for candidate {candidate_id}")
            
            # Store education with embeddings
            education_ids = []
            for idx, edu in enumerate(education):
                edu_doc = {
                    "education_id": f"edu_{uuid.uuid4().hex[:8]}",
                    "candidate_id": candidate_id,
                    "degree": edu.degree,
                    "field_of_study": edu.field_of_study,
                    "institution": edu.institution,
                    "graduation_year": edu.graduation_year,
                    "gpa": edu.gpa,
                    "embedding": embeddings_data.get(f'education_{idx}', {}).get('embedding'),
                    "faiss_index_id": embeddings_data.get(f'education_{idx}', {}).get('faiss_id'),
                    "created_at": timestamp
                }
                self.candidate_education_collection.insert_one(edu_doc)
                education_ids.append(edu_doc['education_id'])
            logger.info(f"Stored {len(education_ids)} education entries for candidate {candidate_id}")
            
            # Extract and store skills from projects and resume
            all_skills = set()
            for project in projects:
                if project.technologies:
                    all_skills.update(project.technologies)
            
            if all_skills:
                skills_doc = {
                    "skills_id": f"skills_{uuid.uuid4().hex[:8]}",
                    "candidate_id": candidate_id,
                    "skills": list(all_skills),
                    "embedding": embeddings_data.get('skills', {}).get('embedding'),
                    "faiss_index_id": embeddings_data.get('skills', {}).get('faiss_id'),
                    "created_at": timestamp
                }
                self.candidate_skills_collection.insert_one(skills_doc)
                skills_id = skills_doc['skills_id']
            else:
                skills_id = None
            
            logger.info(f"Successfully stored candidate {candidate_id} across all collections")
            
            return {
                "candidate_id": candidate_id,
                "project_ids": project_ids,
                "education_ids": education_ids,
                "skills_id": skills_id
            }
            
        except Exception as e:
            logger.error(f"Error storing candidate profile: {e}")
            raise
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete candidate profile"""
        try:
            # Get main candidate
            candidate = self.candidates_collection.find_one({"candidate_id": candidate_id})
            if not candidate:
                return None
            
            # Get related data
            projects = list(self.candidate_projects_collection.find({"candidate_id": candidate_id}))
            education = list(self.candidate_education_collection.find({"candidate_id": candidate_id}))
            skills = self.candidate_skills_collection.find_one({"candidate_id": candidate_id})
            
            # Combine all data
            complete_candidate = {
                **candidate,
                "projects": projects,
                "education": education,
                "skills": skills
            }
            
            return complete_candidate
            
        except Exception as e:
            logger.error(f"Error retrieving candidate: {e}")
            raise
    
    def get_all_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all candidates"""
        try:
            candidates = list(self.candidates_collection.find().limit(limit))
            for candidate in candidates:
                candidate['_id'] = str(candidate['_id'])
            return candidates
        except Exception as e:
            logger.error(f"Error retrieving candidates: {e}")
            raise
    
    def get_candidate_stats(self) -> Dict[str, int]:
        """Get statistics for candidate collections"""
        try:
            return {
                "total_candidates": self.candidates_collection.count_documents({}),
                "total_projects": self.candidate_projects_collection.count_documents({}),
                "total_education_entries": self.candidate_education_collection.count_documents({}),
                "total_skills_entries": self.candidate_skills_collection.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting candidate stats: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Candidate MongoDB connection closed")
