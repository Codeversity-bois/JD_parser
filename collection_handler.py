from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from datetime import datetime
import logging
import uuid

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollectionHandler:
    """Handler for managing separate collections for different job description fields"""
    
    def __init__(self):
        """Initialize MongoDB connection and collections"""
        try:
            self.client = MongoClient(config.MONGODB_URL)
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
            self.db = self.client[config.MONGODB_DB_NAME]
            
            # Define separate collections
            self.skills_collection = self.db['skills']
            self.experience_collection = self.db['experience']
            self.education_collection = self.db['education']
            self.responsibilities_collection = self.db['responsibilities']
            self.benefits_collection = self.db['benefits']
            self.job_details_collection = self.db['job_details']
            self.main_jobs_collection = self.db['jobs']
            
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for all collections"""
        try:
            # Main jobs collection
            self.main_jobs_collection.create_index("job_id", unique=True)
            
            # Other collections
            for collection in [
                self.skills_collection,
                self.experience_collection,
                self.education_collection,
                self.responsibilities_collection,
                self.benefits_collection,
                self.job_details_collection
            ]:
                collection.create_index("job_id")
                collection.create_index("created_at")
            
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_parsed_job(
        self,
        job_id: str,
        title: str,
        company: Optional[str],
        location: Optional[str],
        description: str,
        parsed_data: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        faiss_index_id: Optional[int] = None,
        embeddings_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Store parsed job data across multiple collections with individual embeddings
        
        Args:
            embeddings_data: Dictionary containing embeddings and FAISS IDs for each field type
                            Format: {"skills": {"embedding": [...], "faiss_id": 1}, ...}
        
        Returns:
            Dictionary with all collection IDs
        """
        try:
            timestamp = datetime.utcnow()
            
            # Store in main jobs collection
            main_job_doc = {
                "job_id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "embedding": embedding,
                "faiss_index_id": faiss_index_id,
                "created_at": timestamp,
                "updated_at": timestamp
            }
            self.main_jobs_collection.insert_one(main_job_doc)
            
            # Store skills with embedding
            skills_id = None
            if parsed_data.get('required_skills') or parsed_data.get('preferred_skills'):
                skills_doc = {
                    "skills_id": f"skills_{uuid.uuid4().hex[:8]}",
                    "job_id": job_id,
                    "required_skills": parsed_data.get('required_skills', []),
                    "preferred_skills": parsed_data.get('preferred_skills', []),
                    "embedding": embeddings_data.get('skills', {}).get('embedding') if embeddings_data else None,
                    "faiss_index_id": embeddings_data.get('skills', {}).get('faiss_id') if embeddings_data else None,
                    "created_at": timestamp
                }
                self.skills_collection.insert_one(skills_doc)
                skills_id = skills_doc['skills_id']
            
            # Store experience with embedding
            experience_id = None
            if parsed_data.get('experience_years'):
                exp_doc = {
                    "experience_id": f"exp_{uuid.uuid4().hex[:8]}",
                    "job_id": job_id,
                    "experience_years": parsed_data.get('experience_years'),
                    "experience_description": parsed_data.get('experience_description', ''),
                    "embedding": embeddings_data.get('experience', {}).get('embedding') if embeddings_data else None,
                    "faiss_index_id": embeddings_data.get('experience', {}).get('faiss_id') if embeddings_data else None,
                    "created_at": timestamp
                }
                self.experience_collection.insert_one(exp_doc)
                experience_id = exp_doc['experience_id']
            
            # Store education with embedding
            education_id = None
            if parsed_data.get('education'):
                edu_doc = {
                    "education_id": f"edu_{uuid.uuid4().hex[:8]}",
                    "job_id": job_id,
                    "education": parsed_data.get('education'),
                    "embedding": embeddings_data.get('education', {}).get('embedding') if embeddings_data else None,
                    "faiss_index_id": embeddings_data.get('education', {}).get('faiss_id') if embeddings_data else None,
                    "created_at": timestamp
                }
                self.education_collection.insert_one(edu_doc)
                education_id = edu_doc['education_id']
            
            # Store responsibilities with embedding
            responsibilities_id = None
            if parsed_data.get('responsibilities'):
                resp_doc = {
                    "responsibilities_id": f"resp_{uuid.uuid4().hex[:8]}",
                    "job_id": job_id,
                    "responsibilities": parsed_data.get('responsibilities', []),
                    "embedding": embeddings_data.get('responsibilities', {}).get('embedding') if embeddings_data else None,
                    "faiss_index_id": embeddings_data.get('responsibilities', {}).get('faiss_id') if embeddings_data else None,
                    "created_at": timestamp
                }
                self.responsibilities_collection.insert_one(resp_doc)
                responsibilities_id = resp_doc['responsibilities_id']
            
            # Store benefits with embedding
            benefits_id = None
            if parsed_data.get('benefits'):
                ben_doc = {
                    "benefits_id": f"ben_{uuid.uuid4().hex[:8]}",
                    "job_id": job_id,
                    "benefits": parsed_data.get('benefits', []),
                    "embedding": embeddings_data.get('benefits', {}).get('embedding') if embeddings_data else None,
                    "faiss_index_id": embeddings_data.get('benefits', {}).get('faiss_id') if embeddings_data else None,
                    "created_at": timestamp
                }
                self.benefits_collection.insert_one(ben_doc)
                benefits_id = ben_doc['benefits_id']
            
            # Store job details (type, salary, etc.)
            details_doc = {
                "details_id": f"details_{uuid.uuid4().hex[:8]}",
                "job_id": job_id,
                "job_type": parsed_data.get('job_type'),
                "salary_range": parsed_data.get('salary_range'),
                "work_mode": parsed_data.get('work_mode'),
                "created_at": timestamp
            }
            self.job_details_collection.insert_one(details_doc)
            details_id = details_doc['details_id']
            
            logger.info(f"Successfully stored job {job_id} across all collections")
            
            return {
                "skills_id": skills_id or "N/A",
                "experience_id": experience_id or "N/A",
                "education_id": education_id or "N/A",
                "responsibilities_id": responsibilities_id or "N/A",
                "benefits_id": benefits_id or "N/A",
                "job_details_id": details_id
            }
            
        except Exception as e:
            logger.error(f"Error storing parsed job: {e}")
            raise
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete job data from all collections"""
        try:
            # Get main job
            main_job = self.main_jobs_collection.find_one({"job_id": job_id})
            if not main_job:
                return None
            
            # Get related data
            skills = self.skills_collection.find_one({"job_id": job_id})
            experience = self.experience_collection.find_one({"job_id": job_id})
            education = self.education_collection.find_one({"job_id": job_id})
            responsibilities = self.responsibilities_collection.find_one({"job_id": job_id})
            benefits = self.benefits_collection.find_one({"job_id": job_id})
            details = self.job_details_collection.find_one({"job_id": job_id})
            
            # Combine all data
            complete_job = {
                **main_job,
                "skills": skills,
                "experience": experience,
                "education": education,
                "responsibilities": responsibilities,
                "benefits": benefits,
                "details": details
            }
            
            return complete_job
            
        except Exception as e:
            logger.error(f"Error retrieving job: {e}")
            raise
    
    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all jobs from main collection"""
        try:
            jobs = list(self.main_jobs_collection.find().limit(limit))
            for job in jobs:
                job['_id'] = str(job['_id'])
            return jobs
        except Exception as e:
            logger.error(f"Error retrieving jobs: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections"""
        try:
            return {
                "total_jobs": self.main_jobs_collection.count_documents({}),
                "skills_entries": self.skills_collection.count_documents({}),
                "experience_entries": self.experience_collection.count_documents({}),
                "education_entries": self.education_collection.count_documents({}),
                "responsibilities_entries": self.responsibilities_collection.count_documents({}),
                "benefits_entries": self.benefits_collection.count_documents({}),
                "job_details_entries": self.job_details_collection.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
