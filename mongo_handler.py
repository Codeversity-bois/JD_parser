from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
import logging

from config import config
from models import JobDescription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoHandler:
    """Handler for MongoDB Atlas operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(config.MONGODB_URL)
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
            self.db = self.client[config.MONGODB_DB_NAME]
            self.collection = self.db[config.COLLECTION_NAME]
            
            # Create indexes
            self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary indexes for the collection"""
        try:
            # Create unique index on job_id
            self.collection.create_index("job_id", unique=True)
            # Create index on title for text search
            self.collection.create_index("title")
            # Create index on faiss_index_id
            self.collection.create_index("faiss_index_id")
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def insert_job(self, job: JobDescription) -> str:
        """
        Insert a job description into MongoDB
        
        Args:
            job: JobDescription object
            
        Returns:
            Inserted document ID
        """
        try:
            job_dict = job.model_dump()
            result = self.collection.insert_one(job_dict)
            logger.info(f"Inserted job with ID: {job.job_id}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            logger.warning(f"Job with ID {job.job_id} already exists")
            return self.update_job(job.job_id, job)
        except Exception as e:
            logger.error(f"Error inserting job: {e}")
            raise
    
    def update_job(self, job_id: str, job: JobDescription) -> str:
        """
        Update an existing job description
        
        Args:
            job_id: Job ID to update
            job: Updated JobDescription object
            
        Returns:
            Updated document ID
        """
        try:
            job_dict = job.model_dump()
            job_dict['updated_at'] = datetime.utcnow()
            
            result = self.collection.update_one(
                {"job_id": job_id},
                {"$set": job_dict}
            )
            
            if result.matched_count > 0:
                logger.info(f"Updated job with ID: {job_id}")
                return job_id
            else:
                logger.warning(f"Job with ID {job_id} not found for update")
                return None
        except Exception as e:
            logger.error(f"Error updating job: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a job description by job_id
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Job document or None
        """
        try:
            job = self.collection.find_one({"job_id": job_id})
            if job:
                job['_id'] = str(job['_id'])  # Convert ObjectId to string
            return job
        except Exception as e:
            logger.error(f"Error retrieving job: {e}")
            raise
    
    def get_job_by_faiss_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a job description by FAISS index ID
        
        Args:
            faiss_id: FAISS index ID
            
        Returns:
            Job document or None
        """
        try:
            job = self.collection.find_one({"faiss_index_id": faiss_id})
            if job:
                job['_id'] = str(job['_id'])
            return job
        except Exception as e:
            logger.error(f"Error retrieving job by FAISS ID: {e}")
            raise
    
    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all job descriptions
        
        Args:
            limit: Maximum number of jobs to retrieve
            
        Returns:
            List of job documents
        """
        try:
            jobs = list(self.collection.find().limit(limit))
            for job in jobs:
                job['_id'] = str(job['_id'])
            return jobs
        except Exception as e:
            logger.error(f"Error retrieving all jobs: {e}")
            raise
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job description
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            result = self.collection.delete_one({"job_id": job_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted job with ID: {job_id}")
                return True
            else:
                logger.warning(f"Job with ID {job_id} not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Error deleting job: {e}")
            raise
    
    def count_jobs(self) -> int:
        """
        Count total number of jobs in the collection
        
        Returns:
            Total count of jobs
        """
        try:
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error counting jobs: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
