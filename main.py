import logging
from typing import List, Dict, Any
import uuid

from config import config
from models import JobDescription, SearchQuery, SearchResult
from mongo_handler import MongoHandler
from faiss_handler import FAISSHandler, EmbeddingGenerator
from jd_parser import JDParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JDParserSystem:
    """Main system for parsing job descriptions and managing FAISS index"""
    
    def __init__(self):
        """Initialize all components"""
        try:
            # Validate configuration
            config.validate()
            
            # Initialize handlers
            self.mongo_handler = MongoHandler()
            self.faiss_handler = FAISSHandler()
            self.embedding_generator = EmbeddingGenerator()
            self.parser = JDParser()
            
            logger.info("JD Parser System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def add_job_description(
        self,
        title: str,
        description: str,
        company: str = None,
        location: str = None,
        job_id: str = None
    ) -> str:
        """
        Add a new job description to the system
        
        Args:
            title: Job title
            description: Full job description text
            company: Company name (optional)
            location: Job location (optional)
            job_id: Job ID (optional, will be generated if not provided)
            
        Returns:
            Job ID
        """
        try:
            # Generate job_id if not provided
            if not job_id:
                job_id = f"job_{uuid.uuid4().hex[:8]}"
            
            # Parse job description
            logger.info(f"Parsing job description: {job_id}")
            job = self.parser.create_job_object(
                job_id=job_id,
                title=title,
                description=description,
                company=company,
                location=location,
                use_llm=True
            )
            
            # Generate embedding for the job description
            logger.info(f"Generating embedding for job: {job_id}")
            embedding_text = f"{title} {description}"
            embedding = self.embedding_generator.generate_embedding(embedding_text)
            
            # Add to FAISS index
            logger.info(f"Adding to FAISS index: {job_id}")
            faiss_index_id = self.faiss_handler.add_embedding(embedding, job_id)
            
            # Update job with embedding and FAISS index ID
            job.embedding = embedding
            job.faiss_index_id = faiss_index_id
            
            # Store in MongoDB
            logger.info(f"Storing in MongoDB: {job_id}")
            self.mongo_handler.insert_job(job)
            
            # Save FAISS index to disk
            self.faiss_handler.save_index()
            
            logger.info(f"Successfully added job description: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error adding job description: {e}")
            raise
    
    def search_jobs(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar job descriptions
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Generate embedding for query
            logger.info(f"Searching for: {query}")
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search FAISS index
            results = self.faiss_handler.search(query_embedding, top_k)
            
            # Retrieve job details from MongoDB
            search_results = []
            for idx, distance in results:
                job_id = self.faiss_handler.get_job_id(idx)
                if job_id:
                    job_data = self.mongo_handler.get_job(job_id)
                    if job_data:
                        # Calculate similarity score (1 - normalized distance)
                        score = 1 / (1 + distance)
                        
                        search_result = SearchResult(
                            job_id=job_data['job_id'],
                            title=job_data['title'],
                            company=job_data.get('company'),
                            location=job_data.get('location'),
                            description=job_data['description'][:500],  # Truncate for display
                            score=score,
                            matched_skills=job_data.get('required_skills', [])
                        )
                        search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} matching jobs")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            raise
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get a specific job description
        
        Args:
            job_id: Job ID
            
        Returns:
            Job data dictionary
        """
        return self.mongo_handler.get_job(job_id)
    
    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all job descriptions
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job data dictionaries
        """
        return self.mongo_handler.get_all_jobs(limit)
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job description
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if deleted successfully
        """
        # Note: This doesn't remove from FAISS index
        # For production, you'd need to rebuild the index
        return self.mongo_handler.delete_job(job_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_jobs": self.mongo_handler.count_jobs(),
            "faiss_vectors": self.faiss_handler.get_total_vectors(),
            "mongodb_connected": True
        }
    
    def close(self):
        """Close all connections"""
        self.mongo_handler.close()
        logger.info("System closed")


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = JDParserSystem()
    
    # Example: Add a job description
    example_jd = """
    Senior Python Developer
    
    We are looking for an experienced Python developer to join our team.
    
    Requirements:
    - 5+ years of experience with Python
    - Strong knowledge of FastAPI, Django, or Flask
    - Experience with MongoDB and PostgreSQL
    - Familiarity with Docker and Kubernetes
    - Experience with AWS or GCP
    
    Responsibilities:
    - Design and develop scalable web applications
    - Write clean, maintainable code
    - Collaborate with cross-functional teams
    - Mentor junior developers
    
    Benefits:
    - Competitive salary
    - Remote work options
    - Health insurance
    - Professional development budget
    
    Location: Remote
    Type: Full-time
    """
    
    try:
        # Add the job
        job_id = system.add_job_description(
            title="Senior Python Developer",
            description=example_jd,
            company="Tech Corp",
            location="Remote"
        )
        print(f"Added job: {job_id}")
        
        # Search for similar jobs
        results = system.search_jobs("Python developer with FastAPI experience", top_k=3)
        print(f"\nSearch results:")
        for result in results:
            print(f"- {result.title} at {result.company} (Score: {result.score:.3f})")
        
        # Get statistics
        stats = system.get_stats()
        print(f"\nSystem stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.close()
