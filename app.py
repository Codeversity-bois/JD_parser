from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import uuid

from config import config
from api_schemas import JobDescriptionInput, ParsedJobResponse, HealthResponse
from collection_handler import CollectionHandler
from faiss_handler import FAISSHandler, EmbeddingGenerator
from jd_parser import JDParser
from candidate_models import CandidateInput, CandidateEvaluationResponse
from candidate_handler import CandidateHandler
from profile_matcher import ProfileMatcher
from leetcode_api import LeetCodeAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
collection_handler = None
faiss_handler = None
embedding_generator = None
parser = None
candidate_handler = None
profile_matcher = None
leetcode_api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global collection_handler, faiss_handler, embedding_generator, parser, candidate_handler, profile_matcher, leetcode_api
    
    # Startup
    logger.info("Starting JD Parser API...")
    try:
        config.validate()
        collection_handler = CollectionHandler()
        faiss_handler = FAISSHandler()
        embedding_generator = EmbeddingGenerator()
        parser = JDParser()
        candidate_handler = CandidateHandler()
        leetcode_api = LeetCodeAPI()
        profile_matcher = ProfileMatcher(
            faiss_handler=faiss_handler,
            embedding_generator=embedding_generator,
            collection_handler=collection_handler,
            candidate_handler=candidate_handler
        )
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down JD Parser API...")
    if collection_handler:
        collection_handler.close()
    if candidate_handler:
        candidate_handler.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Job Description Parser & Candidate Matching API",
    description="API for parsing job descriptions and matching candidates with FAISS indexing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Job Description Parser API",
        "version": "1.0.0",
        "endpoints": {
            "parse": "/parse - POST - Parse job description",
            "health": "/health - GET - Health check"
        }
    }


@app.post("/parse", response_model=ParsedJobResponse, tags=["Parser"])
async def parse_job_description(job_input: JobDescriptionInput):
    """
    Parse a job description and store it in separate collections
    
    This endpoint:
    1. Parses the job description using LLM
    2. Extracts structured data (skills, experience, responsibilities, etc.)
    3. Generates embeddings for semantic search
    4. Stores data in separate MongoDB collections:
       - skills (required & preferred)
       - experience (years & description)
       - education (requirements)
       - responsibilities (duties)
       - benefits (perks)
       - job_details (type, salary, work mode)
    5. Indexes embeddings in FAISS for similarity search
    
    Returns:
        Parsed job data with collection IDs for each field type
    """
    try:
        # Generate job ID
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Parse job description
        logger.info(f"Parsing job description: {job_id}")
        parsed_data = parser.parse_with_llm(job_input.job_description)
        
        # Extract title from parsed data if not provided
        title = job_input.job_title or parsed_data.get('title', 'Untitled Position')
        company = job_input.company or parsed_data.get('company')
        location = job_input.location or parsed_data.get('location')
        
        # Generate embedding for main job
        logger.info(f"Generating embeddings for job: {job_id}")
        embedding_text = f"{title} {job_input.job_description}"
        main_embedding = embedding_generator.generate_embedding(embedding_text)
        
        # Add main job to FAISS index
        logger.info(f"Adding main job to FAISS index: {job_id}")
        faiss_index_id = faiss_handler.add_embedding(main_embedding, f"{job_id}_main")
        
        # Generate embeddings for each field type and add to FAISS
        embeddings_data = {}
        
        # Skills embedding
        if parsed_data.get('required_skills') or parsed_data.get('preferred_skills'):
            skills_text = "Required Skills: " + ", ".join(parsed_data.get('required_skills', []))
            skills_text += " Preferred Skills: " + ", ".join(parsed_data.get('preferred_skills', []))
            skills_embedding = embedding_generator.generate_embedding(skills_text)
            skills_faiss_id = faiss_handler.add_embedding(skills_embedding, f"{job_id}_skills")
            embeddings_data['skills'] = {'embedding': skills_embedding, 'faiss_id': skills_faiss_id}
            logger.info(f"Added skills to FAISS: {skills_faiss_id}")
        
        # Experience embedding
        if parsed_data.get('experience_years'):
            exp_text = f"Experience Required: {parsed_data.get('experience_years')}"
            if parsed_data.get('experience_description'):
                exp_text += f" {parsed_data.get('experience_description')}"
            exp_embedding = embedding_generator.generate_embedding(exp_text)
            exp_faiss_id = faiss_handler.add_embedding(exp_embedding, f"{job_id}_experience")
            embeddings_data['experience'] = {'embedding': exp_embedding, 'faiss_id': exp_faiss_id}
            logger.info(f"Added experience to FAISS: {exp_faiss_id}")
        
        # Education embedding
        if parsed_data.get('education'):
            edu_text = f"Education Required: {parsed_data.get('education')}"
            edu_embedding = embedding_generator.generate_embedding(edu_text)
            edu_faiss_id = faiss_handler.add_embedding(edu_embedding, f"{job_id}_education")
            embeddings_data['education'] = {'embedding': edu_embedding, 'faiss_id': edu_faiss_id}
            logger.info(f"Added education to FAISS: {edu_faiss_id}")
        
        # Responsibilities embedding
        if parsed_data.get('responsibilities'):
            resp_text = "Responsibilities: " + ". ".join(parsed_data.get('responsibilities', []))
            resp_embedding = embedding_generator.generate_embedding(resp_text)
            resp_faiss_id = faiss_handler.add_embedding(resp_embedding, f"{job_id}_responsibilities")
            embeddings_data['responsibilities'] = {'embedding': resp_embedding, 'faiss_id': resp_faiss_id}
            logger.info(f"Added responsibilities to FAISS: {resp_faiss_id}")
        
        # Benefits embedding
        if parsed_data.get('benefits'):
            ben_text = "Benefits: " + ", ".join(parsed_data.get('benefits', []))
            ben_embedding = embedding_generator.generate_embedding(ben_text)
            ben_faiss_id = faiss_handler.add_embedding(ben_embedding, f"{job_id}_benefits")
            embeddings_data['benefits'] = {'embedding': ben_embedding, 'faiss_id': ben_faiss_id}
            logger.info(f"Added benefits to FAISS: {ben_faiss_id}")
        
        # Store in collections with embeddings
        logger.info(f"Storing in MongoDB collections: {job_id}")
        collection_ids = collection_handler.store_parsed_job(
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            description=job_input.job_description,
            parsed_data=parsed_data,
            embedding=main_embedding,
            faiss_index_id=faiss_index_id,
            embeddings_data=embeddings_data
        )
        
        # Save FAISS index
        faiss_handler.save_index()
        
        logger.info(f"Successfully processed job: {job_id}")
        
        return ParsedJobResponse(
            job_id=job_id,
            title=title,
            company=company,
            location=location,
            skills_id=collection_ids['skills_id'],
            experience_id=collection_ids['experience_id'],
            education_id=collection_ids['education_id'],
            responsibilities_id=collection_ids['responsibilities_id'],
            benefits_id=collection_ids['benefits_id'],
            job_details_id=collection_ids['job_details_id'],
            parsed_data=parsed_data
        )
        
    except Exception as e:
        logger.error(f"Error parsing job description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse job description: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns system status and statistics
    """
    try:
        stats = collection_handler.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            database=f"Connected - {stats['total_jobs']} jobs",
            faiss_vectors=faiss_handler.get_total_vectors()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get detailed statistics about all collections
    
    Returns:
        Statistics for each collection
    """
    try:
        stats = collection_handler.get_collection_stats()
        stats['faiss_vectors'] = faiss_handler.get_total_vectors()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job(job_id: str):
    """
    Get complete job data by job_id
    
    Retrieves data from all collections
    """
    try:
        job_data = collection_handler.get_job_by_id(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Convert ObjectId to string
        job_data['_id'] = str(job_data['_id'])
        
        return job_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job"
        )


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(limit: int = 50):
    """
    List all jobs
    
    Args:
        limit: Maximum number of jobs to return (default: 50)
    """
    try:
        jobs = collection_handler.get_all_jobs(limit=limit)
        return {
            "total": len(jobs),
            "jobs": jobs
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs"
        )


@app.post("/candidate/submit", tags=["Candidates"])
async def submit_candidate_profile(candidate_input: CandidateInput):
    """
    Submit a candidate profile for evaluation
    
    This endpoint:
    1. Generates embeddings for all candidate data sections
    2. Stores candidate profile in MongoDB with embeddings
    3. Adds embeddings to FAISS index
    
    The candidate can then be evaluated against job descriptions.
    """
    try:
        # Generate candidate ID
        candidate_id = f"candidate_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Processing candidate profile: {candidate_id}")
        
        # Fetch LeetCode profile data (for LLM evaluation only, not for embedding)
        logger.info(f"Fetching LeetCode profile for: {candidate_input.leetcode_username}")
        leetcode_stats = leetcode_api.get_comprehensive_profile(candidate_input.leetcode_username)
        
        if not leetcode_stats.get("exists"):
            logger.warning(f"LeetCode profile not found for {candidate_input.leetcode_username}")
            # Continue with submission but mark LeetCode as unavailable
        
        # Generate embeddings for different sections (excluding LeetCode - only for LLM use)
        embeddings_data = {}
        
        # Overall profile embedding (without LeetCode stats)
        overall_text = f"LeetCode Username: {candidate_input.leetcode_username}\n"
        overall_text += f"Resume: {candidate_input.resume_text}\n"
        overall_text += f"Projects: {len(candidate_input.projects)}\n"
        
        overall_embedding = embedding_generator.generate_embedding(overall_text)
        overall_faiss_id = faiss_handler.add_embedding(overall_embedding, f"{candidate_id}_overall")
        embeddings_data['overall'] = {'embedding': overall_embedding, 'faiss_id': overall_faiss_id}
        logger.info(f"Generated overall embedding: FAISS ID {overall_faiss_id}")
        
        # Resume embedding
        resume_embedding = embedding_generator.generate_embedding(candidate_input.resume_text)
        resume_faiss_id = faiss_handler.add_embedding(resume_embedding, f"{candidate_id}_resume")
        embeddings_data['resume'] = {'embedding': resume_embedding, 'faiss_id': resume_faiss_id}
        
        # Projects embeddings
        for idx, project in enumerate(candidate_input.projects):
            project_text = f"{project.name}: {project.description} Technologies: {', '.join(project.technologies or [])}"
            project_embedding = embedding_generator.generate_embedding(project_text)
            project_faiss_id = faiss_handler.add_embedding(project_embedding, f"{candidate_id}_project_{idx}")
            embeddings_data[f'project_{idx}'] = {'embedding': project_embedding, 'faiss_id': project_faiss_id}
        logger.info(f"Generated {len(candidate_input.projects)} project embeddings")
        
        # Education embeddings
        for idx, edu in enumerate(candidate_input.education):
            edu_text = f"{edu.degree} in {edu.field_of_study} from {edu.institution}"
            edu_embedding = embedding_generator.generate_embedding(edu_text)
            edu_faiss_id = faiss_handler.add_embedding(edu_embedding, f"{candidate_id}_education_{idx}")
            embeddings_data[f'education_{idx}'] = {'embedding': edu_embedding, 'faiss_id': edu_faiss_id}
        logger.info(f"Generated {len(candidate_input.education)} education embeddings")
        
        # Skills embedding (from projects)
        all_skills = set()
        for project in candidate_input.projects:
            if project.technologies:
                all_skills.update(project.technologies)
        
        if all_skills:
            skills_text = "Skills: " + ", ".join(all_skills)
            skills_embedding = embedding_generator.generate_embedding(skills_text)
            skills_faiss_id = faiss_handler.add_embedding(skills_embedding, f"{candidate_id}_skills")
            embeddings_data['skills'] = {'embedding': skills_embedding, 'faiss_id': skills_faiss_id}
            logger.info(f"Generated skills embedding with {len(all_skills)} skills")
        
        # Store in MongoDB
        logger.info(f"Storing candidate profile in MongoDB: {candidate_id}")
        collection_ids = candidate_handler.store_candidate(
            candidate_id=candidate_id,
            leetcode_username=candidate_input.leetcode_username,
            leetcode_stats=leetcode_stats,
            projects=candidate_input.projects,
            resume_text=candidate_input.resume_text,
            education=candidate_input.education,
            interview_questions=candidate_input.interview_questions or {},
            embeddings_data=embeddings_data
        )
        
        # Save FAISS index
        faiss_handler.save_index()
        
        logger.info(f"Successfully stored candidate profile: {candidate_id}")
        
        # Prepare response with LeetCode info
        leetcode_info = {}
        if leetcode_stats.get("exists"):
            solved = leetcode_stats.get("solved_stats", {})
            contest = leetcode_stats.get("contest_info", {})
            leetcode_info = {
                "profile_found": True,
                "total_solved": solved.get("solvedProblem", 0),
                "contest_rating": contest.get("contestRating", 0) if contest else 0
            }
        else:
            leetcode_info = {
                "profile_found": False,
                "message": "LeetCode profile not found or unavailable"
            }
        
        return {
            "candidate_id": candidate_id,
            "leetcode_username": candidate_input.leetcode_username,
            "leetcode_info": leetcode_info,
            "message": "Candidate profile submitted successfully",
            "collection_ids": collection_ids,
            "embeddings_generated": len(embeddings_data),
            "next_step": "Candidates will be evaluated when a job evaluation is triggered"
        }
        
    except Exception as e:
        logger.error(f"Error submitting candidate profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit candidate profile: {str(e)}"
        )


@app.post("/candidate/evaluate/{candidate_id}", tags=["Candidates"])
async def evaluate_candidate_deprecated(candidate_id: str):
    """
    [DEPRECATED] Use /job/evaluate/{job_id} instead
    
    This endpoint evaluated one candidate against all jobs.
    New approach: Evaluate all candidates for one job.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint is deprecated. Use POST /job/evaluate/{job_id} to evaluate all candidates for a specific job."
    )


@app.post("/job/evaluate/{job_id}", tags=["Jobs"])
async def evaluate_job_candidates(job_id: str):
    """
    Evaluate ALL candidates for a specific job position
    
    This endpoint:
    1. Retrieves job description and embeddings
    2. Matches ALL candidates with this job using FAISS
    3. Calculates similarity scores for each candidate
    4. **Eliminates bottom 60% of candidates** (keeps top 40%)
    5. **Evaluates remaining 40% using LLM** (OpenRouter)
    6. Returns final list of recommended candidates for OA round
    
    Returns:
        Complete evaluation with candidate recommendations
    """
    try:
        logger.info(f"Evaluating candidates for job: {job_id}")
        
        # Get job data
        job_data = collection_handler.get_job_by_id(job_id)
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Prepare embeddings dictionary
        job_embeddings = {
            'overall': job_data.get('embedding')
        }
        
        # Get field-specific embeddings if available
        # Skills
        skills_data = collection_handler.skills_collection.find_one({"job_id": job_id})
        if skills_data and skills_data.get('embedding'):
            job_embeddings['skills'] = skills_data['embedding']
        
        # Experience
        exp_data = collection_handler.experience_collection.find_one({"job_id": job_id})
        if exp_data and exp_data.get('embedding'):
            job_embeddings['experience'] = exp_data['embedding']
        
        if not job_embeddings.get('overall'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} has no embeddings. Please re-parse the job description."
            )
        
        # Process all candidates for this job
        evaluation_result = profile_matcher.process_job_candidates(
            job_id=job_id,
            job_embeddings=job_embeddings
        )
        
        logger.info(f"Job {job_id} evaluation complete")
        return evaluation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating job candidates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate candidates: {str(e)}"
        )


@app.get("/candidate/{candidate_id}", tags=["Candidates"])
async def get_candidate_profile(candidate_id: str):
    """
    Get complete candidate profile
    
    Args:
        candidate_id: Candidate identifier
        
    Returns:
        Complete candidate profile with all data
    """
    try:
        candidate = candidate_handler.get_candidate(candidate_id)
        
        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate {candidate_id} not found"
            )
        
        # Convert ObjectId to string in main document
        candidate['_id'] = str(candidate['_id'])
        
        # Convert ObjectId in nested documents (projects, education, skills)
        if candidate.get('projects'):
            for project in candidate['projects']:
                if '_id' in project:
                    project['_id'] = str(project['_id'])
                # Remove embedding from projects
                project.pop('embedding', None)
        
        if candidate.get('education'):
            for edu in candidate['education']:
                if '_id' in edu:
                    edu['_id'] = str(edu['_id'])
                # Remove embedding from education
                edu.pop('embedding', None)
        
        if candidate.get('skills') and isinstance(candidate['skills'], dict):
            if '_id' in candidate['skills']:
                candidate['skills']['_id'] = str(candidate['skills']['_id'])
            # Remove embedding from skills
            candidate['skills'].pop('embedding', None)
        
        # Remove large embedding arrays for response
        candidate.pop('overall_embedding', None)
        candidate.pop('resume_embedding', None)
        
        return candidate
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving candidate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve candidate profile"
        )


@app.get("/candidates", tags=["Candidates"])
async def list_candidates(limit: int = 50):
    """
    List all candidates
    
    Args:
        limit: Maximum number of candidates to return (default: 50)
    """
    try:
        candidates = candidate_handler.get_all_candidates(limit=limit)
        return {
            "total": len(candidates),
            "candidates": candidates
        }
    except Exception as e:
        logger.error(f"Error listing candidates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list candidates"
        )


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
