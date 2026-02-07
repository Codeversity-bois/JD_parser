from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import json
import requests

from config import config
from candidate_models import (
    CandidateInput, MatchResult, ShortlistResult, 
    FinalEvaluationResult, CandidateEvaluationResponse
)
from faiss_handler import EmbeddingGenerator, FAISSHandler
from collection_handler import CollectionHandler
from candidate_handler import CandidateHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfileMatcher:
    """System for matching candidate profiles with job descriptions"""
    
    def __init__(
        self,
        faiss_handler: FAISSHandler,
        embedding_generator: EmbeddingGenerator,
        collection_handler: CollectionHandler,
        candidate_handler: CandidateHandler
    ):
        """Initialize profile matcher"""
        self.faiss_handler = faiss_handler
        self.embedding_generator = embedding_generator
        self.collection_handler = collection_handler
        self.candidate_handler = candidate_handler
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
    
    def calculate_similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Convert to 0-1 range
        return float((similarity + 1) / 2)
    
    def match_job_with_candidates(
        self,
        job_embeddings: Dict[str, List[float]],
        job_data: Dict[str, Any],
        top_k: int = 1000
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Match a job with all candidate profiles using FAISS
        
        Args:
            job_embeddings: Dictionary of job embeddings
            job_data: Job information
            top_k: Number of top candidate matches to return
            
        Returns:
            List of tuples (candidate_id, similarity_score, candidate_data)
        """
        try:
            # Use overall job embedding for initial matching
            overall_embedding = job_embeddings.get('overall')
            if not overall_embedding:
                logger.error("No overall embedding found for job")
                return []
            
            # Search FAISS index for similar candidates
            results = self.faiss_handler.search(overall_embedding, top_k=top_k)
            
            candidate_matches = []
            for faiss_idx, distance in results:
                entry_id = self.faiss_handler.get_job_id(faiss_idx)
                
                # Filter to get only candidate entries (not job entries)
                if entry_id and 'candidate_' in entry_id and '_overall' in entry_id:
                    candidate_id = entry_id.split('_overall')[0]
                    candidate_data = self.candidate_handler.get_candidate(candidate_id)
                    
                    if candidate_data:
                        # Calculate detailed similarity score
                        candidate_embedding = candidate_data.get('overall_embedding')
                        if candidate_embedding:
                            similarity = self.calculate_similarity_score(overall_embedding, candidate_embedding)
                        else:
                            similarity = 1 / (1 + distance)  # Fallback
                        
                        candidate_matches.append((candidate_id, similarity, candidate_data))
            
            # Sort by similarity score (highest first)
            candidate_matches.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(candidate_matches)} candidate matches for job {job_data.get('job_id')}")
            return candidate_matches
            
        except Exception as e:
            logger.error(f"Error matching job with candidates: {e}")
            raise
    
    def apply_60_percent_filter(self, candidate_matches: List[Tuple[str, float, Dict[str, Any]]]) -> Tuple[List[Tuple[str, float, Dict[str, Any]]], int]:
        """
        Filter out bottom 60% of candidates, keep top 40%
        
        Args:
            candidate_matches: List of (candidate_id, similarity_score, candidate_data) tuples
            
        Returns:
            Tuple of (filtered candidates, eliminated count)
        """
        if not candidate_matches:
            return [], 0
        
        # Calculate cutoff for top 40%
        total_candidates = len(candidate_matches)
        top_40_percent_count = max(1, int(total_candidates * 0.4))
        
        # Keep top 40%
        shortlisted = candidate_matches[:top_40_percent_count]
        eliminated = total_candidates - top_40_percent_count
        
        logger.info(f"Filtered: Kept {len(shortlisted)}, Eliminated {eliminated} (60%)")
        return shortlisted, eliminated
    
    def evaluate_with_llm(
        self,
        candidate_data: Dict[str, Any],
        job_data: Dict[str, Any],
        similarity_score: float
    ) -> FinalEvaluationResult:
        """
        Evaluate candidate-job match using LLM
        
        Args:
            candidate_data: Candidate profile data
            job_data: Job description data
            similarity_score: Pre-calculated similarity score
            
        Returns:
            Final evaluation result
        """
        try:
            # Prepare prompt for LLM
            prompt = f"""
You are an expert technical recruiter. Evaluate the following candidate for the job position.

JOB DETAILS:
- Title: {job_data.get('title')}
- Company: {job_data.get('company', 'N/A')}
- Description: {job_data.get('description', '')[:500]}

CANDIDATE PROFILE:
- LeetCode Username: {candidate_data.get('leetcode_username')}
- Resume Summary: {candidate_data.get('resume_text', '')[:500]}
- Projects: {len(candidate_data.get('projects', []))} projects
- Education: {len(candidate_data.get('education', []))} degrees

Initial Similarity Score: {similarity_score:.2f}

TASK:
Evaluate this candidate for the job. Provide:
1. A final score (0-100)
2. A recommendation (Highly Recommended / Recommended / Consider / Not Recommended)
3. Brief reasoning (2-3 sentences)
4. Whether they should proceed to OA round (true/false)

Return ONLY a valid JSON object with these fields:
{{
    "final_score": <number 0-100>,
    "recommendation": "<string>",
    "reasoning": "<string>",
    "proceed_to_oa": <boolean>,
    "strengths": ["<strength1>", "<strength2>"],
    "concerns": ["<concern1>", "<concern2>"]
}}
"""
            
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert technical recruiter. Provide objective, fair evaluations. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            evaluation = json.loads(content)
            
            # Create final evaluation result
            final_result = FinalEvaluationResult(
                candidate_id=candidate_data.get('candidate_id'),
                job_id=job_data.get('job_id'),
                job_title=job_data.get('title'),
                similarity_score=similarity_score,
                llm_evaluation=evaluation,
                final_score=evaluation.get('final_score', 0) / 100.0,  # Normalize to 0-1
                recommendation=evaluation.get('recommendation', 'Not Recommended'),
                reasoning=evaluation.get('reasoning', ''),
                proceed_to_oa=evaluation.get('proceed_to_oa', False)
            )
            
            logger.info(f"LLM evaluation complete: {final_result.recommendation}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            # Return default evaluation on error
            return FinalEvaluationResult(
                candidate_id=candidate_data.get('candidate_id'),
                job_id=job_data.get('job_id'),
                job_title=job_data.get('title', 'Unknown'),
                similarity_score=similarity_score,
                llm_evaluation={"error": str(e)},
                final_score=similarity_score,
                recommendation="Error in evaluation",
                reasoning="Could not complete LLM evaluation",
                proceed_to_oa=False
            )
    
    def process_job_candidates(
        self,
        job_id: str,
        job_embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Complete candidate evaluation pipeline for a specific job
        
        Args:
            job_id: Job identifier
            job_embeddings: Job embeddings dictionary
            
        Returns:
            Complete evaluation response with candidate recommendations
        """
        try:
            # Get job data
            job_data = self.collection_handler.get_job_by_id(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            # Step 1: Match job with all candidates
            logger.info(f"Step 1: Matching job {job_id} with all candidates")
            all_candidate_matches = self.match_job_with_candidates(
                job_embeddings=job_embeddings,
                job_data=job_data,
                top_k=1000
            )
            
            if not all_candidate_matches:
                return {
                    "job_id": job_id,
                    "job_title": job_data.get('title'),
                    "total_candidates_analyzed": 0,
                    "initial_matches": 0,
                    "after_60_percent_filter": 0,
                    "final_recommendations": [],
                    "message": "No candidates found in database"
                }
            
            # Step 2: Apply 60% filter (eliminate bottom 60%, keep top 40%)
            logger.info(f"Step 2: Applying 60% elimination filter")
            shortlisted_candidates, eliminated_count = self.apply_60_percent_filter(all_candidate_matches)
            
            # Step 3: Evaluate remaining 40% with LLM
            logger.info(f"Step 3: Evaluating {len(shortlisted_candidates)} candidates with LLM")
            final_evaluations = []
            
            for candidate_id, similarity_score, candidate_data in shortlisted_candidates:
                evaluation = self.evaluate_with_llm(
                    candidate_data=candidate_data,
                    job_data=job_data,
                    similarity_score=similarity_score
                )
                final_evaluations.append(evaluation)
            
            # Filter only those proceeding to OA
            oa_candidates = [e for e in final_evaluations if e.proceed_to_oa]
            
            # Sort by final score
            oa_candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            response = {
                "job_id": job_id,
                "job_title": job_data.get('title'),
                "company": job_data.get('company'),
                "total_candidates_analyzed": len(all_candidate_matches),
                "initial_matches": len(all_candidate_matches),
                "eliminated_count": eliminated_count,
                "after_60_percent_filter": len(shortlisted_candidates),
                "final_recommendations": oa_candidates,
                "message": f"Evaluation complete. {len(oa_candidates)} candidates recommended for OA round."
            }
            
            logger.info(f"Job {job_id} evaluation complete: {len(oa_candidates)} candidates for OA")
            return response
            
        except Exception as e:
            logger.error(f"Error processing job candidates: {e}")
            raise
