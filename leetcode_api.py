import requests
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeetCodeAPI:
    """Client for Alfa-LeetCode-API"""
    
    BASE_URL = "https://alfa-leetcode-api.onrender.com"
    
    def __init__(self):
        """Initialize LeetCode API client"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "JD-Parser-Candidate-System/1.0"
        })
    
    def get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive user profile from LeetCode
        
        Args:
            username: LeetCode username
            
        Returns:
            User profile data or None if user not found
        """
        try:
            url = f"{self.BASE_URL}/{username}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched LeetCode profile for {username}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"LeetCode user {username} not found")
                return None
            logger.error(f"HTTP error fetching LeetCode profile: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching LeetCode profile for {username}: {e}")
            return None
    
    def get_solved_stats(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get solved questions statistics
        
        Args:
            username: LeetCode username
            
        Returns:
            Solved questions stats
        """
        try:
            url = f"{self.BASE_URL}/{username}/solved"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched solved stats for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching solved stats: {e}")
            return None
    
    def get_badges(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user badges
        
        Args:
            username: LeetCode username
            
        Returns:
            User badges data
        """
        try:
            url = f"{self.BASE_URL}/{username}/badges"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched badges for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching badges: {e}")
            return None
    
    def get_contest_info(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get contest details
        
        Args:
            username: LeetCode username
            
        Returns:
            Contest information
        """
        try:
            url = f"{self.BASE_URL}/{username}/contest"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched contest info for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching contest info: {e}")
            return None
    
    def get_recent_submissions(self, username: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Get recent submissions
        
        Args:
            username: LeetCode username
            limit: Number of submissions to fetch
            
        Returns:
            Recent submissions data
        """
        try:
            url = f"{self.BASE_URL}/{username}/submission?limit={limit}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {limit} submissions for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching submissions: {e}")
            return None
    
    def get_skills(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get skill statistics
        
        Args:
            username: LeetCode username
            
        Returns:
            Skill stats data
        """
        try:
            url = f"{self.BASE_URL}/{username}/skill"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched skills for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching skills: {e}")
            return None
    
    def get_language_stats(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get programming language statistics
        
        Args:
            username: LeetCode username
            
        Returns:
            Language stats data
        """
        try:
            url = f"{self.BASE_URL}/{username}/language"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched language stats for {username}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching language stats: {e}")
            return None
    
    def get_comprehensive_profile(self, username: str) -> Dict[str, Any]:
        """
        Get comprehensive LeetCode profile with all available data
        
        Args:
            username: LeetCode username
            
        Returns:
            Complete profile data
        """
        try:
            logger.info(f"Fetching comprehensive LeetCode profile for {username}")
            
            # Fetch all available data
            profile = {
                "username": username,
                "profile_data": self.get_user_profile(username),
                "solved_stats": self.get_solved_stats(username),
                "badges": self.get_badges(username),
                "contest_info": self.get_contest_info(username),
                "recent_submissions": self.get_recent_submissions(username, limit=10),
                "skills": self.get_skills(username),
                "language_stats": self.get_language_stats(username)
            }
            
            # Check if user exists
            if not profile["profile_data"]:
                logger.warning(f"LeetCode user {username} not found")
                return {
                    "username": username,
                    "exists": False,
                    "error": "User not found"
                }
            
            profile["exists"] = True
            logger.info(f"Successfully fetched comprehensive profile for {username}")
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive profile: {e}")
            return {
                "username": username,
                "exists": False,
                "error": str(e)
            }
    
    def generate_profile_summary(self, profile_data: Dict[str, Any]) -> str:
        """
        Generate a text summary of LeetCode profile for embedding generation
        
        Args:
            profile_data: Comprehensive profile data
            
        Returns:
            Text summary
        """
        if not profile_data.get("exists"):
            return f"LeetCode username: {profile_data.get('username')} (Profile not found)"
        
        summary_parts = [f"LeetCode Profile: {profile_data.get('username')}"]
        
        # Add solved stats
        solved = profile_data.get("solved_stats", {})
        if solved:
            total = solved.get("solvedProblem", 0)
            easy = solved.get("easySolved", 0)
            medium = solved.get("mediumSolved", 0)
            hard = solved.get("hardSolved", 0)
            summary_parts.append(f"Solved {total} problems: {easy} easy, {medium} medium, {hard} hard")
        
        # Add contest info
        contest = profile_data.get("contest_info", {})
        if contest and contest.get("contestAttend"):
            rating = contest.get("contestRating", 0)
            attended = contest.get("contestAttend", 0)
            ranking = contest.get("contestGlobalRanking", "N/A")
            summary_parts.append(f"Contest Rating: {rating}, Attended: {attended} contests, Global Ranking: {ranking}")
        
        # Add badges
        badges = profile_data.get("badges", {})
        if badges and isinstance(badges, dict):
            badge_count = badges.get("badgesCount", 0)
            if badge_count > 0:
                summary_parts.append(f"Earned {badge_count} badges")
        
        # Add skills
        skills = profile_data.get("skills", {})
        if skills and isinstance(skills, dict):
            skill_tags = skills.get("skills", [])
            if skill_tags:
                top_skills = [s.get("tagName") for s in skill_tags[:5] if s.get("tagName")]
                if top_skills:
                    summary_parts.append(f"Top skills: {', '.join(top_skills)}")
        
        # Add language stats
        languages = profile_data.get("language_stats", {})
        if languages and isinstance(languages, dict):
            lang_data = languages.get("languageProblemCount", [])
            if lang_data:
                top_langs = [f"{l.get('languageName')} ({l.get('problemsSolved')})" 
                           for l in lang_data[:3] if l.get("languageName")]
                if top_langs:
                    summary_parts.append(f"Languages: {', '.join(top_langs)}")
        
        return ". ".join(summary_parts)
