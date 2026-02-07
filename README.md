# Job Description Parser & Candidate Matching API

A FastAPI application for parsing job descriptions, managing candidate profiles, and performing intelligent candidate-job matching using FAISS vector indexing and LLM evaluation.

## Features

### Job Description Management
- **FastAPI REST API**: Parse and store job descriptions
- **LLM-based Parsing**: Uses OpenRouter API to extract structured information
- **Separate Collections**: Each field type stored in dedicated MongoDB collections
- **Vector Search**: FAISS indexing for semantic similarity search

### Candidate Profile Selection System
- **Profile Management**: Store candidate profiles with leetcode, projects, resume, education
- **LeetCode Integration**: Fetches comprehensive LeetCode stats (problems solved, contest rating, badges) for LLM evaluation
- **Vector-based Matching**: FAISS similarity search - **1 Job matched against ALL Candidates**
- **Two-Stage Filtering**:
  1. **Initial FAISS Search**: Match job with all 1000+ candidates using resume, projects, education
  2. **60% Elimination Filter**: Remove bottom 60% based on similarity scores
  3. **LLM Evaluation**: Top 40% evaluated by AI using all data including LeetCode stats
- **OA Round Selection**: Automated candidate shortlisting for online assessment

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
MONGODB_URL=mongodb+srv://bhushananokar2023comp_db_user:mongo@cluster0.hs9v1d8.mongodb.net/
MONGODB_DB_NAME=jd_parser_db
COLLECTION_NAME=job_descriptions
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

3. Run the FastAPI server:
```bash
uvicorn app:app --reload --port 8001
```

Or simply:
```bash
python app.py
```

## API Endpoints

### Job Description Endpoints

#### POST /parse
Parse a job description and store in separate collections

**Request Body:**
```json
{
  "job_description": "We are looking for a Senior Python Developer...",
  "job_title": "Senior Python Developer",
  "company": "Tech Corp",
  "location": "Remote"
}
```

#### GET /jobs
List all jobs

#### GET /jobs/{job_id}
Get specific job details

### Candidate Profile Endpoints

#### POST /candidate/submit
Submit a candidate profile for evaluation

**Request Body:**
```json
{
  "leetcode_username": "john_doe",
  "projects": [
    {
      "name": "ML Pipeline",
      "description": "Built an automated ML pipeline...",
      "github_link": "https://github.com/john/ml-pipeline",
      "technologies": ["Python", "TensorFlow", "Docker"]
    },
    {
      "name": "Web Dashboard",
      "description": "Created analytics dashboard...",
      "github_link": "https://github.com/john/dashboard",
      "technologies": ["React", "FastAPI"]
    }
  ],
  "resume_text": "Software Engineer with 5 years of experience...",
  "education": [
    {
      "degree": "Bachelor of Technology",
      "field_of_study": "Computer Science",
      "institution": "ABC University",
      "graduation_year": 2020,
      "gpa": 3.7
    }
  ],
  "interview_questions": {
    "Why this company?": "I am passionate about...",
    "Career goals": "I aim to become..."
  }
}
```

**Response:**
```json
{
  "candidate_id": "candidate_abc123",
  "leetcode_username": "john_doe",
  "message": "Candidate profile submitted successfully",
  "embeddings_generated": 8,
  "next_step": "Candidates will be evaluated when a job evaluation is triggered"
}
```

#### POST /job/evaluate/{job_id}
**Evaluate ALL candidates for a specific job position**

**Process:**
1. Match the job with ALL candidates (1000+) using FAISS
2. Calculate similarity scores for each candidate
3. **Eliminate bottom 60% of candidates** (keep top 40%)
4. **LLM evaluates remaining 40%** using OpenRouter
5. Return final list of candidates for OA round

**Response:**
```json
{
  "job_id": "job_xyz789",
  "job_title": "Senior Python Developer",
  "company": "Tech Corp",
  "total_candidates_analyzed": 1000,
  "initial_matches": 1000,
  "eliminated_count": 600,
  "after_60_percent_filter": 400,
  "final_recommendations": [
    {
      "candidate_id": "candidate_abc123",
      "job_id": "job_xyz789",
      "job_title": "Senior Python Developer",
      "similarity_score": 0.92,
      "llm_evaluation": {
        "final_score": 88,
        "recommendation": "Highly Recommended",
        "strengths": ["Strong Python skills", "Relevant projects"],
        "concerns": ["Limited cloud experience"]
      },
      "final_score": 0.88,
      "recommendation": "Highly Recommended",
      "reasoning": "Excellent technical background with relevant experience...",
      "proceed_to_oa": true
    }
  ],
  "message": "Evaluation complete. 150 candidates recommended for OA round."
}
```

#### GET /candidate/{candidate_id}
Get candidate profile

#### GET /candidates
List all candidates

### System Endpoints

#### GET /health
Health check with system statistics

#### GET /stats
Detailed collection statistics

## Job Posted → Parse & Store with FAISS Embeddings
   ↓
2. Candidates Submit Profiles → Generate Embeddings → Store in MongoDB
   ↓
3. Recruiter Triggers: POST /job/evaluate/{job_id}
   ↓
4. FAISS Search → Match Job with ALL Candidates (1000+)
   ↓
5. Calculate Similarity Scores for Each Candidate
   ↓
6. Eliminate Bottom 60% → Keep Top 40%
   ↓
7. LLM Evaluation (OpenRouter) → Evaluate Remaining 400 Candidates
   ↓
8. Final Output → List of Candidates for OA Round
5. LLM Evaluation → OpenRouter GPT-4o-mini
   ↓
6. Final Recommendations → OA Round Candidates
```

## MongoDB Collections

### Job Collections
1. **jobs** - Main job information with embeddings
2. **skills** - Required and preferred skills with vectors
3. **experience** - Experience requirements with vectors
4. **education** - Education requirements with vectors
5. **responsibilities** - Job responsibilities with vectors
6. **benefits** - Benefits and perks with vectors
7. **job_details** - Job type, salary, work mode

### Candidate Collections
1. **candidates** - Main candidate profiles
2. **candidatejob description
curl -X POST "http://localhost:8001/parse" \\
  -H "Content-Type: application/json" \\
  -d '{
    "job_description": "We need a Senior Python Developer...",
    "job_title": "Senior Python Developer",
    "company": "Tech Corp"
  }'

# 2. Submit candidate profiles
curl -X POST "http://localhost:8001/candidate/submit" \\
  -H "Content-Type: application/json" \\
  -d '{
    "leetcode_username": "john_doe",
    "projects": [...],
    "resume_text": "...",
    "education": [...]
  }'

# 3. Evaluate all candidates for the job
curl -X POST "http://localhost:8001/job/evaluate/job
    "projects": [...],
    "resume_text": "...",
    "education": [...]
  }'

# 2. Evaluate candidate
curl -X POST "http://localhost:8001/candidate/evaluate/candidate_abc123"
```

## API Documentation

Visit `http://localhost:8001/docs` for interactive Swagger UI documentation
