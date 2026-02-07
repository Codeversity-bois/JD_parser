import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
import logging
import requests

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSHandler:
    """Handler for FAISS vector indexing and similarity search"""
    
    def __init__(self):
        """Initialize FAISS index"""
        self.dimension = config.EMBEDDING_DIMENSION
        self.index = None
        self.id_mapping = []  # Maps FAISS index position to job_id
        self.index_path = config.FAISS_INDEX_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Try to load existing index, otherwise create new
        self.load_index()
        
        if self.index is None:
            self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index"""
        # Using IndexFlatL2 for exact search (can be changed to IndexIVFFlat for faster approximate search)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_mapping = []
        logger.info(f"Initialized new FAISS index with dimension {self.dimension}")
    
    def add_embedding(self, embedding: List[float], job_id: str) -> int:
        """
        Add an embedding vector to the FAISS index
        
        Args:
            embedding: Vector embedding
            job_id: Job ID associated with this embedding
            
        Returns:
            Index ID in FAISS
        """
        try:
            # Convert to numpy array
            vector = np.array([embedding], dtype=np.float32)
            
            # Normalize vector (optional, for cosine similarity)
            faiss.normalize_L2(vector)
            
            # Add to index
            self.index.add(vector)
            
            # Store mapping
            index_id = len(self.id_mapping)
            self.id_mapping.append(job_id)
            
            logger.info(f"Added embedding for job {job_id} at index {index_id}")
            return index_id
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (index_id, distance) tuples
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize query vector
            faiss.normalize_L2(query_vector)
            
            # Search
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Return results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.id_mapping):
                    results.append((int(idx), float(dist)))
            
            logger.info(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise
    
    def get_job_id(self, index_id: int) -> Optional[str]:
        """
        Get job_id from FAISS index ID
        
        Args:
            index_id: FAISS index ID
            
        Returns:
            Job ID or None
        """
        if 0 <= index_id < len(self.id_mapping):
            return self.id_mapping[index_id]
        return None
    
    def save_index(self):
        """Save FAISS index and mappings to disk"""
        try:
            # Save FAISS index
            index_file = os.path.join(self.index_path, 'faiss_index.bin')
            faiss.write_index(self.index, index_file)
            
            # Save ID mapping
            mapping_file = os.path.join(self.index_path, 'id_mapping.pkl')
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.id_mapping, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self):
        """Load FAISS index and mappings from disk"""
        try:
            index_file = os.path.join(self.index_path, 'faiss_index.bin')
            mapping_file = os.path.join(self.index_path, 'id_mapping.pkl')
            
            if os.path.exists(index_file) and os.path.exists(mapping_file):
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load ID mapping
                with open(mapping_file, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                logger.info("No existing index found")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
            self.id_mapping = []
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the index"""
        return self.index.ntotal if self.index else 0
    
    def clear_index(self):
        """Clear the FAISS index"""
        self._initialize_index()
        logger.info("Cleared FAISS index")


class EmbeddingGenerator:
    """Generate embeddings using OpenRouter API"""
    
    def __init__(self):
        """Initialize embedding generator"""
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenRouter
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Using text-embedding-3-small via OpenRouter
            url = f"{self.base_url}/embeddings"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "openai/text-embedding-3-small",
                "input": text
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            embedding = result['data'][0]['embedding']
            
            logger.info(f"Generated embedding with dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
