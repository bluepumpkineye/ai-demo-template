import json
import numpy as np
import faiss
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


class InMemoryVectorStore:
    """Stores vectors in memory. No database needed."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks with their embeddings."""
        vectors = np.array([c.embedding for c in chunks]).astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find the most similar chunks to a query."""
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx].content,
                    'metadata': self.chunks[idx].metadata,
                    'similarity': float(score)
                })
        return results