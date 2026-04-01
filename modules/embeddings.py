from sentence_transformers import SentenceTransformer
from typing import List

# Load model once (stays in memory)
_model = None

def get_model():
    global _model
    if _model is None:
        print("  Loading embedding model (first time takes 1-2 minutes)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✓ Model loaded!")
    return _model

def embed_text(text: str) -> List[float]:
    """Turn text into numbers (a vector)."""
    model = get_model()
    return model.encode(text).tolist()

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Turn many texts into vectors at once."""
    model = get_model()
    return [e.tolist() for e in model.encode(texts, show_progress_bar=True)]