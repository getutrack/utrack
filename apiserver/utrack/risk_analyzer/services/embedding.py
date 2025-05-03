import logging
from typing import List, Dict, Any, Union
import re
import string
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from django.conf import settings

logger = logging.getLogger(__name__)

# Global model instance
_model = None

def _get_model():
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    return _model


def _preprocess_text(text: str) -> str:
    """Preprocess text for embedding."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
    except Exception as e:
        logger.warning(f"Error removing stopwords: {e}")
    
    return text


async def generate_embedding(
    text: Union[str, List[str]],
    preprocess: bool = True,
) -> Union[List[float], List[List[float]]]:
    """Generate embeddings for text or a list of texts."""
    model = _get_model()
    
    # Handle single text
    if isinstance(text, str):
        if preprocess:
            text = _preprocess_text(text)
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        return embedding
    
    # Handle batch of texts
    elif isinstance(text, list):
        if preprocess:
            texts = [_preprocess_text(t) for t in text]
        else:
            texts = text
        
        # Generate embeddings
        embeddings = model.encode(texts).tolist()
        return embeddings
    
    else:
        raise ValueError("Text must be a string or list of strings") 