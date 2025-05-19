# sklearn_embeddings/embedding.py
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding(BaseEstimator, TransformerMixin):
    """Transformer for generating sentence embeddings using SentenceTransformer models."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        
    def fit(self, X, y=None):
        """Nothing to fit, just use `transform` to convert documents to embeddings."""
        return self
        
    def transform(self, X):
        """Transform documents to embeddings."""
        return self.model.encode(X)