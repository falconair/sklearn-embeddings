# sklearn_embeddings/embedding.py
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding(BaseEstimator, TransformerMixin):
    """Transformer for generating sentence embeddings using SentenceTransformer models."""
    
    def __init__(self, model:str|SentenceTransformer='all-MiniLM-L6-v2', **kwargs):

        if isinstance(model, str):
            self.model_name = model
            self.model = SentenceTransformer(self.model_name, **kwargs) 
        elif isinstance(model, SentenceTransformer):
            self.model = model
            self.model_name = model.model_card_data.base_model # TODO: is model_card always available?
        else:
            raise ValueError("Model must be a string or a SentenceTransformer instance.")
        
        
    def fit(self, X, y=None):
        """Nothing to fit, just use `transform` to convert documents to embeddings."""
        return self
        
    def transform(self, sentences: str|list[str]|ndarray, **kwargs):
        """Transform documents to embeddings."""
        return self.model.encode(sentences, **kwargs)
    
    #TODO: get_feature_names_out
    # def get_feature_names_out(self, input_features=None):
    #     """Get the feature names for the transformer."""
    #     return [f"{self.model_name}_{i}" for i in range(self.model.get_sentence_embedding_dimension())]