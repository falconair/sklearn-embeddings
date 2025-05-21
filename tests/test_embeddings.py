# tests/test_embedding.py
import pickle

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sentence_transformers import SentenceTransformer
from sklearn_embeddings import SentenceTransformerEmbedding

def test_embedding_clustering_pipeline():
    """Test that the embedding works in a scikit-learn pipeline."""

    X = ["This is a test", "Another test document", "Something completely different"]
    
    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding()),
        ('clustering', KMeans(n_clusters=2))
    ])
    
    pipeline.fit(X)
    labels = pipeline.predict(X)
    assert len(labels) == len(X)

def test_embedding_classification_pipeline():
    """Test that the embedding works in a scikit-learn pipeline for classification."""
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding()),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter

def test_default_model_name():
    """Test that the if no model is provide, the correct default is picked."""
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    sent_transformer = SentenceTransformerEmbedding()
    assert sent_transformer.model_name == 'all-MiniLM-L6-v2'

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding()),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter

def test_custom_model_name():
    """Test that the if a model is provide, it is used."""
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    sent_transformer = SentenceTransformerEmbedding(model='paraphrase-MiniLM-L6-v2')
    assert sent_transformer.model_name == 'paraphrase-MiniLM-L6-v2'

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding(model='paraphrase-MiniLM-L6-v2')),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter

def test_model():
    """Test that the model is a SentenceTransformer instance."""
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding(model)),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter

def test_pickle_unpickle():
    """Test that the embedding can be pickled and unpickled."""
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding()),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    
    # Pickle the pipeline
    pickled_pipeline = pickle.dumps(pipeline)
    
    # Unpickle the pipeline
    unpickled_pipeline = pickle.loads(pickled_pipeline)
    
    # Test that the unpickled pipeline works
    score = unpickled_pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter