# tests/test_embedding.py
from sklearn.pipeline import Pipeline
from src.embeddings import SentenceTransformerEmbedding

def test_embedding_clustering_pipeline():
    """Test that the embedding works in a scikit-learn pipeline."""
    from sklearn.cluster import KMeans

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
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X = ["This is a test", "Another test document", "Something completely different"]
    y = [True, False, True]

    pipeline = Pipeline([
        ('embeddings', SentenceTransformerEmbedding()),
        ('classification', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    assert score > 0.1  # Make sure the parts fit together, the result doesn't matter

def test_pickle_unpickle():
    """Test that the embedding can be pickled and unpickled."""
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
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