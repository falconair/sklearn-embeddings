# sklearn-embeddings

## Overview
`sklearn-embeddings` is a Python package that integrates sentence-transformer based embeddings with scikit-learn classifiers and clustering algorithms. This allows users to leverage powerful natural language processing capabilities within the familiar scikit-learn framework.


## Installation
To install `sklearn-embeddings`, you can use pip:

```bash
pip install sklearn-embeddings
```


## Usage
Here is a simple example of how to use `sklearn-embeddings` with a scikit-learn classifier:

```python
from sklearn_embeddings import SentenceTransformerEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Sample data
documents = ["This is a sentence.", "This is another sentence."]
labels = [True, False]

# Create a pipeline with the embedding model and a classifier
pipeline = make_pipeline(
    SentenceTransformerEmbedding(), 
    # SentenceTransformerEmbedding('paraphrase-MiniLM-L6-v2'), 
    # SentenceTransformerEmbedding(SentenceTransformer('paraphrase-MiniLM-L6-v2')), 
    LogisticRegression()
    )

# Fit the model
pipeline.fit(documents, labels)

# Make predictions
predictions = pipeline.predict(["A new sentence to classify."])
```

