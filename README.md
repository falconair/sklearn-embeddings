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
import joblib

from sklearn_embeddings import SentenceTransformerEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Sample data
documents = ["The food was great.", "Not expensive and good service", "Not worth the money", "I've had better"]

# Labels
is_positive = [True, True, False, False]

# Create a pipeline with the embedding model and a classifier
pipeline = make_pipeline(
    SentenceTransformerEmbedding(), 
    # SentenceTransformerEmbedding('paraphrase-MiniLM-L6-v2'), 
    # SentenceTransformerEmbedding('/my/local/folder/paraphrase-MiniLM-L6-v2'), 
    # SentenceTransformerEmbedding(SentenceTransformer('paraphrase-MiniLM-L6-v2')), 
    LogisticRegression()
    )

# Fit the model
pipeline.fit(documents, is_positive)

# Make predictions
predictions = pipeline.predict(["So delicious!", "Not for me"])

# Write the whole pipeline to disk
joblib.dump(pipeline, 'model.joblib')

```

Perhaps the greatest benefit of this library is that it allows you to use Scikit-learn pipelines to combine encoding and labeling in a single function call.

```python
import joblib

model = joblib.load('model.joblib')

# Use the loaded pipeline as a simple model, it takes care of sentence-transformer encoding for you!
model.predict(["This is a sentence"])
```