from sklearn.base import BaseEstimator, TransformerMixin

class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialization code goes here
        pass

    def fit(self, X, y=None):
        # Fit the transformer based on the input data X (and optionally y).
        # Most preprocessing steps only need to fit to X.
        return self  # Return the transformer
    
    def transform(self, X, y=None):
        return X
