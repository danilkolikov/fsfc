from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class SelectingVectorizer(BaseEstimator, TransformerMixin):
    """
    Combines text vectorizing and feature selection
    """
    def __init__(self, vectorizer, method):
        self.vectorizer = vectorizer
        self.method = method

    def fit(self, raw_documents, y=None):
        x = self.vectorizer.fit_transform(raw_documents)
        self.method.fit(x)
        return self

    def fit_transform(self, raw_documents, y=None, **kwargs):
        x = self.vectorizer.fit_transform(raw_documents)
        return self.method.fit_transform(x)

    def transform(self, raw_documents, copy=True):
        x = self.vectorizer.transform(raw_documents, copy=copy)
        return self.method.transform(x)


class TFIDFFeatureSelector(SelectingVectorizer):
    """
    Computes TF-IDF for text corpus, extracts most relevant features based on specified algorithms
    and returns computed transformed dataset
    """
    def __init__(self, method):
        super().__init__(TfidfVectorizer(), method)
