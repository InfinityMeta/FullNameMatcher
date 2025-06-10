from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class FullnamesBigramVectorizer:
    """Class for transforming russian fullnames to vectors of bigrams.
    
    Attributes:
        vectorizer: CountVectorizer instance.
    """
    def __init__(self, ngram_range: Tuple = (2, 2), vocabulary: Optional[Dict] = None) -> None:
        """Initializes FullnamesBigramVectorizer instance.
        
        Args:
            ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            vocabulary: Mapping where keys are terms and values are indices in the feature matrix.
        """

        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            vocabulary=vocabulary
        )

    def fit(self, dataset: List[str]) -> None:
        """Fits CountVectorizer on dataset of russian fullnames.
        
        Args:
            dataset: dataset containing Russian fullnames.
        """
        self.vectorizer.fit(dataset)
    
    def transform(self, dataset: List[str]) -> np.array:
        """Transforms dataset of russian fullnames to vectors of bigrams.
        
        Args:
            dataset: dataset containing Russian fullnames.

        Returns:
            Numpy array containing vectors of bigrams.
        """ 
        fullnames_vectors = self.vectorizer.transform(dataset).toarray()

        return fullnames_vectors
    
    def fit_transform(self, dataset: List[str]) -> np.array:
        """Fits CountVectorizer and transforms dataset of russian fullnames to vectors of bigrams.
        
        Args:
            dataset: dataset containing Russian fullnames.

        Returns:
            Numpy array containing vectors of bigrams.
        """ 
        fullnames_vectors = self.vectorizer.fit_transform(dataset).toarray()

        return fullnames_vectors
    