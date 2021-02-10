from typing import Union, Optional
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


Vector = Union[pd.Series, np.ndarray]


class StringCluster(BaseEstimator, ClusterMixin):
    def __init__(self, a: Vector, b: Optional[Vector] = None, ngram_size: int = 2,
                 similarity: float = 0.8, stop_tokens: str = '[\W_]+'):
        self.a = a
        self.b = b if b else a.copy()
        self.stop_tokens = re.compile(stop_tokens)
        self.ngram_size = ngram_size
        self.similarity = similarity

    def _clean_series(self) -> "StringCluster":
        a = pd.Series(self.a).apply(self._clean_string)
        b = pd.Series(self.b).apply(self._clean_string)
        return a, b

    def _clean_string(self, string: str) -> str:
        return self.stop_tokens.sub('', string.lower())

    def _vectorize(self, X):
        vec = TfidfVectorizer(analyzer='char', ngram_range=(self.ngram_size, 4))
        return vec.fit_transform(X)

    def _get_dist_matrix(self):
        a, b = self._clean_series()
        return linear_kernel(self._vectorize(a), self._vectorize(b))

    def _get_labels(self):
        distance = self._get_dist_matrix()
        labels = np.where(distance > self.similarity, 1., 0.).argmax(1)
        return self.a[labels]


p = re.compile('[\W_]+')
s = 'Johnson & Johnson Inc.'
p.sub('', s.lower())

series = pd.Series([s]*100)
series.apply(lambda x: p.sub('', x))

series = pd.Series(
    ['Johnson & Johnson, Inc.', 'Johnson & Johnson Inc.', 'Johnson & Johnson Inc',
     'Johnson & Johnson', 'Intel Corp', 'Intel Corp.', 'Intel Corporation', 'Google',
     'Apple', 'Amazon', 'Amazon Inc']
)
c = StringCluster(series)
c._get_labels()
dist = c._get_dist_matrix()
labs = np.where(dist > .8, 1., 0.)
labs = labs.argmax(1)
c.a[labs]



c = StringCluster(series)
c._clean_series()

vec = TfidfVectorizer(analyzer='char', ngram_range=(2,2))
vec.fit(c.a)
mm = vec.transform(c.a)
m = pd.DataFrame(
    vec.transform(c.a).toarray(),
    columns=vec.vocabulary_.keys()
)
dist = linear_kernel(mm, mm, False)
labs = np.asarray(dist.argmax(1)).ravel()
c.a[labs]