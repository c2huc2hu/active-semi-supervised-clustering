import numpy as np
from collections import namedtuple

Constraints = namedtuple('Constraints', ['ml', 'cl'])

class Random:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters

        # 2-tuple containing ml and cl, which are each points (2-tuples)
        self.pairwise_constraints_ = Constraints([], [])

    def fit(self, X, oracle=None):
        n_elems = X.shape[0]

        # Quickly sample n_elems pairs of examples
        from_ = np.random.randint(n_elems, size=(oracle.max_queries_cnt))
        dist = np.random.randint(n_elems - 1, size=(oracle.max_queries_cnt)) # - 1 to avoid pairing an item with itself
        to = (from_ + dist) % n_elems
        constraints = np.vstack((from_, to)).T

        ml, cl = [], []

        for i, j in constraints:
            must_linked = oracle.query(i, j)
            if must_linked:
                ml.append((i, j))
            else:
                cl.append((i, j))

        self.pairwise_constraints_ = Constraints(ml, cl)

        return self
