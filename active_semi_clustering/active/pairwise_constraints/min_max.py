import numpy as np

from scipy.spatial.distance import cdist, pdist

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class MinMax(ExploreConsolidate):
    '''
    More intelligent selection of points. This takes longer than random partially because it propagates constraints
    '''
    def _consolidate(self, neighborhoods, X, oracle):
        '''
        Find the most ambiguous elements, i.e. those that are close to a large number of cluster centers
        '''
        n = X.shape[0]

        traversed_idx = {neigh[0] for neigh in neighborhoods}
        untraversed_idx = set(range(n)) - traversed_idx

        distances = pdist(X) # N^2 memory, no thanks.

        while True:
            try:
                traversed = X[np.array(list(traversed_idx))]

                if len(untraversed_idx) <= self.n_clusters:
                    # TODO: test
                    untraversed = X[np.array(list(untraversed_idx))]
                    mindists = cdist(untraversed, traversed, 'euclidean').max(axis=1)
                    query_point = np.argmin(mindists)
                else:
                    # Computing all the distances is expensive, so randomly sample k points.
                    index_map = np.random.choice(list(untraversed_idx), size=(self.n_clusters,), replace=False)
                    untraversed = X[index_map]
                    mindists = cdist(untraversed, traversed, 'euclidean').max(axis=1)
                    query_point = index_map[np.argmin(mindists)]

                # Query oracle with elements from each cluster 
                # TODO: sort these points intelligently
                for neighborhood in neighborhoods:
                    if oracle.query(query_point, neighborhood[0]):
                        neighborhood.append(query_point)
                        break

                traversed_idx.add(query_point)
                untraversed_idx.remove(query_point)

            except MaximumQueriesExceeded:
                break

        return neighborhoods